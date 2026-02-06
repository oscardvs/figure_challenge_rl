#!/usr/bin/env python3
"""Stage 3: M-GRPO training with live browser rollouts.

Implements Modified Group Relative Policy Optimization following
WebAgent-R1. The model generates G trajectories per challenge, receives
binary success/failure rewards, computes group-relative advantages
(no value network needed), and updates with clipped importance-weighted loss.

Key single-GPU adaptation: alternates between rollout generation (INT4
via the model in eval mode) and gradient updates (QLoRA training mode).

Usage:
    python -m src.training.grpo_train
    python -m src.training.grpo_train --dpo-model models/dpo --episodes 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def generate_rollout(
    model,
    tokenizer,
    env,
    system_prompt: str,
    task_prompt: str,
    max_steps: int = 25,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
) -> dict:
    """Generate a single rollout using the local model.

    Returns dict with: messages, actions, reward, success, log_probs.
    """
    import torch

    obs_text, _ = env.reset()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{task_prompt}\n\n[Step 0] Current page state:\n{obs_text}"},
    ]
    actions = []
    log_probs_list = []
    total_reward = 0.0

    for step in range(max_steps):
        # Tokenize conversation.
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # Generate action with log probs.
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute log probabilities of generated tokens.
        scores = outputs.scores  # tuple of (vocab_size,) tensors
        step_log_probs = []
        for i, score in enumerate(scores):
            if i < len(generated_ids):
                token_id = generated_ids[i]
                log_prob = torch.nn.functional.log_softmax(score[0], dim=-1)
                step_log_probs.append(log_prob[token_id].item())
        log_probs_list.append(sum(step_log_probs))

        # Parse action from response.
        from src.agent.prompts import parse_action_from_response
        action = parse_action_from_response(response_text)
        actions.append(action)

        # Execute in environment.
        try:
            obs_text, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            logger.warning(f"Rollout step error: {e}")
            break

        total_reward = max(total_reward, reward)

        # Update messages.
        messages.append({"role": "assistant", "content": response_text})
        messages.append({
            "role": "user",
            "content": f"[Step {step + 1}] Current page state:\n{obs_text}",
        })

        if terminated or truncated:
            break

    return {
        "messages": messages,
        "actions": actions,
        "reward": total_reward,
        "success": total_reward > 0,
        "log_probs": log_probs_list,
    }


def compute_grpo_loss(
    model,
    tokenizer,
    group_rollouts: list[dict],
    clip_epsilon: float = 0.2,
    kl_coefficient: float = 0.001,
):
    """Compute M-GRPO loss for a group of rollouts.

    Group Relative Policy Optimization:
    - Compute advantages relative to group mean (no value network).
    - Clip importance weights.
    - Add KL penalty against initial policy.
    """
    import torch

    rewards = [r["reward"] for r in group_rollouts]
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards) + 1e-8

    total_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
    num_terms = 0

    for rollout in group_rollouts:
        # Group-relative advantage.
        advantage = (rollout["reward"] - mean_reward) / std_reward

        if advantage == 0:
            continue

        # For each action in the rollout, compute the policy gradient.
        for msg_idx, action_log_prob_old in enumerate(rollout["log_probs"]):
            # Re-compute log prob under current policy.
            # (In practice, we'd re-tokenize and forward pass here.)
            # For simplicity, approximate with stored log probs + gradient.
            messages_so_far = rollout["messages"][:2 + msg_idx * 2]
            if not messages_so_far:
                continue

            input_text = tokenizer.apply_chat_template(
                messages_so_far, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=8192,
            ).to(model.device)

            # Get current policy log prob.
            assistant_msg = rollout["messages"][2 + msg_idx * 2] if (2 + msg_idx * 2) < len(rollout["messages"]) else None
            if not assistant_msg or assistant_msg["role"] != "assistant":
                continue

            target_text = assistant_msg["content"]
            target_ids = tokenizer(target_text, return_tensors="pt")["input_ids"].to(model.device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            log_prob_new = -outputs.loss  # Approximate; negative NLL.

            # Importance ratio.
            ratio = torch.exp(log_prob_new - action_log_prob_old)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

            # Clipped surrogate loss.
            adv_tensor = torch.tensor(advantage, device=model.device)
            surr1 = ratio * adv_tensor
            surr2 = clipped_ratio * adv_tensor
            loss_term = -torch.min(surr1, surr2)

            # KL penalty (approximate).
            kl_term = kl_coefficient * (log_prob_new - action_log_prob_old) ** 2
            total_loss = total_loss + loss_term + kl_term
            num_terms += 1

    if num_terms > 0:
        total_loss = total_loss / num_terms

    return total_loss


def train(
    model_config: dict,
    dpo_model_dir: str,
    output_dir: str,
    num_episodes: int = 500,
    group_size: int = 4,
):
    """Run M-GRPO training loop."""
    import torch

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("Unsloth not installed.")
        raise

    from src.agent.prompts import format_system_prompt, format_task_prompt
    from src.environment.action_space import get_action_description
    from src.environment.browser_env import StepEnv

    grpo_cfg = model_config["grpo"]

    system_prompt = format_system_prompt(get_action_description())

    # Load DPO model.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=dpo_model_dir,
        max_seq_length=grpo_cfg["max_context_length"],
        load_in_4bit=True,
        dtype=None,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-6,
        weight_decay=0.01,
    )

    challenges = list(range(1, 31))
    episodes_done = 0
    successes = 0

    logger.info(f"Starting M-GRPO training: {num_episodes} episodes, group_size={group_size}")

    while episodes_done < num_episodes:
        # Sample a random step.
        step_number = challenges[episodes_done % len(challenges)]
        task_prompt = format_task_prompt(step_number)

        env = StepEnv(
            step_number=step_number,
            max_actions=25,
            headless=True,
        )

        try:
            # Generate G rollouts (group).
            model.eval()
            group_rollouts = []
            for g in range(group_size):
                rollout = generate_rollout(
                    model, tokenizer, env,
                    system_prompt=system_prompt,
                    task_prompt=task_prompt,
                    max_new_tokens=grpo_cfg["max_new_tokens"],
                    temperature=grpo_cfg["sampling_temperature"],
                )
                group_rollouts.append(rollout)
                if rollout["success"]:
                    successes += 1

            # Gradient update.
            model.train()
            optimizer.zero_grad()
            loss = compute_grpo_loss(
                model, tokenizer, group_rollouts,
                clip_epsilon=grpo_cfg["clip_epsilon"],
                kl_coefficient=grpo_cfg["kl_coefficient"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            episodes_done += group_size
            group_rewards = [r["reward"] for r in group_rollouts]

            if episodes_done % 20 == 0:
                logger.info(
                    f"Episode {episodes_done}/{num_episodes}: "
                    f"loss={loss.item():.4f}, "
                    f"group_reward={np.mean(group_rewards):.2f}, "
                    f"success_rate={successes/episodes_done:.1%}"
                )

        except Exception as e:
            logger.error(f"Episode error: {e}", exc_info=True)
        finally:
            env.close()

        # Periodic checkpoint.
        if episodes_done % 100 == 0:
            ckpt_dir = f"{output_dir}/checkpoint-{episodes_done}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"Saved checkpoint to {ckpt_dir}")

    # Save final model.
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"M-GRPO training complete. Final model saved to {output_dir}")
    logger.info(f"Final success rate: {successes/episodes_done:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Stage 3: M-GRPO Training")
    parser.add_argument("--dpo-model", default=str(PROJECT_ROOT / "models" / "dpo"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "models" / "grpo"))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=4)
    args = parser.parse_args()

    with open(PROJECT_ROOT / "config" / "model_config.yaml") as f:
        model_config = yaml.safe_load(f)

    train(
        model_config,
        args.dpo_model,
        args.output_dir,
        num_episodes=args.episodes,
        group_size=args.group_size,
    )


if __name__ == "__main__":
    main()
