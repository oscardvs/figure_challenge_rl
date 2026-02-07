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
    is_gauntlet: bool = False,
    curriculum_max_steps: int = 30,
) -> dict:
    """Generate a single rollout using the local model.

    Returns dict with: messages, actions, reward, success, old_log_probs,
    steps_completed.

    old_log_probs is a list of per-action entries, each containing:
      - token_ids: list[int] — generated token IDs for this action
      - token_log_probs: list[float] — per-token log-probs under the old policy
      - prompt_length: int — length of the prompt prefix (for re-computing)

    When is_gauntlet=True, reward is fractional (steps_completed / 30.0)
    instead of binary.
    """
    import torch

    from src.agent.prompts import parse_action_from_response

    obs_text, info = env.reset()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{task_prompt}\n\n[Step 0] Current page state:\n{obs_text}"},
    ]
    actions = []
    old_log_probs = []
    total_reward = 0.0
    steps_completed = 0

    for step in range(max_steps):
        # Tokenize conversation.
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        prompt_length = inputs["input_ids"].shape[1]

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

        generated_ids = outputs.sequences[0][prompt_length:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute per-token log probabilities under the old (current) policy.
        scores = outputs.scores  # tuple of (vocab_size,) logit tensors
        token_log_probs = []
        for i, score in enumerate(scores):
            if i < len(generated_ids):
                token_id = generated_ids[i].item()
                log_prob = torch.nn.functional.log_softmax(score[0], dim=-1)
                token_log_probs.append(log_prob[token_id].item())

        old_log_probs.append({
            "token_ids": generated_ids.tolist(),
            "token_log_probs": token_log_probs,
            "prompt_length": prompt_length,
        })

        # Parse action from response.
        action = parse_action_from_response(response_text)
        actions.append(action)

        # Execute in environment.
        try:
            obs_text, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            logger.warning(f"Rollout step error: {e}")
            break

        # Track cumulative reward.
        if is_gauntlet:
            task_info = info.get("task_info", info)
            steps_completed = task_info.get("steps_completed", steps_completed)
            total_reward = steps_completed / 30.0
            # Curriculum: stop after reaching max steps for this epoch.
            if steps_completed >= curriculum_max_steps:
                terminated = True
        else:
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
        "old_log_probs": old_log_probs,
        "steps_completed": steps_completed,
    }


def compute_grpo_loss(
    model,
    tokenizer,
    group_rollouts: list[dict],
    clip_epsilon: float = 0.2,
    kl_coefficient: float = 0.001,
):
    """Compute M-GRPO loss for a group of rollouts.

    Proper Group Relative Policy Optimization:
    - Compute group-relative advantages (no value network needed).
    - Re-compute per-token log-probs under the *current* policy via forward pass.
    - Compute per-token importance ratios against stored old log-probs.
    - Apply clipped surrogate objective + KL penalty.
    """
    import torch

    rewards = [r["reward"] for r in group_rollouts]
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards) + 1e-8

    total_loss = torch.tensor(0.0, device=model.device)
    num_tokens = 0

    for rollout in group_rollouts:
        # Group-relative advantage (trajectory-level).
        advantage = (rollout["reward"] - mean_reward) / std_reward

        if abs(advantage) < 1e-8:
            continue

        adv_tensor = torch.tensor(advantage, dtype=torch.float32, device=model.device)

        # For each action turn in the rollout, compute per-token loss.
        for msg_idx, old_lp_data in enumerate(rollout["old_log_probs"]):
            # Reconstruct the prompt (messages up to this action turn).
            messages_so_far = rollout["messages"][:2 + msg_idx * 2]
            if not messages_so_far:
                continue

            old_token_ids = old_lp_data["token_ids"]
            old_token_log_probs = old_lp_data["token_log_probs"]
            if not old_token_ids or not old_token_log_probs:
                continue

            # Build the full sequence: prompt + generated tokens.
            input_text = tokenizer.apply_chat_template(
                messages_so_far, tokenize=False, add_generation_prompt=True,
            )
            prompt_ids = tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=8192,
            )["input_ids"].to(model.device)

            gen_ids = torch.tensor([old_token_ids], device=model.device)
            full_ids = torch.cat([prompt_ids, gen_ids], dim=1)

            # Forward pass to get current policy logits.
            outputs = model(input_ids=full_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

            # Extract logits at positions corresponding to generated tokens.
            # logits[t] predicts token[t+1], so logits at prompt_len-1 .. prompt_len+gen_len-2
            # predict tokens at prompt_len .. prompt_len+gen_len-1 (the generated tokens).
            prompt_len = prompt_ids.shape[1]
            gen_len = min(len(old_token_ids), len(old_token_log_probs))
            if gen_len == 0:
                continue

            # Slice logits for the generated span.
            gen_logits = logits[0, prompt_len - 1 : prompt_len - 1 + gen_len, :]
            gen_log_probs = torch.nn.functional.log_softmax(gen_logits, dim=-1)

            # Gather log-probs for the actual generated tokens.
            gen_token_tensor = torch.tensor(
                old_token_ids[:gen_len], device=model.device
            )
            current_token_log_probs = gen_log_probs.gather(
                -1, gen_token_tensor.unsqueeze(-1)
            ).squeeze(-1)  # (gen_len,)

            # Old log-probs (detached, from rollout generation).
            old_lp_tensor = torch.tensor(
                old_token_log_probs[:gen_len],
                dtype=torch.float32, device=model.device,
            )

            # Per-token importance ratio with safety clamping on log-space
            # to prevent exp() overflow when policies diverge significantly.
            log_ratio = current_token_log_probs - old_lp_tensor
            log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

            # Clipped surrogate objective (per-token, same advantage for all tokens
            # in a trajectory since rewards are trajectory-level).
            surr1 = ratio * adv_tensor
            surr2 = clipped_ratio * adv_tensor
            token_loss = -torch.min(surr1, surr2)

            # KL penalty per token.
            kl_penalty = kl_coefficient * (current_token_log_probs - old_lp_tensor)
            token_loss = token_loss + kl_penalty

            total_loss = total_loss + token_loss.sum()
            num_tokens += gen_len

    if num_tokens > 0:
        total_loss = total_loss / num_tokens

    return total_loss


def train(
    model_config: dict,
    dpo_model_dir: str,
    output_dir: str,
    num_episodes: int = 500,
    group_size: int = 4,
):
    """Run M-GRPO training loop.

    Supports two rollout modes:
    - "gauntlet": Uses GauntletEnv with randomized challenges. Reward is
      fractional (steps_completed / 30.0). Includes curriculum learning.
    - "step": Original per-step mode using StepEnv with binary reward.
    """
    import torch

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("Unsloth not installed.")
        raise

    from src.agent.prompts import (
        format_gauntlet_task_prompt,
        format_pure_agent_system_prompt,
        format_system_prompt,
        format_task_prompt,
    )
    from src.environment.action_space import get_action_description

    grpo_cfg = model_config["grpo"]
    rollout_mode = grpo_cfg.get("rollout_mode", "step")
    curriculum_max_steps = grpo_cfg.get("curriculum_max_steps", 5)
    is_gauntlet = rollout_mode == "gauntlet"

    action_desc = get_action_description()
    if is_gauntlet:
        system_prompt = format_pure_agent_system_prompt(action_desc)
    else:
        system_prompt = format_system_prompt(action_desc)

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
    recent_rewards: list[float] = []  # For curriculum adjustment.

    logger.info(
        f"Starting M-GRPO training: {num_episodes} episodes, "
        f"group_size={group_size}, mode={rollout_mode}"
    )

    while episodes_done < num_episodes:
        if is_gauntlet:
            from src.environment.browser_env import GauntletEnv
            env = GauntletEnv(
                headless=True,
                max_actions_per_step=15,
            )
            task_prompt = format_gauntlet_task_prompt(current_step=1)
        else:
            from src.environment.browser_env import StepEnv
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
                    is_gauntlet=is_gauntlet,
                    curriculum_max_steps=curriculum_max_steps,
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
            recent_rewards.extend(group_rewards)

            if episodes_done % 20 == 0:
                logger.info(
                    f"Episode {episodes_done}/{num_episodes}: "
                    f"loss={loss.item():.4f}, "
                    f"group_reward={np.mean(group_rewards):.2f}, "
                    f"success_rate={successes/episodes_done:.1%}"
                    + (f", curriculum_steps={curriculum_max_steps}" if is_gauntlet else "")
                )

            # Curriculum adjustment: increase max steps when success rate is high.
            if is_gauntlet and len(recent_rewards) >= 20:
                avg_reward = np.mean(recent_rewards[-20:])
                # If solving >60% of curriculum steps, expand.
                if avg_reward > curriculum_max_steps * 0.6 / 30.0:
                    curriculum_max_steps = min(30, curriculum_max_steps + 2)
                    logger.info(f"Curriculum expanded to {curriculum_max_steps} steps")
                recent_rewards = recent_rewards[-20:]

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
