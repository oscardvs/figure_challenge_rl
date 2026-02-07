#!/usr/bin/env python3
"""Stage 1: Supervised Fine-Tuning on successful MCTS trajectories.

Trains Qwen2.5-3B-Instruct with QLoRA on expert trajectories collected
during MCTS data collection. Each training example is a multi-turn
conversation: system prompt → task → observation → action → ...

Usage:
    python -m src.training.sft_train
    python -m src.training.sft_train --data-dir data/trajectories --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_trajectories(data_dir: Path) -> list[dict]:
    """Load successful trajectories from collected data."""
    trajectories = []
    for path in sorted(data_dir.glob("step_*.json")):
        with open(path) as f:
            step_trajs = json.load(f)
        for traj in step_trajs:
            if traj.get("success", False):
                trajectories.append(traj)
    logger.info(f"Loaded {len(trajectories)} successful trajectories")
    return trajectories


def format_as_conversations(
    trajectories: list[dict],
    system_prompt: str,
) -> list[dict]:
    """Convert trajectories into chat-format training data.

    Each trajectory becomes a multi-turn conversation:
    [system, user(task+obs), assistant(action), user(obs), assistant(action), ...]
    """
    conversations = []

    for traj in trajectories:
        messages = [{"role": "system", "content": system_prompt}]

        for i, step in enumerate(traj["steps"]):
            # User turn: observation.
            if i == 0:
                user_content = (
                    f"Complete step {traj['step_number']}.\n\n"
                    f"[Action {step['step_index']}] Current page state:\n"
                    f"{step['obs_text']}"
                )
            else:
                user_content = (
                    f"[Action {step['step_index']}] Current page state:\n"
                    f"{step['obs_text']}"
                )
            messages.append({"role": "user", "content": user_content})

            # Assistant turn: reasoning + action.
            reasoning = step.get("reasoning", "")
            action = step["action"]
            if reasoning:
                assistant_content = f"<think>\n{reasoning}\n</think>\n{action}"
            else:
                assistant_content = action
            messages.append({"role": "assistant", "content": assistant_content})

        conversations.append({"messages": messages})

    logger.info(f"Formatted {len(conversations)} conversations")
    return conversations


def train(
    conversations: list[dict],
    model_config: dict,
    output_dir: str,
    num_epochs: int = 3,
):
    """Run SFT training with Unsloth + QLoRA."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error(
            "Unsloth not installed. Install with: "
            "pip install unsloth"
        )
        raise

    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    sft_cfg = model_config["sft"]
    qlora_cfg = model_config["qlora"]

    # Load model with QLoRA.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config["base_model"]["quantized_name"],
        max_seq_length=sft_cfg["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=qlora_cfg["r"],
        lora_alpha=qlora_cfg["alpha"],
        lora_dropout=qlora_cfg["dropout"],
        target_modules=qlora_cfg["target_modules"],
        use_gradient_checkpointing=sft_cfg["gradient_checkpointing"],
    )

    # Create dataset.
    dataset = Dataset.from_list(conversations)

    # Training config.
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=sft_cfg["per_device_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        num_train_epochs=num_epochs,
        warmup_ratio=sft_cfg["warmup_ratio"],
        bf16=sft_cfg["bf16"],
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=sft_cfg["max_seq_length"],
        dataset_text_field=None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    # Save adapter weights.
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"SFT model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1: SFT Training")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "trajectories"))
    parser.add_argument("--expert-dir", default=str(PROJECT_ROOT / "data" / "expert_trajectories"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "models" / "sft"))
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    with open(PROJECT_ROOT / "config" / "model_config.yaml") as f:
        model_config = yaml.safe_load(f)

    from src.agent.prompts import (
        format_pure_agent_system_prompt,
        format_system_prompt,
    )
    from src.environment.action_space import get_action_description

    action_desc = get_action_description()
    system_prompt = format_system_prompt(action_desc)
    pure_system_prompt = format_pure_agent_system_prompt(action_desc)

    # Load MCTS trajectories (use original system prompt).
    mcts_trajectories = load_trajectories(Path(args.data_dir))
    conversations = format_as_conversations(mcts_trajectories, system_prompt)

    # Load expert trajectories (use pure agent system prompt).
    expert_dir = Path(args.expert_dir)
    if expert_dir.exists():
        expert_trajectories = load_trajectories(expert_dir)
        expert_conversations = format_as_conversations(
            expert_trajectories, pure_system_prompt
        )
        conversations.extend(expert_conversations)
        logger.info(
            f"Combined: {len(mcts_trajectories)} MCTS + "
            f"{len(expert_trajectories)} expert trajectories"
        )

    if not conversations:
        logger.error("No successful trajectories found. Run data collection first.")
        sys.exit(1)

    train(conversations, model_config, args.output_dir, args.epochs)


if __name__ == "__main__":
    main()
