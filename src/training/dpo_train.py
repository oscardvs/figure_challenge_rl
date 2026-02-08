#!/usr/bin/env python3
"""Stage 2: DPO training on MCTS preference pairs.

Uses TRL's DPOTrainer with the adapter-as-reference trick
(ref_model=None + peft_config) to avoid loading a separate reference
model, keeping VRAM at ~10 GB for Qwen3-4B with 2048 max_seq_length.

Usage:
    python -m src.training.dpo_train
    python -m src.training.dpo_train --sft-model models/sft --epochs 1
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


def load_preference_pairs(data_dir: Path) -> list[dict]:
    """Load preference pairs from collected data.

    Globs both challenge_*.json (MCTS pairs) and step_*.json (expert pairs).
    """
    pairs = []
    for pattern in ("challenge_*.json", "step_*.json"):
        for path in sorted(data_dir.glob(pattern)):
            with open(path) as f:
                file_pairs = json.load(f)
            pairs.extend(file_pairs)
    logger.info(f"Loaded {len(pairs)} preference pairs from {data_dir}")
    return pairs


def format_dpo_dataset(
    pairs: list[dict],
    system_prompt: str,
) -> list[dict]:
    """Convert preference pairs to DPO training format.

    Each example needs: prompt, chosen, rejected.
    """
    examples = []

    for pair in pairs:
        # Strip the HTML Content section — it's redundant with the AXTree
        # and won't fit in the 2048 token budget anyway.  SFT (at 4096
        # max_seq_length) still uses the full observation.
        obs = pair["obs_text"]
        if "\nHTML Content:\n" in obs:
            obs = obs.split("\nHTML Content:\n", 1)[0]

        prompt = (
            f"{system_prompt}\n\n"
            f"[Step {pair['step_index']}] Current page state:\n"
            f"{obs}"
        )

        examples.append({
            "prompt": prompt,
            "chosen": pair["chosen_action"],
            "rejected": pair["rejected_action"],
        })

    logger.info(f"Formatted {len(examples)} DPO examples")
    return examples


def train(
    examples: list[dict],
    model_config: dict,
    sft_model_dir: str,
    output_dir: str,
    num_epochs: int = 1,
):
    """Run DPO training with adapter-as-reference.

    With compact observations (no HTML snippet, pruned AXTree) and a short
    DPO system prompt, sequences fit within 2048 tokens.  At this length
    the dual forward pass (policy + ref) uses ~10 GB VRAM — no precompute
    or memory tricks needed on a 16 GB GPU.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("Unsloth not installed. Install with: pip install unsloth")
        raise

    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    dpo_cfg = model_config["dpo"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=sft_model_dir,
        max_seq_length=dpo_cfg["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )

    FastLanguageModel.for_training(model)

    dataset = Dataset.from_list(examples).shuffle(seed=42)

    training_args = DPOConfig(
        output_dir=output_dir,
        beta=dpo_cfg["beta"],
        per_device_train_batch_size=dpo_cfg["per_device_batch_size"],
        gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
        learning_rate=dpo_cfg["learning_rate"],
        num_train_epochs=num_epochs,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        max_length=dpo_cfg["max_seq_length"],
        max_prompt_length=dpo_cfg["max_seq_length"] - 512,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    logger.info("Starting DPO training...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"DPO model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: DPO Training")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "preference_pairs"))
    parser.add_argument("--expert-pairs-dir", default=str(PROJECT_ROOT / "data" / "expert_preference_pairs"))
    parser.add_argument("--sft-model", default=str(PROJECT_ROOT / "models" / "sft"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "models" / "dpo"))
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    with open(PROJECT_ROOT / "config" / "model_config.yaml") as f:
        model_config = yaml.safe_load(f)

    from src.agent.prompts import DPO_SYSTEM_PROMPT
    system_prompt = DPO_SYSTEM_PROMPT

    # Load MCTS preference pairs.
    pairs = load_preference_pairs(Path(args.data_dir))

    # Load expert preference pairs.
    expert_dir = Path(args.expert_pairs_dir)
    if expert_dir.exists():
        expert_pairs = load_preference_pairs(expert_dir)
        pairs.extend(expert_pairs)
        logger.info(f"Combined: {len(pairs)} total preference pairs")

    if not pairs:
        logger.error("No preference pairs found. Run data collection first.")
        sys.exit(1)

    examples = format_dpo_dataset(pairs, system_prompt)
    train(examples, model_config, args.sft_model, args.output_dir, args.epochs)


if __name__ == "__main__":
    main()
