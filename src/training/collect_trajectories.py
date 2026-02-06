#!/usr/bin/env python3
"""MCTS trajectory collection script.

Runs MCTS on all 30 steps using an API model (Claude, GPT-4o, or Gemini),
collecting trajectories, preference pairs, and step-level data for
downstream SFT, DPO, and M-GRPO training.

NOTE: The challenge is a sequential SPA. StepEnv can only directly reach
step 1 (home → START → step1). For steps > 1, you would need to first
solve prior steps. This collector currently works best for step 1 or
requires a mechanism to replay prior solutions.

Usage:
    python -m src.training.collect_trajectories
    python -m src.training.collect_trajectories --steps 1 2 3
    python -m src.training.collect_trajectories --provider google --model gemini-3.0-flash
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from src.agent.mcts import MCTSSearch
from src.agent.policy import LLMPolicy
from src.agent.prompts import format_task_prompt
from src.environment.action_space import get_action_description
from src.environment.browser_env import StepEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "challenge_config.yaml"
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.yaml"
DATA_DIR = PROJECT_ROOT / "data"


def load_configs():
    with open(CONFIG_PATH) as f:
        challenge_config = yaml.safe_load(f)
    with open(MODEL_CONFIG_PATH) as f:
        model_config = yaml.safe_load(f)
    return challenge_config, model_config


def collect_for_step(
    step_number: int,
    policy: LLMPolicy,
    challenge_config: dict,
    mcts_config: dict,
) -> dict:
    """Run MCTS on a single step and return collected data."""
    defaults = challenge_config["defaults"]

    env = StepEnv(
        step_number=step_number,
        max_actions=defaults["max_actions_per_step"],
        headless=defaults["headless"],
    )

    task_prompt = format_task_prompt(step_number)

    searcher = MCTSSearch(
        env=env,
        policy=policy,
        task_prompt=task_prompt,
        num_iterations=mcts_config["iterations_per_step"],
        candidates_per_node=mcts_config["candidates_per_node"],
        exploration_constant=mcts_config["exploration_constant"],
        q_blend_alpha=mcts_config["q_blend_alpha"],
        min_q_diff_for_dpo=mcts_config["min_q_diff_for_dpo"],
    )

    try:
        searcher.search()
        return searcher.get_results()
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Collect MCTS trajectories")
    parser.add_argument(
        "--steps", nargs="+", type=int,
        help="Specific step numbers to collect (default: all 1-30)",
    )
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "google"])
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    challenge_config, model_config = load_configs()
    mcts_config = challenge_config["mcts"]

    # Determine provider and model.
    provider = args.provider
    if args.model:
        model = args.model
    else:
        api_cfg = model_config["api_models"]
        if provider == "anthropic":
            model = api_cfg["primary"]
        elif provider == "google":
            model = api_cfg.get("google", "gemini-3-flash-preview")
        else:
            model = api_cfg["fallback"]

    action_desc = get_action_description()

    policy = LLMPolicy(
        provider=provider,
        model=model,
        temperature=model_config["api_models"]["temperature"],
        max_tokens=model_config["api_models"]["max_tokens"],
        action_description=action_desc,
    )

    steps = args.steps or list(range(1, challenge_config["num_steps"] + 1))
    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR
    traj_dir = output_dir / "trajectories"
    pref_dir = output_dir / "preference_pairs"
    traj_dir.mkdir(parents=True, exist_ok=True)
    pref_dir.mkdir(parents=True, exist_ok=True)

    all_trajectories = []
    all_preferences = []
    all_stats = {
        "total_trajectories": 0,
        "successful_trajectories": 0,
        "total_steps": 0,
        "total_preference_pairs": 0,
    }

    start_time = time.time()

    for step_num in steps:
        logger.info(f"{'='*40} Step {step_num} {'='*40}")
        try:
            results = collect_for_step(step_num, policy, challenge_config, mcts_config)

            # Save per-step data.
            with open(traj_dir / f"step_{step_num:02d}.json", "w") as f:
                json.dump(results["trajectories"], f, indent=2)

            with open(pref_dir / f"step_{step_num:02d}.json", "w") as f:
                json.dump(results["preference_pairs"], f, indent=2)

            all_trajectories.extend(results["trajectories"])
            all_preferences.extend(results["preference_pairs"])

            stats = results["stats"]
            all_stats["total_trajectories"] += stats["total_trajectories"]
            all_stats["successful_trajectories"] += stats["successful_trajectories"]
            all_stats["total_steps"] += stats["total_steps"]
            all_stats["total_preference_pairs"] += stats["total_preference_pairs"]

            logger.info(
                f"Step {step_num}: {stats['successful_trajectories']}/"
                f"{stats['total_trajectories']} successful, "
                f"{stats['total_steps']} actions, "
                f"{stats['total_preference_pairs']} pref pairs"
            )

        except Exception as e:
            logger.error(f"Step {step_num} failed: {e}", exc_info=True)

    elapsed = time.time() - start_time

    # Save aggregate data.
    all_stats["elapsed_seconds"] = elapsed
    all_stats["api_tokens"] = policy.total_tokens

    with open(output_dir / "collection_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Collection complete in {elapsed:.0f}s")
    logger.info(f"Trajectories: {all_stats['total_trajectories']} "
                f"({all_stats['successful_trajectories']} successful)")
    logger.info(f"Actions: {all_stats['total_steps']}")
    logger.info(f"Preference pairs: {all_stats['total_preference_pairs']}")
    logger.info(f"API tokens: {policy.total_tokens}")
    logger.info(f"Data saved to {output_dir}")


if __name__ == "__main__":
    main()
