#!/usr/bin/env python3
"""Main entry point: solve all 30 steps of the gauntlet.

The challenge is a sequential 30-step gauntlet (not 30 independent challenges).
The agent must solve each step in order to advance to the next.

Usage:
    python -m src.runner.solve_all                    # API model, full gauntlet
    python -m src.runner.solve_all --mode api          # Explicit API mode
    python -m src.runner.solve_all --provider openai   # Use GPT-4o instead
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from src.agent.policy import LLMPolicy
from src.agent.prompts import format_gauntlet_task_prompt
from src.environment.action_space import get_action_description
from src.environment.browser_env import GauntletEnv
from src.runner.metrics import ChallengeMetrics, RunMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_configs():
    with open(PROJECT_ROOT / "config" / "challenge_config.yaml") as f:
        challenge_config = yaml.safe_load(f)
    with open(PROJECT_ROOT / "config" / "model_config.yaml") as f:
        model_config = yaml.safe_load(f)
    return challenge_config, model_config


def solve_gauntlet_api(
    policy: LLMPolicy,
    challenge_config: dict,
) -> RunMetrics:
    """Solve the full 30-step gauntlet using the API model."""
    defaults = challenge_config["defaults"]
    max_total = defaults.get("max_actions_total", 500)

    env = GauntletEnv(
        headless=defaults["headless"],
        max_actions_per_step=defaults.get("max_actions_per_step", 30),
    )

    run_metrics = RunMetrics()
    run_metrics.start()

    try:
        obs_text, info = env.reset()
        task_info = info.get("task_info", info)
        current_step = task_info.get("current_step", 1)
        action_history: list[str] = []
        step_start_time = time.time()
        step_actions = 0

        task_prompt = format_gauntlet_task_prompt(current_step)

        for action_idx in range(max_total):
            action, reasoning = policy.select_action(
                obs_text=obs_text,
                task_prompt=task_prompt,
                action_history=action_history[-10:],  # Keep recent history only.
                step=action_idx,
            )

            logger.info(f"[Step {current_step} | Action {step_actions + 1}] {action}")

            obs_text, reward, terminated, truncated, info = env.step(action)
            action_history.append(action)
            step_actions += 1

            task_info = info.get("task_info", info)
            new_step = task_info.get("current_step", current_step)

            # Log errors from the browser.
            last_error = obs_text.split("Action error: ")[-1].split("\n")[0] if "Action error:" in obs_text else ""
            if last_error:
                logger.warning(f"  -> Error: {last_error}")

            # Detect step advancement.
            if new_step > current_step:
                elapsed = time.time() - step_start_time
                logger.info(
                    f"Step {current_step} SOLVED in {step_actions} actions, "
                    f"{elapsed:.1f}s"
                )
                run_metrics.add_challenge(ChallengeMetrics(
                    challenge_id=current_step,
                    success=True,
                    steps=step_actions,
                    elapsed_seconds=elapsed,
                ))

                current_step = new_step
                task_prompt = format_gauntlet_task_prompt(current_step)
                step_start_time = time.time()
                step_actions = 0

            if terminated or truncated:
                break

        # Record the last step if it wasn't completed.
        if step_actions > 0 and (not terminated or current_step <= 30):
            run_metrics.add_challenge(ChallengeMetrics(
                challenge_id=current_step,
                success=False,
                steps=step_actions,
                elapsed_seconds=time.time() - step_start_time,
            ))

    except Exception as e:
        logger.error(f"Gauntlet error: {e}", exc_info=True)
    finally:
        env.close()

    run_metrics.finish()
    return run_metrics


def main():
    parser = argparse.ArgumentParser(description="Solve the 30-step gauntlet")
    parser.add_argument("--mode", default="api", choices=["api", "local"])
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "google"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None, help="Metrics output path")
    args = parser.parse_args()

    challenge_config, model_config = load_configs()
    output_path = args.output or str(PROJECT_ROOT / "results" / "metrics.json")

    if args.mode == "api":
        api_cfg = model_config["api_models"]
        if args.model:
            model = args.model
        elif args.provider == "anthropic":
            model = api_cfg["primary"]
        elif args.provider == "google":
            model = api_cfg.get("google", "gemini-3-flash-preview")
        else:
            model = api_cfg["fallback"]
        action_desc = get_action_description()
        policy = LLMPolicy(
            provider=args.provider,
            model=model,
            action_description=action_desc,
            max_tokens=api_cfg.get("max_tokens", 2048),
            temperature=api_cfg.get("temperature", 0.7),
        )

        run_metrics = solve_gauntlet_api(policy, challenge_config)

    elif args.mode == "local":
        raise NotImplementedError(
            "Local model inference not yet implemented. "
            "Use --mode api for now, or complete training first."
        )

    run_metrics.save(output_path)
    run_metrics.print_summary()


if __name__ == "__main__":
    main()
