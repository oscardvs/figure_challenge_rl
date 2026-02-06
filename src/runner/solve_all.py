#!/usr/bin/env python3
"""Main entry point: solve all 30 challenges.

Supports both local model inference (fine-tuned Qwen2.5-3B) and
API-based fallback (Claude/GPT-4o).

Usage:
    python -m src.runner.solve_all                    # Local model
    python -m src.runner.solve_all --mode api         # API model
    python -m src.runner.solve_all --parallel 4       # 4 parallel workers
    python -m src.runner.solve_all --challenges 1 5   # Specific challenges
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.policy import LLMPolicy
from src.agent.prompts import format_task_prompt
from src.environment.action_space import get_action_description
from src.environment.browser_env import AdcockChallengeEnv
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


def solve_challenge_api(
    challenge_id: int,
    policy: LLMPolicy,
    challenge_config: dict,
) -> ChallengeMetrics:
    """Solve a single challenge using the API model."""
    metrics = ChallengeMetrics(challenge_id=challenge_id)
    base_url = challenge_config["base_url"]
    defaults = challenge_config["defaults"]
    start = time.time()

    env = AdcockChallengeEnv(
        challenge_id=challenge_id,
        max_steps=defaults["max_steps"],
        headless=defaults["headless"],
    )
    task_prompt = format_task_prompt(challenge_id, base_url)

    try:
        obs_text, _ = env.reset()
        action_history: list[str] = []

        for step in range(defaults["max_steps"]):
            action, reasoning = policy.select_action(
                obs_text=obs_text,
                task_prompt=task_prompt,
                action_history=action_history,
                step=step,
            )

            obs_text, reward, terminated, truncated, info = env.step(action)
            action_history.append(action)
            metrics.steps = step + 1

            if terminated or truncated:
                metrics.success = reward > 0
                break

    except Exception as e:
        logger.error(f"Challenge {challenge_id} error: {e}")
        metrics.error = str(e)
    finally:
        env.close()

    metrics.elapsed_seconds = time.time() - start
    tokens = policy.total_tokens
    metrics.input_tokens = tokens["input"]
    metrics.output_tokens = tokens["output"]

    return metrics


def solve_challenge_local(
    challenge_id: int,
    challenge_config: dict,
    model_path: str,
) -> ChallengeMetrics:
    """Solve a single challenge using the fine-tuned local model.

    TODO: Implement after training is complete. Will use vLLM for inference.
    """
    raise NotImplementedError(
        "Local model inference not yet implemented. "
        "Use --mode api for now, or complete training first."
    )


def main():
    parser = argparse.ArgumentParser(description="Solve all 30 challenges")
    parser.add_argument("--mode", default="api", choices=["api", "local"])
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--challenges", nargs="+", type=int)
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None, help="Metrics output path")
    args = parser.parse_args()

    challenge_config, model_config = load_configs()
    challenges = args.challenges or list(range(1, challenge_config["num_challenges"] + 1))
    output_path = args.output or str(PROJECT_ROOT / "results" / "metrics.json")

    run_metrics = RunMetrics()
    run_metrics.start()

    if args.mode == "api":
        api_cfg = model_config["api_models"]
        model = args.model or (
            api_cfg["primary"] if args.provider == "anthropic" else api_cfg["fallback"]
        )
        action_desc = get_action_description()
        policy = LLMPolicy(
            provider=args.provider,
            model=model,
            action_description=action_desc,
        )

        if args.parallel > 1:
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = {
                    executor.submit(
                        solve_challenge_api, cid, policy, challenge_config
                    ): cid
                    for cid in challenges
                }
                for future in as_completed(futures):
                    cid = futures[future]
                    try:
                        m = future.result()
                        run_metrics.add_challenge(m)
                        status = "SOLVED" if m.success else "FAILED"
                        logger.info(
                            f"Challenge {cid}: {status} "
                            f"({m.steps} steps, {m.elapsed_seconds:.1f}s)"
                        )
                    except Exception as e:
                        logger.error(f"Challenge {cid} failed: {e}")
        else:
            for cid in challenges:
                logger.info(f"{'='*40} Challenge {cid} {'='*40}")
                m = solve_challenge_api(cid, policy, challenge_config)
                run_metrics.add_challenge(m)
                status = "SOLVED" if m.success else "FAILED"
                logger.info(
                    f"Challenge {cid}: {status} "
                    f"({m.steps} steps, {m.elapsed_seconds:.1f}s)"
                )

    elif args.mode == "local":
        model_path = str(PROJECT_ROOT / "models" / "qwen25-3b-browser-rl")
        for cid in challenges:
            m = solve_challenge_local(cid, challenge_config, model_path)
            run_metrics.add_challenge(m)

    run_metrics.finish()
    run_metrics.save(output_path)
    run_metrics.print_summary()


if __name__ == "__main__":
    main()
