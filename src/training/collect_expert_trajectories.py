#!/usr/bin/env python3
"""Collect expert trajectories by running the deterministic solver with recording.

The solver acts as a teacher: it solves challenges while a RecordingPage proxy
captures every Playwright call and maps it to BrowserGym action format. The
resulting trajectories can be used directly for SFT training.

Usage:
    python -m src.training.collect_expert_trajectories
    python -m src.training.collect_expert_trajectories --steps 1 2 3 --runs-per-step 3
    python -m src.training.collect_expert_trajectories --output-dir data/expert_trajectories
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "challenge_config.yaml"


def _make_obs_extractor(env, page, pruner):
    """Create an observation extractor function for the RecordingPage.

    Returns a callable () -> str that extracts the current AXTree + HTML
    observation from the environment.
    """
    from src.environment.observation import extract_html_snippet, extract_obs_text

    def extract() -> str:
        try:
            obs = env._get_obs()
        except Exception:
            # Fallback: return minimal observation.
            return f"URL: {page.url}\n[observation extraction failed]"
        html_snippet = extract_html_snippet(page, max_chars=6000)
        return extract_obs_text(obs, pruner, html_snippet=html_snippet)

    return extract


def collect_step(
    config: dict,
    step_number: int,
    run_index: int,
) -> dict | None:
    """Run the solver on one step with recording, return trajectory dict.

    Creates a fresh GauntletEnv, navigates to the target step by solving
    prior steps with the solver, then records the target step solve.
    """
    from src.agent.trajectory_recorder import RecordingPage
    from src.environment.browser_env import GauntletEnv
    from src.environment.observation import AXTreePruner
    from src.solver.challenge_detector import ChallengeDetector
    from src.solver.deterministic_solver import DeterministicSolver

    env = GauntletEnv(
        headless=config.get("headless", True),
        max_actions_per_step=30,
    )
    solver = DeterministicSolver(max_attempts=15, step_timeout=25.0)
    detector = ChallengeDetector()
    pruner = AXTreePruner(visible_only=False, with_bid_only=True, target_tokens=2000)

    try:
        obs_text, info = env.reset()
        page = env._env.page

        # Solve prior steps without recording to reach target step.
        for s in range(1, step_number):
            logger.info(f"  Run {run_index}: solving step {s} (pre-target)")
            result = solver.solve_step(page, s)
            if not result.success:
                logger.warning(f"  Run {run_index}: failed to solve pre-step {s}")
                return None
            # Wait for step transition.
            time.sleep(1.0)

        # Now record the target step.
        logger.info(f"  Run {run_index}: recording step {step_number}")

        # Detect challenge type.
        detections = detector.detect(page)
        challenge_type = detections[0].challenge_type.name if detections else "UNKNOWN"

        # Create recording proxy.
        obs_fn = _make_obs_extractor(env._env, page, pruner)
        recording_page = RecordingPage(
            real_page=page,
            obs_extractor_fn=obs_fn,
            step_number=step_number,
            challenge_type=challenge_type,
        )

        # Solve with recording.
        result = solver.solve_step(recording_page, step_number)

        # Build trajectory.
        trajectory = recording_page.get_trajectory(
            success=result.success,
            code_found=result.code_found,
        )

        return trajectory.to_sft_format()

    except Exception as e:
        logger.error(f"  Run {run_index} error: {e}", exc_info=True)
        return None
    finally:
        env.close()


def generate_preference_pairs(
    trajectories: list[dict],
    step_number: int,
) -> list[dict]:
    """Generate preference pairs from success/failure trajectory contrasts.

    For each observation state that appears in both a successful and failed
    trajectory, create a preference pair (chosen = successful action,
    rejected = failed action).
    """
    successful = [t for t in trajectories if t.get("success")]
    failed = [t for t in trajectories if not t.get("success")]

    if not successful or not failed:
        return []

    pairs = []
    # Use the first successful trajectory's actions as "chosen".
    for good_traj in successful:
        for good_step in good_traj["steps"]:
            obs_text = good_step["obs_text"]
            chosen_action = good_step["action"]

            # Find a failed trajectory with a different action for similar state.
            for bad_traj in failed:
                for bad_step in bad_traj["steps"]:
                    if bad_step["action"] != chosen_action:
                        pairs.append({
                            "step_number": step_number,
                            "step_index": good_step["step_index"],
                            "obs_text": obs_text,
                            "chosen_action": chosen_action,
                            "rejected_action": bad_step["action"],
                        })
                        break  # One pair per good step.
                break  # One bad trajectory per good step.

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Collect expert trajectories from deterministic solver"
    )
    parser.add_argument(
        "--steps", nargs="+", type=int,
        help="Specific step numbers to collect (default: all 1-30)",
    )
    parser.add_argument(
        "--runs-per-step", type=int, default=3,
        help="Number of solver runs per step for trajectory diversity",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "expert_trajectories"),
    )
    parser.add_argument(
        "--pairs-dir",
        default=str(PROJECT_ROOT / "data" / "expert_preference_pairs"),
    )
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    steps = args.steps or list(range(1, config.get("num_steps", 30) + 1))
    output_dir = Path(args.output_dir)
    pairs_dir = Path(args.pairs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    total_trajectories = 0
    total_successful = 0
    total_pairs = 0
    start_time = time.time()

    for step_num in steps:
        logger.info(f"{'=' * 40} Step {step_num} {'=' * 40}")

        step_trajectories = []
        for run_idx in range(args.runs_per_step):
            traj = collect_step(config["defaults"], step_num, run_idx)
            if traj is not None:
                step_trajectories.append(traj)
                total_trajectories += 1
                if traj["success"]:
                    total_successful += 1

        # Merge with existing trajectories (append, don't overwrite).
        out_path = output_dir / f"step_{step_num:02d}.json"
        existing: list = []
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text())
            except (json.JSONDecodeError, OSError):
                existing = []
        merged = existing + step_trajectories
        with open(out_path, "w") as f:
            json.dump(merged, f, indent=2)
        new_succ = sum(1 for t in step_trajectories if t["success"])
        logger.info(
            f"Step {step_num}: {new_succ}/{len(step_trajectories)} successful "
            f"(total: {len(merged)} trajectories), saved to {out_path}"
        )

        # Generate and save preference pairs (merge with existing).
        all_for_pairs = merged  # Generate pairs from all trajectories for this step.
        pairs = generate_preference_pairs(all_for_pairs, step_num)
        if pairs:
            pairs_path = pairs_dir / f"step_{step_num:02d}.json"
            with open(pairs_path, "w") as f:
                json.dump(pairs, f, indent=2)
            total_pairs += len(pairs)
            logger.info(f"  Generated {len(pairs)} preference pairs")

    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Collection complete in {elapsed:.0f}s")
    logger.info(f"Trajectories: {total_trajectories} ({total_successful} successful)")
    logger.info(f"Preference pairs: {total_pairs}")
    logger.info(f"Data saved to {output_dir}")


if __name__ == "__main__":
    main()
