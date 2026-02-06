#!/usr/bin/env python3
"""Manual exploration script for the 30 Netlify challenges.

Usage:
    python scripts/explore_challenges.py                # Explore all challenges
    python scripts/explore_challenges.py --challenge 5  # Explore challenge 5
    python scripts/explore_challenges.py --interactive  # Interactive step-through mode

In interactive mode, type actions at the prompt (e.g., click('btn1')),
or press Enter to see the current AXTree again, or type 'next' to skip
to the next challenge.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment.browser_env import AdcockChallengeEnv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "data" / "challenge_taxonomy.json"


def explore_challenge(challenge_id: int, interactive: bool = False, headless: bool = True):
    """Explore a single challenge, logging the AXTree at each step."""
    logger.info(f"=== Challenge {challenge_id} ===")

    env = AdcockChallengeEnv(
        challenge_id=challenge_id,
        headless=headless,
        target_tokens=5000,  # More tokens for exploration.
    )

    try:
        obs_text, info = env.reset()
        print(f"\n{'='*60}")
        print(f"Challenge {challenge_id} — Initial State")
        print(f"{'='*60}")
        print(obs_text)
        print(f"{'='*60}\n")

        step = 0
        while True:
            if interactive:
                action = input(f"[Step {step}] Enter action (or 'next'/'quit'): ").strip()
                if action.lower() == "quit":
                    return None
                if action.lower() == "next":
                    break
                if not action:
                    print(obs_text)
                    continue
            else:
                # In non-interactive mode, just capture the initial state and return.
                return {
                    "challenge_id": challenge_id,
                    "initial_obs": obs_text,
                    "url": info.get("url", ""),
                }

            try:
                obs_text, reward, terminated, truncated, info = env.step(action)
                step += 1
                print(f"\n--- Step {step} | Reward: {reward} | Done: {terminated} ---")
                print(obs_text)
                if terminated or truncated:
                    print(f"\nChallenge {'SOLVED' if reward > 0 else 'FAILED'}")
                    break
            except Exception as e:
                print(f"Error executing action: {e}")

    finally:
        env.close()

    return None


def explore_all(headless: bool = True):
    """Explore all 30 challenges and save initial AXTrees."""
    taxonomy = []

    for cid in range(1, 31):
        try:
            result = explore_challenge(cid, interactive=False, headless=headless)
            if result:
                taxonomy.append(result)
                logger.info(f"Challenge {cid}: captured initial state")
        except Exception as e:
            logger.error(f"Challenge {cid}: failed — {e}")
            taxonomy.append({
                "challenge_id": cid,
                "error": str(e),
            })

    # Save taxonomy.
    TAXONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TAXONOMY_PATH, "w") as f:
        json.dump(taxonomy, f, indent=2)
    logger.info(f"Saved taxonomy to {TAXONOMY_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Explore Netlify challenges")
    parser.add_argument("--challenge", type=int, help="Specific challenge ID (1-30)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--no-headless", action="store_true", help="Show browser window")
    args = parser.parse_args()

    headless = not args.no_headless

    if args.challenge:
        explore_challenge(args.challenge, interactive=args.interactive, headless=headless)
    elif args.interactive:
        for cid in range(1, 31):
            result = explore_challenge(cid, interactive=True, headless=headless)
            if result is None and args.interactive:
                # User typed 'quit'.
                break
    else:
        explore_all(headless=headless)


if __name__ == "__main__":
    main()
