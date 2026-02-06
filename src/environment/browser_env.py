"""Custom BrowserGym environment for the 30 Netlify challenges."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import yaml
from browsergym.core.env import BrowserEnv
from browsergym.core.task import AbstractBrowserTask

from src.environment.action_space import get_action_set
from src.environment.observation import AXTreePruner, extract_obs_text

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "challenge_config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class AdcockChallengeTask(AbstractBrowserTask):
    """BrowserGym task for a single Netlify challenge.

    Each challenge is identified by its integer ID (1-30). The task:
    1. Navigates to the challenge URL.
    2. Validates completion via page content heuristics.
    """

    def __init__(self, seed: int, challenge_id: int, base_url: str, max_steps: int = 25):
        super().__init__(seed)
        self.challenge_id = challenge_id
        self.base_url = base_url.rstrip("/")
        self.max_steps = max_steps
        self.start_time: float = 0
        self.step_count: int = 0

        # Override defaults from AbstractBrowserTask.
        self.viewport = {"width": 1280, "height": 720}
        self.slow_mo = 0
        self.timeout = 10000

    @classmethod
    def get_task_id(cls) -> str:
        return "adcock_challenge"

    def setup(self, page) -> tuple[list[dict], dict]:
        """Navigate to the challenge page and return the goal."""
        url = f"{self.base_url}/challenge/{self.challenge_id}"
        page.goto(url, wait_until="networkidle", timeout=15000)
        self.start_time = time.time()
        self.step_count = 0

        goal = [
            {
                "type": "text",
                "text": (
                    f"Complete challenge {self.challenge_id}. "
                    "Interact with the page elements to solve whatever task "
                    "is presented. Do not hardcode knowledge — respond to "
                    "what you observe."
                ),
            }
        ]
        return goal, {"challenge_id": self.challenge_id}

    def validate(self, page, chat_messages) -> tuple[float, bool, str, dict]:
        """Check if the challenge has been completed.

        Heuristics for detecting completion:
        1. Page contains explicit success indicators (e.g., "success",
           "complete", "congratulations", "well done", checkmark).
        2. URL changed to indicate completion.
        3. A specific element with success-related attributes appeared.
        """
        self.step_count += 1
        elapsed = time.time() - self.start_time

        try:
            # Check page content for success signals.
            content = page.content().lower()
            success_signals = [
                "success", "congratulations", "well done", "challenge complete",
                "you did it", "passed", "✓", "✅", "completed",
            ]
            found_success = any(s in content for s in success_signals)

            # Also check if there is a visible success element.
            success_el = page.query_selector(
                "[class*='success'], [class*='complete'], [data-success], "
                "[class*='passed'], [class*='congrat']"
            )
            if success_el and success_el.is_visible():
                found_success = True

        except Exception as e:
            logger.warning(f"Validation error on challenge {self.challenge_id}: {e}")
            found_success = False

        done = found_success or self.step_count >= self.max_steps
        reward = 1.0 if found_success else 0.0

        info = {
            "challenge_id": self.challenge_id,
            "step_count": self.step_count,
            "elapsed_seconds": elapsed,
            "success": found_success,
        }

        return reward, done, "", info

    def teardown(self) -> None:
        pass


class AdcockChallengeEnv:
    """Gymnasium-style wrapper around BrowserGym for the 30 challenges.

    Usage:
        env = AdcockChallengeEnv(challenge_id=1)
        obs_text, info = env.reset()
        while True:
            obs_text, reward, terminated, truncated, info = env.step("click('button1')")
            if terminated or truncated:
                break
        env.close()
    """

    def __init__(
        self,
        challenge_id: int,
        max_steps: int = 25,
        headless: bool = True,
        record_video_dir: Optional[str] = None,
        target_tokens: int = 3000,
    ):
        config = _load_config()
        self.challenge_id = challenge_id
        self.base_url = config["base_url"]
        self.max_steps = max_steps

        self.pruner = AXTreePruner(
            visible_only=True,
            with_bid_only=True,
            target_tokens=target_tokens,
        )

        action_set = get_action_set()

        self._env = BrowserEnv(
            task_entrypoint=AdcockChallengeTask,
            task_kwargs={
                "challenge_id": challenge_id,
                "base_url": self.base_url,
                "max_steps": max_steps,
            },
            headless=headless,
            action_mapping=action_set.to_python_code,
            record_video_dir=record_video_dir,
        )

        self.action_description = action_set.describe(
            with_long_description=True, with_examples=True,
        )

    def reset(self, seed: int | None = None) -> tuple[str, dict]:
        """Reset the environment and return the initial observation text."""
        obs, info = self._env.reset(seed=seed)
        obs_text = extract_obs_text(obs, self.pruner)
        return obs_text, info

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        """Execute an action and return (obs_text, reward, terminated, truncated, info)."""
        obs, reward, terminated, truncated, info = self._env.step(action)
        obs_text = extract_obs_text(obs, self.pruner)
        return obs_text, reward, terminated, truncated, info

    def close(self):
        """Close the environment and release resources."""
        self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def make_challenge_env(
    challenge_id: int,
    headless: bool = True,
    **kwargs,
) -> AdcockChallengeEnv:
    """Factory function for creating challenge environments."""
    return AdcockChallengeEnv(
        challenge_id=challenge_id,
        headless=headless,
        **kwargs,
    )
