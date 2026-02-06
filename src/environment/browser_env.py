"""Custom BrowserGym environment for the 30-step Netlify gauntlet.

The challenge is a sequential gauntlet:
  Home page → click START → /step1?version=3 → ... → /step30?version=3

Each step presents a unique puzzle. Solving the puzzle reveals a 6-character
code that must be entered in a textbox and submitted to advance to the next step.

We provide two environment modes:
  1. GauntletEnv: Full 30-step sequential run (for submission / full evaluation).
  2. StepEnv: Single-step environment (for MCTS training data collection per step).
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

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


def _get_step_from_url(url: str) -> int | None:
    """Extract the step number from a URL like /step5?version=3."""
    match = re.search(r"/step(\d+)", url)
    return int(match.group(1)) if match else None


class GauntletTask(AbstractBrowserTask):
    """BrowserGym task for the full 30-step gauntlet.

    Starts at the home page, clicks START, then the agent must solve
    all 30 steps sequentially. Reward is given per step completion.
    """

    def __init__(
        self,
        seed: int,
        base_url: str,
        version: int = 3,
        num_steps: int = 30,
        max_actions_per_step: int = 30,
    ):
        super().__init__(seed)
        self.base_url = base_url.rstrip("/")
        self.version = version
        self.num_steps = num_steps
        self.max_actions_per_step = max_actions_per_step
        self.start_time: float = 0
        self.action_count: int = 0
        self.current_step: int = 0
        self.steps_completed: int = 0
        self.actions_this_step: int = 0

        self.viewport = {"width": 1280, "height": 720}
        self.slow_mo = 0
        self.timeout = 15000

    @classmethod
    def get_task_id(cls) -> str:
        return "adcock_gauntlet"

    def setup(self, page) -> tuple[list[dict], dict]:
        """Navigate to home page and click START."""
        page.goto(self.base_url, wait_until="networkidle", timeout=15000)
        self.start_time = time.time()
        self.action_count = 0
        self.current_step = 0
        self.steps_completed = 0
        self.actions_this_step = 0

        # Click the START button to begin.
        try:
            start_btn = page.get_by_role("button", name="START")
            start_btn.click(timeout=5000)
            page.wait_for_url("**/step1**", timeout=10000)
            self.current_step = 1
        except Exception as e:
            logger.warning(f"Failed to click START: {e}")

        goal = [
            {
                "type": "text",
                "text": (
                    "Complete all 30 steps of the Browser Navigation Challenge. "
                    "Each step presents a puzzle. Solve the puzzle to reveal a "
                    "6-character code, enter it in the code input field, and "
                    "click Submit Code to advance. Ignore decoy buttons, popups, "
                    "and distractions — focus on finding and entering the code."
                ),
            }
        ]
        return goal, {"current_step": self.current_step}

    def validate(self, page, chat_messages) -> tuple[float, bool, str, dict]:
        """Check if the agent has advanced to the next step."""
        self.action_count += 1
        self.actions_this_step += 1
        elapsed = time.time() - self.start_time

        url = page.url
        detected_step = _get_step_from_url(url)

        reward = 0.0
        done = False

        if detected_step is not None and detected_step > self.current_step:
            # Agent advanced to a new step.
            steps_jumped = detected_step - self.current_step
            reward = float(steps_jumped)
            self.steps_completed += steps_jumped
            self.current_step = detected_step
            self.actions_this_step = 0
            logger.info(
                f"Advanced to step {self.current_step} "
                f"(action {self.action_count}, {elapsed:.1f}s)"
            )

        # Detect step 30 completion — URL may go to /complete, /success,
        # or page content may indicate gauntlet finished.
        if self.current_step == self.num_steps and detected_step is None:
            try:
                body_text = page.inner_text("body", timeout=2000).lower()
                if any(w in body_text for w in ("congratulat", "complete", "finished", "you win", "all 30")):
                    self.steps_completed = self.num_steps
                    reward += 1.0
                    logger.info("Detected gauntlet completion via page content")
            except Exception:
                pass

        # Check if all steps are complete.
        if self.steps_completed >= self.num_steps:
            done = True
            reward += 10.0  # Bonus for completing all 30.

        # Check for timeout / action budget per step.
        step_stuck = False
        if self.actions_this_step >= self.max_actions_per_step:
            logger.warning(
                f"Exceeded {self.max_actions_per_step} actions on step "
                f"{self.current_step}, skipping this step"
            )
            step_stuck = True
            self.current_step += 1
            self.actions_this_step = 0
            # If skipping past the last step, terminate.
            if self.current_step > self.num_steps:
                done = True

        info = {
            "current_step": self.current_step,
            "steps_completed": self.steps_completed,
            "action_count": self.action_count,
            "actions_this_step": self.actions_this_step,
            "elapsed_seconds": elapsed,
            "url": url,
            "step_skipped": step_stuck,
        }

        return reward, done, "", info

    def teardown(self) -> None:
        pass


class StepTask(AbstractBrowserTask):
    """BrowserGym task for a single step of the gauntlet.

    IMPORTANT: The challenge site is a React SPA with client-side routing.
    Direct navigation to /stepN?version=3 only works for step 1. For all
    other steps, you must navigate through the app sequentially (home →
    START → solve step 1 → step 2 → ... → step N).

    For step 1, this task navigates directly. For step N > 1, it starts
    from the home page and auto-completes prior steps (not yet implemented;
    use GauntletEnv instead).
    """

    def __init__(
        self,
        seed: int,
        step_number: int,
        base_url: str,
        version: int = 3,
        max_actions: int = 30,
    ):
        super().__init__(seed)
        self.step_number = step_number
        self.base_url = base_url.rstrip("/")
        self.version = version
        self.max_actions = max_actions
        self.start_time: float = 0
        self.action_count: int = 0

        self.viewport = {"width": 1280, "height": 720}
        self.slow_mo = 0
        self.timeout = 15000

    @classmethod
    def get_task_id(cls) -> str:
        return "adcock_step"

    def setup(self, page) -> tuple[list[dict], dict]:
        """Navigate to the step via the app's client-side routing.

        For step 1: go to home page and click START.
        For step N > 1: currently only step 1 is directly reachable.
        Use GauntletEnv for multi-step access.
        """
        # Always start from home and click START (SPA client-side routing).
        page.goto(self.base_url, wait_until="networkidle", timeout=15000)
        try:
            start_btn = page.get_by_role("button", name="START")
            start_btn.click(timeout=5000)
            page.wait_for_url("**/step1**", timeout=10000)
        except Exception as e:
            logger.warning(f"Failed to click START: {e}")

        self.start_time = time.time()
        self.action_count = 0

        goal = [
            {
                "type": "text",
                "text": (
                    f"Complete step {self.step_number} of the Browser Navigation "
                    f"Challenge. Find the 6-character code by solving the puzzle, "
                    f"enter it in the code input, and click Submit Code."
                ),
            }
        ]
        return goal, {"step_number": self.step_number}

    def validate(self, page, chat_messages) -> tuple[float, bool, str, dict]:
        """Check if the step has been completed (URL advanced to next step)."""
        self.action_count += 1
        elapsed = time.time() - self.start_time

        url = page.url
        detected_step = _get_step_from_url(url)
        next_step = self.step_number + 1

        success = detected_step is not None and detected_step >= next_step
        done = success or self.action_count >= self.max_actions
        reward = 1.0 if success else 0.0

        info = {
            "step_number": self.step_number,
            "action_count": self.action_count,
            "elapsed_seconds": elapsed,
            "success": success,
            "url": url,
        }

        return reward, done, "", info

    def teardown(self) -> None:
        pass


class GauntletEnv:
    """Full 30-step gauntlet environment.

    Usage:
        env = GauntletEnv()
        obs_text, info = env.reset()
        while True:
            obs_text, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()
    """

    def __init__(
        self,
        headless: bool = True,
        record_video_dir: Optional[str] = None,
        target_tokens: int = 3000,
        max_actions_per_step: int = 30,
    ):
        config = _load_config()
        self.base_url = config["base_url"]
        self.version = config.get("version", 3)
        self.num_steps = config.get("num_steps", 30)
        self.max_actions_per_step = max_actions_per_step

        self.pruner = AXTreePruner(
            visible_only=True,
            with_bid_only=True,
            target_tokens=target_tokens,
        )

        action_set = get_action_set()
        self.action_description = action_set.describe(
            with_long_description=True, with_examples=True,
        )

        self._env = BrowserEnv(
            task_entrypoint=GauntletTask,
            task_kwargs={
                "base_url": self.base_url,
                "version": self.version,
                "num_steps": self.num_steps,
                "max_actions_per_step": max_actions_per_step,
            },
            headless=headless,
            action_mapping=action_set.to_python_code,
            record_video_dir=record_video_dir,
        )

    def reset(self, seed: int | None = None) -> tuple[str, dict]:
        obs, info = self._env.reset(seed=seed)
        obs_text = extract_obs_text(obs, self.pruner)
        return obs_text, info

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        obs_text = extract_obs_text(obs, self.pruner)
        return obs_text, reward, terminated, truncated, info

    def close(self):
        self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class StepEnv:
    """Single-step environment for MCTS training data collection.

    Navigates directly to step N. Success = URL advances to step N+1.
    Note: Direct navigation to /stepN may not work if the site requires
    completing prior steps. In that case, use GauntletEnv to reach step N
    first, then collect data from that point.

    Usage:
        env = StepEnv(step_number=5)
        obs_text, info = env.reset()
        while True:
            obs_text, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()
    """

    def __init__(
        self,
        step_number: int,
        headless: bool = True,
        record_video_dir: Optional[str] = None,
        target_tokens: int = 3000,
        max_actions: int = 30,
    ):
        config = _load_config()
        self.step_number = step_number
        self.base_url = config["base_url"]
        self.version = config.get("version", 3)
        self.max_actions = max_actions

        self.pruner = AXTreePruner(
            visible_only=True,
            with_bid_only=True,
            target_tokens=target_tokens,
        )

        action_set = get_action_set()
        self.action_description = action_set.describe(
            with_long_description=True, with_examples=True,
        )

        self._env = BrowserEnv(
            task_entrypoint=StepTask,
            task_kwargs={
                "step_number": step_number,
                "base_url": self.base_url,
                "version": self.version,
                "max_actions": max_actions,
            },
            headless=headless,
            action_mapping=action_set.to_python_code,
            record_video_dir=record_video_dir,
        )

    def reset(self, seed: int | None = None) -> tuple[str, dict]:
        obs, info = self._env.reset(seed=seed)
        obs_text = extract_obs_text(obs, self.pruner)
        return obs_text, info

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        obs_text = extract_obs_text(obs, self.pruner)
        return obs_text, reward, terminated, truncated, info

    def close(self):
        self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Backward compatibility aliases.
AdcockChallengeEnv = StepEnv
AdcockChallengeTask = StepTask
