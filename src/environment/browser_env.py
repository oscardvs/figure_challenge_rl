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

# JavaScript to dismiss popup overlays that block interaction.
# The challenge pages spawn multiple fixed-position overlays (z-index 9996-10000)
# including cookie consent, "you won a prize", fake alerts, and fullscreen backdrops.
# These are intentional distractions, not puzzle content.
_DISMISS_OVERLAYS_JS = """\
(() => {
  // Remove fullscreen backdrop overlays (fixed inset-0, high z-index).
  document.querySelectorAll('.fixed.inset-0').forEach(el => el.remove());
  // Remove popup modals (fixed position, z-index >= 9990).
  document.querySelectorAll('[class*="fixed"]').forEach(el => {
    const z = parseInt(getComputedStyle(el).zIndex) || 0;
    const text = (el.textContent || '').toLowerCase();
    // Keep the step progress header — make it click-through instead.
    if (text.includes('of 30') && text.includes('browser navigation')) {
      el.style.pointerEvents = 'none';
      return;
    }
    // Remove popup overlays (cookie consent, prizes, alerts, etc.)
    if (z >= 9990) {
      el.remove();
    }
  });
  // Remove floating clickable distractors (absolute/fixed positioned "Click Me" etc.)
  document.querySelectorAll('[class*="fixed"][class*="cursor-pointer"]').forEach(el => {
    const text = (el.textContent || '').trim();
    if (['Click Me!', 'Link!', 'Here!', 'Try This!', 'Button!', 'Moving!'].includes(text)) {
      el.remove();
    }
  });
})();
"""

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "challenge_config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _get_step_from_url(url: str) -> int | None:
    """Extract the step number from a URL like /step5?version=3."""
    match = re.search(r"/step(\d+)", url)
    return int(match.group(1)) if match else None


def _wait_for_content(page, timeout: float = 5.0, min_length: int = 1000):
    """Poll until React renders substantial content after SPA navigation."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if page.evaluate("document.body.innerHTML.length") >= min_length:
                return
        except Exception:
            pass
        time.sleep(0.3)
    logger.warning(f"_wait_for_content timed out after {timeout}s")


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

    When prior_solutions is provided, setup() auto-replays prior steps
    by entering known codes and clicking Submit to reach the target step.
    """

    def __init__(
        self,
        seed: int,
        step_number: int,
        base_url: str,
        version: int = 3,
        max_actions: int = 30,
        prior_solutions: dict[int, str] | None = None,
    ):
        super().__init__(seed)
        self.step_number = step_number
        self.base_url = base_url.rstrip("/")
        self.version = version
        self.max_actions = max_actions
        self.prior_solutions = prior_solutions or {}
        self.start_time: float = 0
        self.action_count: int = 0

        self.viewport = {"width": 1280, "height": 720}
        self.slow_mo = 0
        self.timeout = 15000

    @classmethod
    def get_task_id(cls) -> str:
        return "adcock_step"

    def _replay_prior_steps(self, page):
        """Auto-replay prior steps using known solution codes."""
        for step_idx in range(1, self.step_number):
            code = self.prior_solutions.get(step_idx)
            if not code:
                logger.warning(
                    f"No solution for step {step_idx}, cannot reach step {self.step_number}"
                )
                return False

            # Wait for content to render after SPA transition.
            _wait_for_content(page)

            try:
                # Find the code input and fill it.
                code_input = page.get_by_placeholder("Enter 6-character code")
                code_input.fill(code, timeout=5000)

                # Click Submit Code.
                submit_btn = page.get_by_role("button", name="Submit Code")
                submit_btn.click(timeout=5000)

                # Wait for navigation to next step.
                next_step = step_idx + 1
                page.wait_for_url(f"**/step{next_step}**", timeout=10000)
                logger.info(f"Replayed step {step_idx} → step {next_step}")
            except Exception as e:
                logger.warning(f"Failed to replay step {step_idx}: {e}")
                return False

        return True

    def setup(self, page) -> tuple[list[dict], dict]:
        """Navigate to the step via the app's client-side routing.

        For step 1: go to home page and click START.
        For step N > 1: auto-replay prior steps if solutions are available.
        """
        # Always start from home and click START (SPA client-side routing).
        page.goto(self.base_url, wait_until="networkidle", timeout=15000)
        try:
            start_btn = page.get_by_role("button", name="START")
            start_btn.click(timeout=5000)
            page.wait_for_url("**/step1**", timeout=10000)
        except Exception as e:
            logger.warning(f"Failed to click START: {e}")

        # Auto-replay prior steps if needed.
        if self.step_number > 1:
            self._replay_prior_steps(page)

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
        self._prev_step: int | None = None

        self.pruner = AXTreePruner(
            visible_only=False,
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

    def _dismiss_overlays(self):
        """Remove popup overlays that block page interaction."""
        try:
            self._env.page.evaluate(_DISMISS_OVERLAYS_JS)
        except Exception:
            pass  # Page might not be ready yet.

    def _wait_and_refresh_obs(self) -> dict:
        """Wait for React content to render, dismiss overlays, re-extract obs."""
        _wait_for_content(self._env.page)
        self._dismiss_overlays()
        return self._env._get_obs()

    def reset(self, seed: int | None = None) -> tuple[str, dict]:
        obs, info = self._env.reset(seed=seed)

        # Wait for React to render step 1 content after START → step1 transition.
        _wait_for_content(self._env.page)
        self._dismiss_overlays()
        obs = self._env._get_obs()

        task_info = info.get("task_info", info)
        self._prev_step = task_info.get("current_step", 1)

        obs_text = extract_obs_text(obs, self.pruner)
        return obs_text, info

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)

        task_info = info.get("task_info", info)
        current_step = task_info.get("current_step", self._prev_step)

        # Detect step transition — React SPA needs time to render new step content.
        if self._prev_step is not None and current_step > self._prev_step:
            obs = self._wait_and_refresh_obs()

        self._prev_step = current_step
        self._dismiss_overlays()
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

    Navigates to step N (auto-replaying prior steps if solutions are provided).
    Success = URL advances to step N+1.

    Usage:
        env = StepEnv(step_number=5, prior_solutions={1: "ABC123", 2: "DEF456", ...})
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
        prior_solutions: dict[int, str] | None = None,
    ):
        config = _load_config()
        self.step_number = step_number
        self.base_url = config["base_url"]
        self.version = config.get("version", 3)
        self.max_actions = max_actions

        self.pruner = AXTreePruner(
            visible_only=False,
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
                "prior_solutions": prior_solutions,
            },
            headless=headless,
            action_mapping=action_set.to_python_code,
            record_video_dir=record_video_dir,
        )

    def _dismiss_overlays(self):
        """Remove popup overlays that block page interaction."""
        try:
            self._env.page.evaluate(_DISMISS_OVERLAYS_JS)
        except Exception:
            pass

    def reset(self, seed: int | None = None) -> tuple[str, dict]:
        obs, info = self._env.reset(seed=seed)

        # Wait for React to render content after SPA navigation.
        _wait_for_content(self._env.page)
        self._dismiss_overlays()
        obs = self._env._get_obs()

        obs_text = extract_obs_text(obs, self.pruner)
        return obs_text, info

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        prev_url = self._env.page.url
        obs, reward, terminated, truncated, info = self._env.step(action)

        # Detect URL-based step transition and wait for React to render.
        current_url = self._env.page.url
        prev_step = _get_step_from_url(prev_url)
        curr_step = _get_step_from_url(current_url)
        if prev_step is not None and curr_step is not None and curr_step > prev_step:
            _wait_for_content(self._env.page)
            self._dismiss_overlays()
            obs = self._env._get_obs()

        self._dismiss_overlays()
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
