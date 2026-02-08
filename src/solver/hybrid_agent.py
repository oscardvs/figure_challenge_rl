"""Deterministic-first, LLM-fallback agent for the 30-step gauntlet.

The ``HybridAgent`` wraps ``DeterministicSolver`` and an optional
``LLMPolicy``.  For each step it tries the deterministic solver first;
if that fails it falls back to the LLM policy via ``GauntletEnv.step()``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from src.solver.challenge_detector import ChallengeType
from src.solver.challenge_handlers import check_progress
from src.solver.deterministic_solver import DeterministicSolver

logger = logging.getLogger(__name__)


@dataclass
class StepTrajectory:
    step_number: int = 0
    challenge_type: str = "UNKNOWN"
    actions: list[dict] = field(default_factory=list)
    code_found: str | None = None
    success: bool = False
    solve_method: str = "deterministic"  # "deterministic" | "llm" | "hybrid"
    elapsed_seconds: float = 0.0


class HybridAgent:
    """Deterministic-first, LLM-fallback agent for the full gauntlet."""

    def __init__(
        self,
        policy=None,
        deterministic_first: bool = True,
        max_llm_actions: int = 20,
    ):
        self.solver = DeterministicSolver()
        self.policy = policy
        self.deterministic_first = deterministic_first
        self.max_llm_actions = max_llm_actions

    def solve_gauntlet(self, env) -> list[StepTrajectory]:
        """Full 30-step sequential run with trajectory logging.

        *env* must be a ``GauntletEnv`` instance.  The agent accesses
        the underlying Playwright page via ``env._env.page`` for the
        deterministic solver, and uses ``env.step()`` for LLM fallback.
        """
        trajectories: list[StepTrajectory] = []
        # Track codes used in prior steps so the solver can skip stale
        # codes that linger in React fiber state after SPA transitions.
        used_codes: list[str] = []

        obs_text, info = env.reset()
        task_info = info.get("task_info", info)
        current_step = task_info.get("current_step", 1)

        for step in range(current_step, 31):
            t0 = time.time()
            logger.info("=" * 60)
            logger.info("  STEP %d/30", step)
            logger.info("=" * 60)

            traj = StepTrajectory(step_number=step)

            if self.deterministic_first:
                traj = self._solve_step_deterministic(env, step, used_codes)

            if not traj.success and self.policy is not None:
                logger.info("Step %d: deterministic failed, falling back to LLM", step)
                # Refresh observation after deterministic attempt may have mutated page
                try:
                    obs_text, _, _, _, _ = env.step("noop()")
                except Exception:
                    pass
                llm_traj = self._solve_step_llm(env, obs_text, step)
                if llm_traj.success:
                    traj = llm_traj
                    traj.solve_method = "llm" if not self.deterministic_first else "hybrid"

            traj.elapsed_seconds = time.time() - t0
            trajectories.append(traj)

            status = "PASSED" if traj.success else "FAILED"
            logger.info("Step %d %s (%.1fs, method=%s, type=%s)",
                        step, status, traj.elapsed_seconds, traj.solve_method, traj.challenge_type)

            if not traj.success:
                logger.warning("Step %d failed — stopping gauntlet", step)
                break

            # After success, get fresh observation for next step
            # The deterministic solver may have advanced the page already
            # Re-sync with GauntletEnv's internal state
            try:
                task_info = env._task_info()
                new_step = task_info.get("current_step", step)
                if new_step > step:
                    # Environment noticed the transition
                    obs_text = env._get_obs_text()
                else:
                    # Force env to notice by doing a noop
                    obs_text, _, terminated, _, info = env.step("noop()")
                    if terminated:
                        break
            except Exception:
                # If env sync fails, just continue
                pass

        return trajectories

    def _solve_step_deterministic(
        self, env, step: int, used_codes: list[str] | None = None,
    ) -> StepTrajectory:
        """Try deterministic solver using direct page access."""
        traj = StepTrajectory(step_number=step, solve_method="deterministic")

        try:
            page = env._env.page
        except AttributeError:
            logger.warning("Cannot access page for deterministic solving")
            return traj

        solve_result = self.solver.solve_step(page, step, stale_codes=used_codes)
        traj.success = solve_result.success
        traj.code_found = solve_result.code_found
        traj.challenge_type = solve_result.challenge_type.name
        traj.actions = [{"action": a, "source": "deterministic"} for a in solve_result.actions_log]
        traj.elapsed_seconds = solve_result.elapsed_seconds
        # Accumulate codes tried for stale-code tracking
        if used_codes is not None and solve_result.codes_tried:
            used_codes.extend(solve_result.codes_tried)
        return traj

    def _solve_step_llm(self, env, obs_text: str, step: int) -> StepTrajectory:
        """LLM fallback via env.step() for BrowserGym observations."""
        from src.agent.prompts import format_gauntlet_task_prompt

        traj = StepTrajectory(step_number=step, solve_method="llm")
        task_prompt = format_gauntlet_task_prompt(step)
        action_history: list[str] = []

        for action_idx in range(self.max_llm_actions):
            action, reasoning = self.policy.select_action(
                obs_text=obs_text,
                task_prompt=task_prompt,
                action_history=action_history[-10:],
                step=action_idx,
            )

            logger.info("[LLM Step %d | Action %d] %s", step, action_idx + 1, action)

            obs_text, reward, terminated, truncated, info = env.step(action)
            action_history.append(action)
            traj.actions.append({
                "action": action,
                "reasoning": reasoning,
                "reward": reward,
                "source": "llm",
            })

            task_info = info.get("task_info", info)
            new_step = task_info.get("current_step", step)

            if new_step > step:
                traj.success = True
                break

            if terminated or truncated:
                break

        return traj
