"""Parallel challenge execution for submission time.

Runs multiple challenges simultaneously in separate browser tabs
with batched inference via vLLM for the fine-tuned local model.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from src.runner.metrics import ChallengeMetrics, RunMetrics

logger = logging.getLogger(__name__)


def run_parallel(
    challenge_ids: list[int],
    solve_fn: Callable[[int], ChallengeMetrics],
    num_workers: int = 4,
    time_budget_seconds: float = 300.0,
) -> RunMetrics:
    """Execute challenges in parallel with a time budget.

    Args:
        challenge_ids: List of challenge IDs to solve.
        solve_fn: Function that takes a challenge_id and returns metrics.
        num_workers: Number of parallel workers.
        time_budget_seconds: Total time budget (default 5 minutes).

    Returns:
        RunMetrics with results from all challenges.
    """
    run_metrics = RunMetrics()
    run_metrics.start()
    deadline = time.time() + time_budget_seconds

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(solve_fn, cid): cid
            for cid in challenge_ids
        }

        for future in as_completed(futures):
            cid = futures[future]
            remaining = deadline - time.time()

            if remaining <= 0:
                logger.warning(f"Time budget exceeded, skipping remaining challenges")
                for f in futures:
                    if not f.done():
                        f.cancel()
                break

            try:
                m = future.result(timeout=max(1, remaining))
                run_metrics.add_challenge(m)
                status = "SOLVED" if m.success else "FAILED"
                logger.info(
                    f"Challenge {cid}: {status} "
                    f"({m.steps} steps, {m.elapsed_seconds:.1f}s) "
                    f"[{remaining:.0f}s remaining]"
                )
            except Exception as e:
                logger.error(f"Challenge {cid}: ERROR - {e}")
                run_metrics.add_challenge(ChallengeMetrics(
                    challenge_id=cid,
                    error=str(e),
                ))

    run_metrics.finish()
    return run_metrics
