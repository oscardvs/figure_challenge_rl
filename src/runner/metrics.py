"""Metrics tracking for challenge execution."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ChallengeMetrics:
    """Metrics for a single challenge attempt."""
    challenge_id: int
    success: bool = False
    steps: int = 0
    elapsed_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None


@dataclass
class RunMetrics:
    """Aggregate metrics for a full 30-challenge run."""
    challenges: list[ChallengeMetrics] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0
    start_time: float = 0.0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total_elapsed_seconds = time.time() - self.start_time

    def add_challenge(self, metrics: ChallengeMetrics):
        self.challenges.append(metrics)

    @property
    def num_solved(self) -> int:
        return sum(1 for c in self.challenges if c.success)

    @property
    def success_rate(self) -> float:
        if not self.challenges:
            return 0.0
        return self.num_solved / len(self.challenges)

    @property
    def total_steps(self) -> int:
        return sum(c.steps for c in self.challenges)

    @property
    def total_tokens(self) -> dict:
        return {
            "input": sum(c.input_tokens for c in self.challenges),
            "output": sum(c.output_tokens for c in self.challenges),
        }

    @property
    def avg_time_per_challenge(self) -> float:
        if not self.challenges:
            return 0.0
        return sum(c.elapsed_seconds for c in self.challenges) / len(self.challenges)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_challenges": len(self.challenges),
                "solved": self.num_solved,
                "success_rate": f"{self.success_rate:.1%}",
                "total_elapsed_seconds": round(self.total_elapsed_seconds, 1),
                "avg_seconds_per_challenge": round(self.avg_time_per_challenge, 1),
                "total_steps": self.total_steps,
                "total_tokens": self.total_tokens,
            },
            "challenges": [
                {
                    "id": c.challenge_id,
                    "success": c.success,
                    "steps": c.steps,
                    "elapsed_seconds": round(c.elapsed_seconds, 2),
                    "tokens": {"input": c.input_tokens, "output": c.output_tokens},
                    "error": c.error,
                }
                for c in self.challenges
            ],
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self):
        d = self.to_dict()["summary"]
        print(f"\n{'='*50}")
        print(f"Run Summary")
        print(f"{'='*50}")
        for k, v in d.items():
            print(f"  {k}: {v}")
        print(f"{'='*50}")
