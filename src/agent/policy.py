"""LLM policy wrappers for action proposal and critique.

Supports Claude (Anthropic) and GPT-4o (OpenAI) as backend models
for MCTS data collection.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import anthropic
import openai

from src.agent.prompts import (
    format_observation_message,
    format_system_prompt,
    format_task_prompt,
    parse_action_from_response,
)

logger = logging.getLogger(__name__)


@dataclass
class ActionCandidate:
    """A candidate action proposed by the LLM policy."""
    action: str
    reasoning: str
    critic_score: float  # 0.0–1.0 estimated utility from the critic


class LLMPolicy:
    """Wraps an API LLM for action proposal and critique.

    Used during MCTS data collection (Phase 2) where the API model
    serves as both the policy (proposing actions) and the critic
    (ranking actions by estimated utility).
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 512,
        action_description: str = "",
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.action_description = action_description

        if provider == "anthropic":
            self.client = anthropic.Anthropic()
        elif provider == "openai":
            self.client = openai.OpenAI()
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self.system_prompt = format_system_prompt(action_description)
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def propose_actions(
        self,
        obs_text: str,
        task_prompt: str,
        action_history: list[str],
        step: int,
        k: int = 5,
    ) -> list[ActionCandidate]:
        """Propose K candidate actions for the current state.

        Args:
            obs_text: Current observation (URL + AXTree).
            task_prompt: Task description.
            action_history: List of previous actions taken.
            step: Current step number.
            k: Number of candidate actions to propose.

        Returns:
            List of ActionCandidate objects with actions and scores.
        """
        history_str = ""
        if action_history:
            history_str = "Previous actions:\n" + "\n".join(
                f"  Step {i}: {a}" for i, a in enumerate(action_history)
            )

        user_msg = (
            f"{task_prompt}\n\n"
            f"{history_str}\n\n"
            f"{format_observation_message(obs_text, step)}\n\n"
            f"Propose {k} different candidate actions for this state. "
            f"For each, give:\n"
            f"1. A reasoning chain\n"
            f"2. The action\n"
            f"3. A confidence score (0.0-1.0)\n\n"
            f"Format each candidate as:\n"
            f"CANDIDATE N:\n"
            f"Reasoning: ...\n"
            f"Action: action_here(args)\n"
            f"Confidence: 0.X"
        )

        response_text = self._call_llm(user_msg)
        return self._parse_candidates(response_text, k)

    def critique_action(
        self,
        obs_text: str,
        task_prompt: str,
        action: str,
        action_history: list[str],
        step: int,
    ) -> float:
        """Score a single action by estimated utility (0.0-1.0)."""
        history_str = ""
        if action_history:
            history_str = "Previous actions:\n" + "\n".join(
                f"  Step {i}: {a}" for i, a in enumerate(action_history)
            )

        user_msg = (
            f"{task_prompt}\n\n"
            f"{history_str}\n\n"
            f"{format_observation_message(obs_text, step)}\n\n"
            f"Proposed action: {action}\n\n"
            f"Rate this action's utility for completing the task on a scale "
            f"from 0.0 (useless) to 1.0 (optimal). "
            f"Reply with just the number."
        )

        response_text = self._call_llm(user_msg)
        try:
            score = float(re.search(r"(\d+\.?\d*)", response_text).group(1))
            return max(0.0, min(1.0, score))
        except (AttributeError, ValueError):
            return 0.5

    def select_action(
        self,
        obs_text: str,
        task_prompt: str,
        action_history: list[str],
        step: int,
    ) -> tuple[str, str]:
        """Select a single best action (no MCTS, direct policy query).

        Returns:
            (action, reasoning) tuple.
        """
        history_str = ""
        if action_history:
            history_str = "Previous actions:\n" + "\n".join(
                f"  Step {i}: {a}" for i, a in enumerate(action_history)
            )

        user_msg = (
            f"{task_prompt}\n\n"
            f"{history_str}\n\n"
            f"{format_observation_message(obs_text, step)}"
        )

        response_text = self._call_llm(user_msg)
        action = parse_action_from_response(response_text)
        return action, response_text

    @property
    def total_tokens(self) -> dict:
        return {
            "input": self._total_input_tokens,
            "output": self._total_output_tokens,
        }

    def _call_llm(self, user_message: str) -> str:
        """Make an API call and return the response text."""
        if self.provider == "anthropic":
            return self._call_anthropic(user_message)
        else:
            return self._call_openai(user_message)

    def _call_anthropic(self, user_message: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        self._total_input_tokens += response.usage.input_tokens
        self._total_output_tokens += response.usage.output_tokens
        return response.content[0].text

    def _call_openai(self, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        usage = response.usage
        self._total_input_tokens += usage.prompt_tokens
        self._total_output_tokens += usage.completion_tokens
        return response.choices[0].message.content

    def _parse_candidates(self, response: str, k: int) -> list[ActionCandidate]:
        """Parse K candidates from the LLM response."""
        candidates = []

        # Split by CANDIDATE markers.
        parts = re.split(r"CANDIDATE\s+\d+:", response, flags=re.IGNORECASE)
        for part in parts[1:]:  # Skip text before first CANDIDATE.
            reasoning = ""
            action = ""
            confidence = 0.5

            reasoning_match = re.search(
                r"Reasoning:\s*(.+?)(?=Action:|$)", part, re.DOTALL | re.IGNORECASE
            )
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()

            action_match = re.search(
                r"Action:\s*(.+?)(?=Confidence:|$)", part, re.DOTALL | re.IGNORECASE
            )
            if action_match:
                action = action_match.group(1).strip().split("\n")[0]

            conf_match = re.search(r"Confidence:\s*(\d+\.?\d*)", part, re.IGNORECASE)
            if conf_match:
                confidence = max(0.0, min(1.0, float(conf_match.group(1))))

            if action:
                candidates.append(ActionCandidate(
                    action=action,
                    reasoning=reasoning,
                    critic_score=confidence,
                ))

        # If parsing failed, try to extract any action-like patterns.
        if not candidates:
            action = parse_action_from_response(response)
            if action:
                candidates.append(ActionCandidate(
                    action=action,
                    reasoning=response,
                    critic_score=0.5,
                ))

        return candidates[:k]
