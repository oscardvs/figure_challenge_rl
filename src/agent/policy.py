"""LLM policy wrappers for action proposal and critique.

Supports Claude (Anthropic), GPT-4o (OpenAI), Gemini (Google)
as API backends, and local LoRA models via Unsloth/PEFT.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.agent.prompts import (
    format_observation_message,
    format_pure_agent_system_prompt,
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

    Supported providers: "anthropic", "openai", "google".
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
            import anthropic
            self.client = anthropic.Anthropic()
        elif provider == "openai":
            import openai
            self.client = openai.OpenAI()
        elif provider == "google":
            from google import genai
            self.client = genai.Client()
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
            logger.warning("Failed to parse critic score from: %.100s", response_text)
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
        elif self.provider == "google":
            return self._call_google(user_message)
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

    def _call_google(self, user_message: str) -> str:
        """Call the Gemini API via the google-genai SDK."""
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        # Track token usage from Gemini's usage metadata.
        if response.usage_metadata:
            self._total_input_tokens += response.usage_metadata.prompt_token_count or 0
            self._total_output_tokens += response.usage_metadata.candidates_token_count or 0

        return response.text

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


class LocalPolicy:
    """Policy backed by a locally-loaded LoRA adapter (Qwen3-4B + QLoRA).

    Uses the same ``select_action`` interface as ``LLMPolicy`` so it can be
    dropped into ``solve_gauntlet_api`` without changes.

    Loading order:
      1. Unsloth ``FastLanguageModel`` (preferred — applies inference optimizations).
      2. PEFT ``AutoPeftModelForCausalLM`` with 4-bit BitsAndBytes (fallback).
    """

    def __init__(
        self,
        adapter_dir: str = "models/sft",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        max_seq_length: int = 6144,
        action_description: str = "",
    ):
        self.adapter_dir = adapter_dir
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_seq_length = max_seq_length

        self.system_prompt = format_pure_agent_system_prompt(action_description)

        # Conversation history for the current step (reset on step transitions).
        self._conversation: list[dict[str, str]] = []
        self._current_step: int | None = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the adapter + base model for inference."""
        # Patch HF Hub timeouts before any from_pretrained call.
        import huggingface_hub.constants
        huggingface_hub.constants.HF_HUB_ETAG_TIMEOUT = 120
        huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = 120

        try:
            from unsloth import FastLanguageModel

            logger.info("Loading model via Unsloth from %s", self.adapter_dir)
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.adapter_dir,
                max_seq_length=self.max_seq_length,
                load_in_4bit=True,
                dtype=None,
            )
            FastLanguageModel.for_inference(self.model)
            logger.info("Model loaded via Unsloth (inference mode)")

        except ImportError:
            logger.info(
                "Unsloth not available, falling back to PEFT + BitsAndBytes"
            )
            import torch
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoTokenizer, BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                self.adapter_dir,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_dir)
            self.model.eval()
            logger.info("Model loaded via PEFT + BitsAndBytes")

        # Build EOS token list: <|im_end|> + <|endoftext|> for robust stopping.
        self._eos_token_ids = [self.tokenizer.eos_token_id]
        endoftext_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if isinstance(endoftext_id, int) and endoftext_id != self.tokenizer.unk_token_id:
            self._eos_token_ids.append(endoftext_id)
        self._pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # Cache tokenized system prompt length for context budget calculations.
        sys_msg = [{"role": "system", "content": self.system_prompt}]
        sys_text = self.tokenizer.apply_chat_template(
            sys_msg, tokenize=False, add_generation_prompt=False,
        )
        self._system_prompt_tokens = len(self.tokenizer.encode(sys_text))
        logger.info(
            "EOS tokens: %s, system prompt: %d tokens",
            self._eos_token_ids, self._system_prompt_tokens,
        )

    def select_action(
        self,
        obs_text: str,
        task_prompt: str,
        action_history: list[str],
        step: int,
    ) -> tuple[str, str]:
        """Select a single action given the current observation.

        Maintains multi-turn conversation history within a step, matching the
        training format: system → user(task+obs) → assistant → user(obs) → ...

        Returns:
            (action, reasoning) tuple.
        """
        import torch

        # Detect step transition from task_prompt (contains "step N").
        current_step = self._parse_step_from_prompt(task_prompt)
        if current_step != self._current_step:
            self._conversation = []
            self._current_step = current_step
            logger.info("LocalPolicy: new step %s, reset conversation", current_step)

        # Build the user message matching training format.
        action_idx = len(self._conversation) // 2  # each turn = user + assistant
        if not self._conversation:
            user_content = (
                f"Complete step {current_step}.\n\n"
                f"[Action {action_idx}] Current page state:\n{obs_text}"
            )
        else:
            user_content = (
                f"[Action {action_idx}] Current page state:\n{obs_text}"
            )

        self._conversation.append({"role": "user", "content": user_content})

        # Trim conversation so system + conversation + generation fits in context.
        self._trim_conversation_to_fit()

        # Build full messages for the model.
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self._conversation,
        ]

        # Tokenize and generate.
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            input_text, return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]
        logger.info(
            "LocalPolicy: %d conversation msgs, %d input tokens",
            len(self._conversation), input_len,
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=self._pad_token_id,
                eos_token_id=self._eos_token_ids,
            )

        # Decode only the newly generated tokens.
        new_tokens = output_ids[0, input_len:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Parse action from response.
        action = parse_action_from_response(response_text)

        # Store only the action in conversation history (not full think+response).
        # This keeps context compact — the model saw multi-turn (obs→action) during
        # training, but the think blocks in history add noise and waste tokens.
        self._conversation.append({"role": "assistant", "content": action})

        logger.debug(
            "LocalPolicy generated (%d tokens): %s",
            len(new_tokens), action,
        )
        return action, response_text

    def _trim_conversation_to_fit(self) -> None:
        """Remove oldest conversation turns to fit within context budget.

        Keeps the first user turn (contains task instruction) and the most
        recent turns. Removes pairs (user+assistant) from the middle.
        """
        # Budget: max_seq_length minus generation headroom minus system prompt.
        token_budget = self.max_seq_length - self.max_new_tokens - self._system_prompt_tokens - 50

        while len(self._conversation) > 1:
            # Estimate token count of conversation.
            conv_text = "".join(m["content"] for m in self._conversation)
            conv_tokens = len(self.tokenizer.encode(conv_text))
            if conv_tokens <= token_budget:
                break

            # Remove the oldest completed turn pair (index 1 & 2, after first user msg).
            # If only first user msg + current user msg remain, can't trim further.
            if len(self._conversation) <= 2:
                break
            # Remove one user+assistant pair from position 1 (right after first user turn).
            removed = self._conversation[1:3]
            self._conversation = self._conversation[:1] + self._conversation[3:]
            logger.debug(
                "Trimmed conversation: removed turn pair, %d msgs remaining",
                len(self._conversation),
            )

    @staticmethod
    def _parse_step_from_prompt(task_prompt: str) -> int:
        """Extract the step number from the task prompt."""
        match = re.search(r"step\s+(\d+)", task_prompt, re.IGNORECASE)
        return int(match.group(1)) if match else 0
