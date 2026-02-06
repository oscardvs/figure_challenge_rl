"""System prompts and formatting templates for the browser agent."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are an autonomous browser agent that solves web challenges. You interact \
with web pages by reading the accessibility tree and issuing actions.

## Observation Format
You receive the current page state as a pruned accessibility tree. Elements \
with a [bid] tag are interactive — use these bid values in your actions.

## Available Actions
{action_description}

## Rules
1. Issue exactly ONE action per turn.
2. Reason step-by-step in a <think> block, then output the action.
3. If a previous action failed, read the error and try a different approach.
4. Do NOT hardcode knowledge about specific challenges — respond to what you \
observe in the accessibility tree.
5. When the task is complete, the environment will detect it automatically.

## Output Format
<think>
[Your step-by-step reasoning about what you observe and what action to take]
</think>
action_here(arguments)
"""

TASK_PROMPT = """\
Your task: Navigate to the challenge page and complete the challenge. \
The challenge will present interactive elements — buttons, forms, pop-ups, \
navigation traps, hidden elements, timers, etc. Solve whatever the challenge \
presents by interacting with the page elements.

Challenge URL: {challenge_url}
Challenge number: {challenge_id}
"""


def format_system_prompt(action_description: str) -> str:
    """Format the system prompt with the action description."""
    return SYSTEM_PROMPT.format(action_description=action_description)


def format_task_prompt(challenge_id: int, base_url: str) -> str:
    """Format the task prompt for a specific challenge."""
    challenge_url = f"{base_url}/challenge/{challenge_id}"
    return TASK_PROMPT.format(
        challenge_url=challenge_url,
        challenge_id=challenge_id,
    )


def format_observation_message(observation_text: str, step: int) -> str:
    """Format an observation as a user message for the LLM."""
    return f"[Step {step}] Current page state:\n{observation_text}"


def parse_action_from_response(response: str) -> str:
    """Extract the action from an LLM response.

    Expects the action to appear after a </think> block (if present),
    or as the last non-empty line.
    """
    # Try to find content after </think>
    if "</think>" in response:
        after_think = response.split("</think>")[-1].strip()
        if after_think:
            # Return the first non-empty line after </think>
            for line in after_think.split("\n"):
                line = line.strip()
                if line:
                    return line

    # Fallback: return the last non-empty line
    for line in reversed(response.strip().split("\n")):
        line = line.strip()
        if line and not line.startswith("<"):
            return line

    return response.strip()
