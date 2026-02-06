"""System prompts and formatting templates for the browser agent."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are an autonomous browser agent solving a 30-step web navigation gauntlet. \
You interact with web pages by reading the accessibility tree and issuing actions.

## Challenge Structure
Each step presents a unique puzzle (scroll challenges, timers, hidden elements, \
form filling, etc.). Solving the puzzle reveals a 6-character code. You must:
1. Solve the puzzle to reveal the code.
2. Find the code input textbox (labeled "Enter 6-character code").
3. Type the code into the textbox using fill(bid, "CODE").
4. Click the "Submit Code" button to advance to the next step.

## Traps & Distractions
- Many decoy buttons ("Next", "Continue", "Proceed", etc.) — these do NOT advance you.
- Popup overlays (cookie consent, "you won a prize", alerts) — close or dismiss them.
- Floating clickable elements ("Click Me!", "Link!", "Here!") — ignore these.
- Only the "Submit Code" button after entering the correct code advances the step.

## Observation Format
You receive the current page state as a pruned accessibility tree. Elements \
with a [bid] tag are interactive — use these bid values in your actions.

## Available Actions
{action_description}

## Rules
1. Issue exactly ONE action per turn.
2. Reason step-by-step in a <think> block, then output the action.
3. If a previous action failed, read the error and try a different approach.
4. Focus on finding the 6-character code and entering it. Ignore distractions.
5. The code is always exactly 6 alphanumeric characters (e.g., "KNYM9C").

## Output Format
<think>
[Your step-by-step reasoning about what you observe and what action to take]
</think>
action_here(arguments)
"""

TASK_PROMPT_GAUNTLET = """\
Complete all 30 steps of the Browser Navigation Challenge gauntlet. \
You are currently on step {current_step}. Each step has a unique puzzle \
that reveals a 6-character code. Enter the code and submit to advance.
"""

TASK_PROMPT_STEP = """\
Complete step {step_number} of the Browser Navigation Challenge. \
Find the 6-character code by solving the step's puzzle, enter it in the \
code input field, and click Submit Code to advance to step {next_step}.
"""


def format_system_prompt(action_description: str) -> str:
    """Format the system prompt with the action description."""
    return SYSTEM_PROMPT.format(action_description=action_description)


def format_task_prompt(step_number: int, base_url: str = "") -> str:
    """Format the task prompt for a specific step."""
    return TASK_PROMPT_STEP.format(
        step_number=step_number,
        next_step=step_number + 1,
    )


def format_gauntlet_task_prompt(current_step: int) -> str:
    """Format the task prompt for the full gauntlet."""
    return TASK_PROMPT_GAUNTLET.format(current_step=current_step)


def format_observation_message(observation_text: str, step: int) -> str:
    """Format an observation as a user message for the LLM."""
    return f"[Action {step}] Current page state:\n{observation_text}"


def parse_action_from_response(response: str) -> str:
    """Extract the action from an LLM response.

    Expects the action to appear after a </think> block (if present),
    or as the last non-empty line.
    """
    # Try to find content after </think>
    if "</think>" in response:
        after_think = response.split("</think>")[-1].strip()
        if after_think:
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
