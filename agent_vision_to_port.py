"""Gemini 3 vision analyzer for the agent-based challenge solver.

Uses thinking_level (not legacy thinking_budget) and supports both
gemini-3-flash-preview and gemini-3-pro-preview for escalation.
"""
import json
from enum import Enum
from typing import Optional
from pydantic import BaseModel
from google import genai
from google.genai import types


class ActionType(str, Enum):
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    SCROLL_UP = "scroll_up"
    WAIT = "wait"
    HOVER = "hover"
    KEYBOARD = "keyboard"
    EXTRACT_CODE = "extract_code"
    CLICK_REVEAL = "click_reveal"
    CANVAS_DRAW = "canvas_draw"


class ActionResponse(BaseModel):
    action_type: ActionType
    target_selector: Optional[str] = None
    value: Optional[str] = None
    reasoning: str
    code_found: Optional[str] = None
    confidence: float = 0.0


# Escalation: (model, thinking_level)
ESCALATION = [
    ("gemini-3-flash-preview", "low"),
    ("gemini-3-flash-preview", "low"),
    ("gemini-3-flash-preview", "low"),
    ("gemini-3-flash-preview", "medium"),
    ("gemini-3-flash-preview", "medium"),
    ("gemini-3-flash-preview", "medium"),
    ("gemini-3-flash-preview", "high"),
    ("gemini-3-flash-preview", "high"),
    ("gemini-3-flash-preview", "high"),
    ("gemini-3-pro-preview", "high"),
    ("gemini-3-pro-preview", "high"),
    ("gemini-3-pro-preview", "high"),
]

SYSTEM_PROMPT = """You are an expert browser automation agent solving a 30-step navigation challenge.
Each step hides a 6-character alphanumeric code (uppercase A-Z and digits 0-9, like "TWA8Q7" or "P4HWBQ").
You must find the code and enter it to proceed.

CRITICAL RULES:
- The code is ALWAYS exactly 6 characters: uppercase letters A-Z and digits 0-9
- Real codes almost always contain BOTH letters AND digits (e.g., "3KW9PL", not "BUTTON")
- NEVER suggest common English/Latin words: BUTTON, SCROLL, HIDDEN, ACCEPT, COOKIE, ALIQUA, BEATAE, CILLUM, DOLORE, FUGIAT, LABORE, MOLLIT, TEMPOR, VENIAM, SUBMIT, OPTION, REVEAL, CHOICE, CANVAS, PUZZLE, etc.
- If a code was already tried and FAILED, do NOT suggest it again

CHALLENGE TYPES (the page may have multiple overlaid):
1. SCROLL REVEAL: Must scroll down 500+ px. Look for "Scroll to Reveal" or progress bar showing scroll distance.
2. CLICK REVEAL: "Click the button to reveal the code" - click the Reveal Code button
3. HOVER REVEAL: Hover over a highlighted/bordered element for 1+ seconds
4. HIDDEN DOM: Code in data-* attrs, aria-* attrs, HTML comments, Base64 strings
5. RADIO MODAL: Modal with radio options + "Submit & Continue" button. Usually one option says "correct"
6. KEYBOARD SEQUENCE: Press key combos like Control+A, Shift+K shown on page
7. CANVAS DRAWING: Draw strokes/shapes on a canvas element
8. TIMING/COUNTDOWN: Wait for timer, then click "Capture Now!"
9. AUDIO: Click Play, wait for speech, click Complete
10. DRAG AND DROP: Drag pieces into slots
11. SPLIT PARTS: Click scattered "Part N" elements across the page
12. ROTATING CODE: Click "Capture" multiple times as code rotates
13. MULTI-TAB: Click tab buttons to collect code parts
14. MATH PUZZLE: Solve expression (e.g., "24 + 18 = ?"), type answer, click Solve
15. VIDEO FRAMES: Navigate to target frame using +1/-1/+10/-10 buttons
16. SEQUENCE: Perform 4 actions in order: click button, hover area, type text, scroll box
17. FAKE POPUPS: Popups with fake close buttons - our code handles these automatically
18. TRAP BUTTONS: Many "Proceed"/"Continue"/"Next" buttons are traps - don't click them!

IMPORTANT TRICKS:
- Many buttons labeled "Proceed", "Continue", "Next Step" etc. are TRAPS - clicking them does nothing or triggers "Wrong Button" popups
- The REAL navigation happens by entering the correct code and clicking the Submit button near the input
- Popups saying "This close button is fake!" should be ignored (our code hides them)
- On some pages you need to SCROLL DOWN to reveal the code or a hidden section
- "Click here X more times to reveal" means click that specific text element multiple times

CSS SELECTORS ONLY - do NOT use Playwright :has-text() selectors.
Examples: .class, #id, button[type="submit"], input[type="text"], [data-testid="foo"]

RESPONSE FORMAT (JSON, no markdown):
{"action_type":"click|type|scroll|scroll_up|hover|wait|keyboard|click_reveal|extract_code|canvas_draw","target_selector":"CSS selector or null","value":"text to type, key combo like Control+A, or shape name (square/circle/triangle)","code_found":"ABC123 or null","reasoning":"what you see and plan to do","confidence":0.0-1.0}"""


class AgentVision:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.total_tokens_in = 0
        self.total_tokens_out = 0

    def analyze(
        self,
        screenshot_bytes: bytes,
        html_snippet: str,
        step: int,
        attempt: int,
        dom_codes: list[str],
        failed_codes: list[str],
        history: list[str],
    ) -> tuple[ActionResponse, int, int]:
        """Analyze page screenshot and return next action.

        Args:
            screenshot_bytes: PNG screenshot
            html_snippet: First 6000 chars of HTML
            step: Current challenge step (1-30)
            attempt: Current attempt number (0-based)
            dom_codes: Codes extracted from DOM
            failed_codes: Codes already tried and failed
            history: List of previous action descriptions for context

        Returns: (action, input_tokens, output_tokens)
        """
        # Select model and thinking level based on attempt
        esc_idx = min(attempt, len(ESCALATION) - 1)
        model_name, thinking_level = ESCALATION[esc_idx]

        # Build context
        context_parts = []
        context_parts.append(f"Step {step}/30, attempt {attempt + 1}")
        if dom_codes:
            context_parts.append(f"DOM codes found: {dom_codes}")
        if failed_codes:
            context_parts.append(f"FAILED codes (do NOT suggest): {failed_codes}")
        if history:
            context_parts.append(f"Previous actions this step: {'; '.join(history[-5:])}")

        user_prompt = f"""{chr(10).join(context_parts)}

Look at the screenshot carefully. What do you see? What action should I take next to find and submit the 6-character code?

HTML (truncated):
{html_snippet}"""

        print(f"    [agent] {model_name} thinking={thinking_level}, attempt={attempt+1}", flush=True)

        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=screenshot_bytes, mime_type="image/png"),
                            types.Part.from_text(text=user_prompt),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                ),
            )
        except Exception as e:
            # Fallback: if thinking_level fails (API version issue), try thinking_budget
            print(f"    [agent] thinking_level failed ({e}), trying thinking_budget fallback", flush=True)
            budget_map = {"minimal": 1024, "low": 2048, "medium": 4096, "high": 8192}
            budget = budget_map.get(thinking_level, 4096)
            response = self.client.models.generate_content(
                model=model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=screenshot_bytes, mime_type="image/png"),
                            types.Part.from_text(text=user_prompt),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=budget),
                ),
            )

        tokens_in = response.usage_metadata.prompt_token_count or 0
        tokens_out = response.usage_metadata.candidates_token_count or 0
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out

        print(f"    [agent] tokens: {tokens_in} in / {tokens_out} out", flush=True)

        # Parse response
        try:
            text = response.text.strip()
            # Remove markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0].strip()
            # Extract first valid JSON object
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                depth = 0
                end_idx = 0
                for i, ch in enumerate(text):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end_idx = i + 1
                            break
                if end_idx > 0:
                    data = json.loads(text[:end_idx])
                else:
                    raise

            action = ActionResponse(**data)
            print(f"    [agent] -> {action.action_type} target={action.target_selector} code={action.code_found}", flush=True)
            print(f"    [agent] reasoning: {action.reasoning[:150]}", flush=True)
        except Exception as e:
            print(f"    [agent] PARSE ERROR: {e}, raw={text[:200] if 'text' in dir() else 'N/A'}", flush=True)
            action = ActionResponse(
                action_type=ActionType.SCROLL,
                reasoning=f"Parse error, scrolling as fallback: {e}",
                confidence=0.0,
            )

        return action, tokens_in, tokens_out
