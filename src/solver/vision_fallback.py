"""Gemini vision fallback for the deterministic solver.

When heuristics fail, takes a screenshot and asks Gemini what to do.
Executes the returned action directly via Playwright (sync API).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ActionType(Enum):
    CLICK = auto()
    TYPE = auto()
    SCROLL = auto()
    SCROLL_UP = auto()
    HOVER = auto()
    KEYBOARD = auto()
    WAIT = auto()
    EXTRACT_CODE = auto()
    CANVAS_DRAW = auto()
    NOOP = auto()


@dataclass
class VisionAction:
    action_type: ActionType = ActionType.NOOP
    target_selector: str = ""
    value: str = ""
    code_found: str = ""
    reasoning: str = ""


_VISION_PROMPT = """\
You are a web navigation agent solving a challenge puzzle. You see a screenshot of the page.

Current step: {step}/30
Attempt: {attempt}
Codes already tried (FAILED): {failed_codes}
DOM-extracted codes (may or may not be valid): {dom_codes}
Recent action history: {history}

HTML snippet (first 6000 chars):
```
{html}
```

TASK: Find a 6-character alphanumeric code hidden on this page. Enter it in the input field and submit to advance.

The code might be:
- Hidden in the DOM, data attributes, CSS, or comments
- Revealed after clicking a button, hovering an element, scrolling, or completing a mini-challenge
- In a shadow DOM, iframe, canvas, or dynamically loaded

Respond with a JSON object (no markdown fences):
{{
    "reasoning": "Brief analysis of what you see and what to try",
    "action_type": "CLICK|TYPE|SCROLL|SCROLL_UP|HOVER|KEYBOARD|WAIT|EXTRACT_CODE|CANVAS_DRAW",
    "target_selector": "CSS selector for the target element (if applicable)",
    "value": "value to type, key to press, etc. (if applicable)",
    "code_found": "6-char code if you found one, empty string otherwise"
}}
"""


class VisionFallback:
    """Gemini vision fallback — screenshot → action."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._client = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("GOOGLE_API_KEY not set — cannot use vision fallback")
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def analyze(
        self,
        screenshot_bytes: bytes,
        html_snippet: str,
        step: int,
        attempt: int,
        dom_codes: list[str],
        failed_codes: list[str],
        history: list[str],
    ) -> VisionAction:
        """Send screenshot + context to Gemini and return a structured action."""
        client = self._get_client()
        from google.genai import types

        prompt = _VISION_PROMPT.format(
            step=step,
            attempt=attempt,
            failed_codes=", ".join(failed_codes[-10:]) or "none",
            dom_codes=", ".join(dom_codes[:10]) or "none",
            history="; ".join(history[-5:]) or "none",
            html=html_snippet[:6000],
        )

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    types.Content(parts=[
                        types.Part.from_bytes(data=screenshot_bytes, mime_type="image/png"),
                        types.Part.from_text(text=prompt),
                    ]),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1024,
                ),
            )

            if response.usage_metadata:
                self._total_input_tokens += response.usage_metadata.prompt_token_count or 0
                self._total_output_tokens += response.usage_metadata.candidates_token_count or 0

            return self._parse_response(response.text)

        except Exception as e:
            logger.warning("Gemini vision error: %s", e)
            return VisionAction(reasoning=f"API error: {e}")

    def _parse_response(self, text: str) -> VisionAction:
        """Parse Gemini JSON response into a VisionAction."""
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text.strip())

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("Could not parse vision response: %.200s", text)
                    return VisionAction(reasoning=text[:200])
            else:
                logger.warning("No JSON in vision response: %.200s", text)
                return VisionAction(reasoning=text[:200])

        action_type_str = data.get("action_type", "NOOP").upper()
        try:
            action_type = ActionType[action_type_str]
        except KeyError:
            action_type = ActionType.NOOP

        return VisionAction(
            action_type=action_type,
            target_selector=data.get("target_selector", ""),
            value=data.get("value", ""),
            code_found=data.get("code_found", ""),
            reasoning=data.get("reasoning", ""),
        )

    @property
    def total_tokens(self) -> dict:
        return {
            "input": self._total_input_tokens,
            "output": self._total_output_tokens,
        }


def execute_vision_action(page, action: VisionAction) -> str:
    """Execute a VisionAction on a sync Playwright page. Returns description."""
    atype = action.action_type
    target = action.target_selector
    value = action.value

    try:
        if atype == ActionType.CLICK:
            if target:
                try:
                    page.click(target, timeout=2000)
                    return f"clicked {target}"
                except Exception:
                    page.evaluate(f"""() => {{
                        const el = document.querySelector('{target}');
                        if (el) el.click();
                    }}""")
                    return f"js-clicked {target}"
            return "click (no target)"

        elif atype == ActionType.TYPE:
            if target and value:
                try:
                    page.fill(target, value)
                except Exception:
                    page.evaluate(f"""(val) => {{
                        const el = document.querySelector('{target}');
                        if (el) {{
                            const s = Object.getOwnPropertyDescriptor(
                                HTMLInputElement.prototype, 'value').set;
                            s.call(el, val);
                            el.dispatchEvent(new Event('input', {{bubbles: true}}));
                            el.dispatchEvent(new Event('change', {{bubbles: true}}));
                        }}
                    }}""", value)
                return f"typed '{value}' in {target}"
            return "type (no target/value)"

        elif atype == ActionType.SCROLL:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(0.3)
            return "scrolled to bottom"

        elif atype == ActionType.SCROLL_UP:
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(0.3)
            return "scrolled to top"

        elif atype == ActionType.HOVER:
            if target:
                try:
                    loc = page.locator(target)
                    if loc.count() > 0:
                        loc.first.hover(timeout=2000)
                        time.sleep(1.5)
                        return f"hovered {target} for 1.5s"
                except Exception:
                    pass
                page.evaluate(f"""() => {{
                    const el = document.querySelector('{target}');
                    if (el) {{
                        el.scrollIntoView({{behavior: 'instant', block: 'center'}});
                        const rect = el.getBoundingClientRect();
                        const opts = {{bubbles: true, clientX: rect.x + rect.width/2,
                                       clientY: rect.y + rect.height/2}};
                        el.dispatchEvent(new MouseEvent('mouseenter', opts));
                        el.dispatchEvent(new MouseEvent('mouseover', opts));
                        el.dispatchEvent(new MouseEvent('mousemove', opts));
                    }}
                }}""")
                time.sleep(1.5)
                return f"js-hovered {target}"
            return "hover (no target)"

        elif atype == ActionType.KEYBOARD:
            if value:
                keys = [k.strip() for k in value.split(",")]
                page.evaluate("() => document.body.focus()")
                for key in keys:
                    page.keyboard.press(key.strip())
                    time.sleep(0.3)
                return f"pressed keys: {value}"
            return "keyboard (no value)"

        elif atype == ActionType.WAIT:
            time.sleep(1.0)
            return "waited 1s"

        elif atype == ActionType.EXTRACT_CODE:
            return "extract_code (handled by caller)"

        elif atype == ActionType.CANVAS_DRAW:
            return "canvas_draw (handled by caller)"

        else:
            return "noop"

    except Exception as e:
        return f"action error: {e}"
