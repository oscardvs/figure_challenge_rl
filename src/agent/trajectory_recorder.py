"""Recording proxy that intercepts solver Playwright calls and maps to BrowserGym actions.

The deterministic solver uses low-level Playwright calls (page.evaluate(),
page.keyboard.type(), page.mouse.click()) but the agent uses BrowserGym's
high-level actions (js_eval(), fill(), press(), mouse_click()). This module
bridges the gap by wrapping the Playwright Page object to transparently
capture and translate every action.

Usage:
    from src.agent.trajectory_recorder import RecordingPage, RecordedTrajectory

    recorder = RecordingPage(page, obs_extractor_fn, step_number=1, challenge_type="scroll")
    # Pass recorder to solver instead of raw page
    solver.solve_step(recorder, step_number=1)
    trajectory = recorder.get_trajectory(success=True, code_found="ABC123")
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Side-effect classification for page.evaluate() calls
# ---------------------------------------------------------------------------

# JS patterns that represent MEANINGFUL agent actions (not cleanup/utility).
# Only these are recorded as side-effect actions.
_AGENT_ACTION_PATTERNS = re.compile(
    r"window\.scrollTo|window\.scroll\("
    r"|\.scrollTop\s*=\s*\w"  # el.scrollTop = N (scroll container)
    r"|\.dispatchEvent\s*\(\s*new\s+(Drag|Mouse|Keyboard|Input|Pointer)Event"
    r"|\.value\s*=\s*['\"]"  # input.value = "code"
    r"|\.checked\s*=\s*true"  # radio.checked = true
    r"|\.focus\(\)",
    re.IGNORECASE,
)

# JS patterns that are UTILITY/CLEANUP — never recorded even if they
# technically modify the DOM. These are popup dismissal, overlay hiding, etc.
_UTILITY_PATTERNS = re.compile(
    r"\.style\.display\s*=\s*['\"]none"
    r"|\.style\.pointerEvents\s*=\s*['\"]none"
    r"|\.style\.visibility\s*=\s*['\"]hidden"
    r"|\.style\.zIndex\s*=\s*['\"]?-"
    r"|el\.remove\(\)|\.remove\(\)\s*;?\s*cleared"  # clear_popups pattern
    r"|let\s+cleared\s*=\s*0"  # clear_popups boilerplate
    r"|hide\s*\(\s*el\s*\)",
    re.IGNORECASE,
)

# Loose pattern: JS that MIGHT click buttons inside forEach loops (solver
# handler utility actions). Only recorded if NOT also matching utility.
_LOOSE_CLICK_PATTERN = re.compile(
    r"\.click\(\)",
    re.IGNORECASE,
)


@dataclass
class RecordedAction:
    """A single action recorded from the solver, mapped to BrowserGym format."""

    timestamp: float
    browsergym_action: str  # e.g. 'js_eval("...")', 'fill("42", "ABC123")'
    obs_text: str  # AXTree observation captured BEFORE this action
    side_effect: bool  # True if the action modifies DOM state


@dataclass
class RecordedTrajectory:
    """Complete trajectory for one step, ready for SFT/DPO conversion."""

    step_number: int
    challenge_type: str
    actions: list[RecordedAction] = field(default_factory=list)
    success: bool = False
    code_found: str | None = None
    elapsed_seconds: float = 0.0

    def to_sft_format(self, max_steps: int = 25) -> dict:
        """Convert to format expected by sft_train.py load_trajectories().

        Returns a dict with keys: step_number, success, steps.
        Each step has: step_index, obs_text, action, reasoning.

        Filters: only side-effect actions, deduplicated, capped at max_steps.
        """
        steps: list[dict] = []
        prev_action: str | None = None
        dup_count = 0

        for act in self.actions:
            if not act.side_effect:
                continue

            action = act.browsergym_action

            # Deduplicate consecutive identical actions.
            if action == prev_action:
                dup_count += 1
                # Allow at most 3 consecutive identical actions.
                if dup_count >= 3:
                    continue
            else:
                dup_count = 0

            prev_action = action
            steps.append({
                "step_index": len(steps),
                "obs_text": act.obs_text,
                "action": action,
                "reasoning": "",  # Filled in later by generate_cot.py
            })

            if len(steps) >= max_steps:
                break

        # Validate code_found — must be exactly 6 uppercase alphanumeric chars.
        # The solver sometimes attributes success to the wrong code (e.g.,
        # "Submit") when the URL happened to advance from an earlier fill.
        code = self.code_found
        if not code or not re.fullmatch(r"[A-Z0-9]{6}", code):
            code = _extract_code_from_actions(steps)

        return {
            "step_number": self.step_number,
            "challenge_type": self.challenge_type,
            "success": self.success,
            "code_found": code,
            "elapsed_seconds": self.elapsed_seconds,
            "steps": steps,
        }


_CODE_RE = re.compile(r'["\']([A-Z0-9]{6})["\']')


def _extract_code_from_actions(steps: list[dict]) -> str | None:
    """Try to find the 6-char code from fill() or js_eval() actions in the trajectory."""
    # First pass: look for fill() actions (most reliable).
    for step in reversed(steps):
        action = step.get("action", "")
        if action.startswith("fill("):
            m = _CODE_RE.search(action)
            if m:
                return m.group(1)
    # Second pass: look for js_eval() actions that inject a code into an input
    # (solver uses React-compatible setter: s.call(inp, 'CODE')).
    for step in reversed(steps):
        action = step.get("action", "")
        if action.startswith("js_eval(") and "input" in action.lower():
            m = _CODE_RE.search(action)
            if m:
                return m.group(1)
    return None


def _classify_js(js_code: str) -> bool:
    """Classify a JS expression: True = meaningful agent action, False = skip.

    Three-tier classification:
    1. If it matches utility patterns → skip (False)
    2. If it matches agent action patterns → record (True)
    3. If it has .click() but not utility → record (True)
    4. Otherwise → skip (False)
    """
    if not isinstance(js_code, str):
        return False

    # Utility/cleanup JS — never record.
    if _UTILITY_PATTERNS.search(js_code):
        return False

    # Explicit agent actions — always record.
    if _AGENT_ACTION_PATTERNS.search(js_code):
        return True

    # Loose click pattern: record if it clicks buttons (handler logic)
    # but doesn't look like utility.
    if _LOOSE_CLICK_PATTERN.search(js_code):
        return True

    return False


def _escape_for_action(s: str) -> str:
    """Escape a string for inclusion in a BrowserGym action argument."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


class RecordingKeyboard:
    """Proxy for page.keyboard that records type() and press() calls."""

    def __init__(self, real_keyboard, recorder: RecordingPage):
        self._keyboard = real_keyboard
        self._recorder = recorder

    def type(self, text: str, **kwargs):
        """Record as fill() action, then delegate."""
        focused_bid = self._recorder._get_focused_bid()
        action_str = f'fill("{focused_bid}", "{_escape_for_action(text)}")'
        self._recorder._record_action(action_str, side_effect=True)
        return self._keyboard.type(text, **kwargs)

    def press(self, key: str, **kwargs):
        """Record as press() action, then delegate."""
        focused_bid = self._recorder._get_focused_bid()
        action_str = f'press("{focused_bid}", "{_escape_for_action(key)}")'
        self._recorder._record_action(action_str, side_effect=True)
        return self._keyboard.press(key, **kwargs)

    def down(self, key: str, **kwargs):
        return self._keyboard.down(key, **kwargs)

    def up(self, key: str, **kwargs):
        return self._keyboard.up(key, **kwargs)

    def insert_text(self, text: str, **kwargs):
        return self._keyboard.insert_text(text, **kwargs)


class RecordingMouse:
    """Proxy for page.mouse that records click/move/drag/wheel calls."""

    def __init__(self, real_mouse, recorder: RecordingPage):
        self._mouse = real_mouse
        self._recorder = recorder
        self._drag_start: tuple[float, float] | None = None
        self._is_down = False
        self._last_move: tuple[float, float] = (0, 0)

    def click(self, x: float, y: float, **kwargs):
        """Record as mouse_click() action."""
        action_str = f'mouse_click({x:.0f}, {y:.0f})'
        self._recorder._record_action(action_str, side_effect=True)
        return self._mouse.click(x, y, **kwargs)

    def dblclick(self, x: float, y: float, **kwargs):
        action_str = f'mouse_click({x:.0f}, {y:.0f})'
        self._recorder._record_action(action_str, side_effect=True)
        return self._mouse.dblclick(x, y, **kwargs)

    def move(self, x: float, y: float, **kwargs):
        """Record mouse_move or accumulate for drag."""
        self._last_move = (x, y)
        if self._is_down:
            # Part of a drag — will be recorded on mouse.up()
            return self._mouse.move(x, y, **kwargs)
        action_str = f'mouse_move({x:.0f}, {y:.0f})'
        self._recorder._record_action(action_str, side_effect=True)
        return self._mouse.move(x, y, **kwargs)

    def down(self, **kwargs):
        """Start of a drag — record start position."""
        self._is_down = True
        self._drag_start = self._last_move
        return self._mouse.down(**kwargs)

    def up(self, **kwargs):
        """End of a drag — record as mouse_drag()."""
        result = self._mouse.up(**kwargs)
        if self._is_down and self._drag_start is not None:
            sx, sy = self._drag_start
            ex, ey = self._last_move
            action_str = f'mouse_drag({sx:.0f}, {sy:.0f}, {ex:.0f}, {ey:.0f})'
            self._recorder._record_action(action_str, side_effect=True)
        self._is_down = False
        self._drag_start = None
        return result

    def wheel(self, delta_x: float, delta_y: float, **kwargs):
        """Record as scroll() action."""
        action_str = f'scroll({delta_x:.0f}, {delta_y:.0f})'
        self._recorder._record_action(action_str, side_effect=True)
        return self._mouse.wheel(delta_x, delta_y, **kwargs)

    def __getattr__(self, name):
        return getattr(self._mouse, name)


class RecordingLocator:
    """Proxy for Playwright Locator that records interactions."""

    def __init__(self, real_locator, recorder: RecordingPage):
        self._locator = real_locator
        self._recorder = recorder

    def fill(self, value: str, **kwargs):
        """Record as fill() action."""
        bid = self._recorder._get_focused_bid()
        action_str = f'fill("{bid}", "{_escape_for_action(value)}")'
        self._recorder._record_action(action_str, side_effect=True)
        return self._locator.fill(value, **kwargs)

    def click(self, **kwargs):
        """Record as click() action."""
        bid = self._recorder._get_focused_bid()
        action_str = f'click("{bid}")'
        self._recorder._record_action(action_str, side_effect=True)
        return self._locator.click(**kwargs)

    @property
    def first(self):
        return RecordingLocator(self._locator.first, self._recorder)

    def __getattr__(self, name):
        attr = getattr(self._locator, name)
        if callable(attr):
            return attr
        return attr


class RecordingPage:
    """Transparent proxy wrapping a Playwright Page.

    Intercepts solver Playwright calls, captures observations before each
    action, and maps calls to BrowserGym action format.

    Observation capture is throttled: at most once every `obs_interval`
    seconds, reusing the last observation for intermediate actions.
    """

    def __init__(
        self,
        real_page,
        obs_extractor_fn,
        step_number: int,
        challenge_type: str,
        obs_interval: float = 2.0,
    ):
        """
        Args:
            real_page: The actual sync Playwright Page object.
            obs_extractor_fn: Callable that returns current obs_text string.
                Signature: () -> str
            step_number: Current step number being solved.
            challenge_type: Detected challenge type string.
            obs_interval: Minimum seconds between observation captures.
        """
        self._page = real_page
        self._obs_extractor_fn = obs_extractor_fn
        self._step_number = step_number
        self._challenge_type = challenge_type
        self._obs_interval = obs_interval
        self._actions: list[RecordedAction] = []
        self._start_time = time.time()
        self._last_obs: str = ""
        self._last_obs_time: float = 0.0

        # Initialize proxies.
        self.keyboard = RecordingKeyboard(real_page.keyboard, self)
        self.mouse = RecordingMouse(real_page.mouse, self)

    def _get_focused_bid(self) -> str:
        """Get the BrowserGym bid of the currently focused element."""
        try:
            bid = self._page.evaluate(
                "() => document.activeElement?.getAttribute('bid') || ''"
            )
            return str(bid) if bid else ""
        except Exception:
            logger.debug("Could not get focused element bid")
            return ""

    def _capture_obs(self) -> str:
        """Capture current observation, throttled by obs_interval.

        Only calls the (expensive) obs_extractor_fn if enough time has
        elapsed since the last capture. Otherwise reuses the cached obs.
        """
        now = time.time()
        if now - self._last_obs_time >= self._obs_interval or not self._last_obs:
            try:
                self._last_obs = self._obs_extractor_fn()
                self._last_obs_time = now
            except Exception as e:
                logger.warning(f"Observation capture failed: {e}")
        return self._last_obs

    def _record_action(self, action_str: str, side_effect: bool = True):
        """Record an action with its pre-action observation."""
        obs = self._capture_obs() if side_effect else self._last_obs
        self._actions.append(RecordedAction(
            timestamp=time.time(),
            browsergym_action=action_str,
            obs_text=obs,
            side_effect=side_effect,
        ))

    def evaluate(self, expression, arg=None):
        """Intercept page.evaluate() calls.

        Uses _classify_js() to determine if the JS represents a meaningful
        agent action (scrollTo, button click, form fill) vs utility/cleanup
        (popup dismissal, style changes). Only meaningful actions are recorded.
        """
        is_agent_action = _classify_js(
            expression if isinstance(expression, str) else ""
        )

        if is_agent_action:
            escaped = _escape_for_action(
                expression if isinstance(expression, str) else str(expression)
            )
            action_str = f'js_eval("{escaped}")'
            self._record_action(action_str, side_effect=True)

        if arg is not None:
            return self._page.evaluate(expression, arg)
        return self._page.evaluate(expression)

    def locator(self, selector, **kwargs):
        """Return a RecordingLocator proxy."""
        real_locator = self._page.locator(selector, **kwargs)
        return RecordingLocator(real_locator, self)

    def content(self):
        return self._page.content()

    def wait_for_selector(self, selector, **kwargs):
        return self._page.wait_for_selector(selector, **kwargs)

    def wait_for_url(self, url, **kwargs):
        return self._page.wait_for_url(url, **kwargs)

    def wait_for_timeout(self, timeout):
        return self._page.wait_for_timeout(timeout)

    def goto(self, url, **kwargs):
        return self._page.goto(url, **kwargs)

    def reload(self, **kwargs):
        return self._page.reload(**kwargs)

    def get_by_role(self, role, **kwargs):
        return self._page.get_by_role(role, **kwargs)

    def get_by_placeholder(self, text, **kwargs):
        return self._page.get_by_placeholder(text, **kwargs)

    def get_by_text(self, text, **kwargs):
        return self._page.get_by_text(text, **kwargs)

    def inner_text(self, selector, **kwargs):
        return self._page.inner_text(selector, **kwargs)

    def query_selector(self, selector):
        return self._page.query_selector(selector)

    def query_selector_all(self, selector):
        return self._page.query_selector_all(selector)

    @property
    def url(self) -> str:
        return self._page.url

    @property
    def frames(self):
        return self._page.frames

    def get_trajectory(
        self,
        success: bool = False,
        code_found: str | None = None,
    ) -> RecordedTrajectory:
        """Build the complete trajectory from recorded actions."""
        return RecordedTrajectory(
            step_number=self._step_number,
            challenge_type=self._challenge_type,
            actions=list(self._actions),
            success=success,
            code_found=code_found,
            elapsed_seconds=time.time() - self._start_time,
        )

    def __getattr__(self, name):
        """Forward any unhandled attribute access to the real page."""
        return getattr(self._page, name)
