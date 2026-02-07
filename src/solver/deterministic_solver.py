"""Orchestrates the per-step deterministic solve loop.

Ported from ``AgentChallengeSolver._solve_step()`` in
``agent_solver_to_port.py``.  All Playwright calls are **synchronous**.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from src.solver.challenge_detector import ChallengeDetector, ChallengeType
from src.solver.challenge_handlers import (
    ChallengeHandlers,
    FALSE_POSITIVES,
    check_progress,
    clear_popups,
    deep_code_extraction,
    extract_hidden_codes,
    fill_and_submit,
    sort_codes_by_priority,
)

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    success: bool = False
    code_found: str | None = None
    challenge_type: ChallengeType = ChallengeType.UNKNOWN
    actions_log: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    attempts: int = 0


class DeterministicSolver:
    """Deterministic solver — no LLM calls, pure heuristics + Playwright."""

    def __init__(self, max_attempts: int = 15, step_timeout: float = 25.0):
        self.max_attempts = max_attempts
        self.step_timeout = step_timeout
        self.detector = ChallengeDetector()
        self.handlers = ChallengeHandlers()

    def solve_step(self, page, step_number: int) -> SolveResult:
        """Solve the current step.  *page* must already be at the step."""
        t0 = time.time()
        result = SolveResult()
        failed_codes: list[str] = []
        submit_is_trap = False
        scroll_attempted = False

        # Wait for content
        for _ in range(10):
            html = page.content()
            if len(html) > 1000 and ("button" in html.lower() or "input" in html.lower()):
                break
            time.sleep(0.5)

        for attempt in range(self.max_attempts):
            elapsed = time.time() - t0
            if elapsed > self.step_timeout:
                logger.info("Step %d: timeout after %.1fs", step_number, elapsed)
                break

            # 1. Check if already advanced
            if check_progress(page.url, step_number):
                result.success = True
                break

            # 2. Clear popups
            cleared = clear_popups(page)
            if cleared:
                time.sleep(0.2)

            # 3. Attempt 0 — fast path
            if attempt == 0:
                # Scroll triggers
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(0.3)
                page.evaluate("window.scrollTo(0, 0)")
                time.sleep(0.2)
                page.evaluate("window.scrollTo(0, 1000)")
                time.sleep(0.3)

                # Click reveal / accept / register / connect / trigger buttons
                page.evaluate("""\
(() => {
    document.querySelectorAll('button').forEach(btn => {
        const t = btn.textContent.toLowerCase();
        if ((t.includes('reveal') || t.includes('accept') ||
             t.includes('register') || t.includes('retrieve') ||
             t.includes('connect') || t.includes('trigger')) &&
            btn.offsetParent && !btn.disabled) btn.click();
    });
})()""")
                time.sleep(0.3)

                # Click "click here to reveal" elements
                for _ in range(5):
                    clicked = page.evaluate("""\
(() => {
    let clicked = 0;
    document.querySelectorAll('div, p, span').forEach(el => {
        const text = el.textContent || '';
        if (text.includes('click here') && text.includes('to reveal')) { el.click(); clicked++; }
    });
    return clicked;
})()""")
                    if clicked == 0:
                        break
                    time.sleep(0.2)

                # Scroll modal containers
                page.evaluate("""\
(() => {
    document.querySelectorAll('[class*="overflow-y"], [class*="overflow-auto"], [class*="max-h"]').forEach(el => {
        if (el.scrollHeight > el.clientHeight) el.scrollTop = el.scrollHeight;
    });
})()""")

                # Try DOM codes
                html = page.content()
                codes = extract_hidden_codes(html)
                if codes:
                    logger.info("Step %d: DOM codes: %s", step_number, codes[:8])
                    for code in codes:
                        if code in failed_codes:
                            continue
                        ok, submit_is_trap = fill_and_submit(page, code, step_number, submit_is_trap)
                        if ok:
                            result.success = True
                            result.code_found = code
                            break
                        failed_codes.append(code)
                    if result.success:
                        break

                # Radio brute force
                hr = self.handlers.handle_radio_brute_force(page, step_number)
                if hr.success:
                    result.success = True
                    break

                # Get body text for challenge detection
                html_text = page.evaluate("() => document.body.textContent || ''")
                html_lower = html_text.lower()

                # Keyboard sequence
                if "keyboard sequence" in html_lower or ("press" in html_lower and "keys" in html_lower):
                    self.handlers.handle_keyboard_sequence(page)

                # Math puzzle
                if "puzzle" in html_lower and ("= ?" in html_text or "=?" in html_text):
                    hr = self.handlers.handle_math_puzzle(page, failed_codes)
                    for code in hr.codes_found:
                        if code in failed_codes:
                            continue
                        ok, submit_is_trap = fill_and_submit(page, code, step_number, submit_is_trap)
                        if ok:
                            result.success = True
                            result.code_found = code
                            break
                        failed_codes.append(code)
                    if result.success:
                        break
                    # Post-math deep extraction
                    post_math = deep_code_extraction(page, set(failed_codes))
                    for code in post_math[:5]:
                        if code in failed_codes:
                            continue
                        ok, submit_is_trap = fill_and_submit(page, code, step_number, submit_is_trap)
                        if ok:
                            result.success = True
                            result.code_found = code
                            break
                        failed_codes.append(code)
                    if result.success:
                        break

                # Timing capture
                if "capture" in html_lower and ("timing" in html_lower or "second" in html_lower):
                    self.handlers.handle_timing_capture(page)

                # Hover reveal
                if "hover" in html_lower and ("reveal" in html_lower or "code" in html_lower):
                    self.handlers.handle_hover_reveal(page)

                # "I Remember" buttons
                page.evaluate("""\
(() => {
    document.querySelectorAll('button').forEach(btn => {
        const t = (btn.textContent || '').trim().toLowerCase();
        if (t.includes('i remember') && btn.offsetParent && !btn.disabled) btn.click();
    });
})()""")

                # Audio
                if "audio" in html_lower and ("play" in html_lower or "listen" in html_lower):
                    self.handlers.handle_audio(page)

                # Canvas
                has_canvas = page.evaluate("() => !!document.querySelector('canvas')")
                if has_canvas and ("draw" in html_lower or "canvas" in html_lower or "stroke" in html_lower):
                    self.handlers.handle_canvas_draw(page)

                # Service Worker
                if "service worker" in html_lower or ("register" in html_lower and "cache" in html_lower):
                    hr = self.handlers.handle_service_worker(page)
                    for code in hr.codes_found:
                        if code in failed_codes:
                            continue
                        ok, submit_is_trap = fill_and_submit(page, code, step_number, submit_is_trap)
                        if ok:
                            result.success = True
                            result.code_found = code
                            break
                        failed_codes.append(code)
                    if result.success:
                        break

                # Shadow DOM
                if "shadow" in html_lower and ("layer" in html_lower or "level" in html_lower or "nested" in html_lower):
                    hr = self.handlers.handle_shadow_dom(page)
                    for code in hr.codes_found:
                        if code in failed_codes:
                            continue
                        ok, submit_is_trap = fill_and_submit(page, code, step_number, submit_is_trap)
                        if ok:
                            result.success = True
                            result.code_found = code
                            break
                        failed_codes.append(code)
                    if result.success:
                        break

                # WebSocket
                if "websocket" in html_lower or ("connect" in html_lower and "server" in html_lower):
                    hr = self.handlers.handle_websocket(page)
                    for code in hr.codes_found:
                        if code in failed_codes:
                            continue
                        ok, submit_is_trap = fill_and_submit(page, code, step_number, submit_is_trap)
                        if ok:
                            result.success = True
                            result.code_found = code
                            break
                        failed_codes.append(code)
                    if result.success:
                        break

                # Delayed Reveal
                if "delayed" in html_lower and ("reveal" in html_lower or "remaining" in html_lower or "wait" in html_lower):
                    hr = self.handlers.handle_delayed_reveal(page)
                    for code in hr.codes_found:
                        if code in failed_codes:
                            continue
                        ok, submit_is_trap = fill_and_submit(page, code, step_number, submit_is_trap)
                        if ok:
                            result.success = True
                            result.code_found = code
                            break
                        failed_codes.append(code)
                    if result.success:
                        break

                # Re-evaluate for late-loading challenges
                html_text = page.evaluate("() => document.body.textContent || ''")
                html_lower = html_text.lower()

                # Detect by button text
                challenge_buttons = page.evaluate("""\
(() => {
    const btns = [...document.querySelectorAll('button')].filter(b => b.offsetParent && !b.disabled);
    const texts = btns.map(b => b.textContent.trim().toLowerCase());
    return {
        hasTrigger: texts.some(t => t.includes('trigger mutation') || t.includes('trigger')),
        hasGoDeeper: texts.some(t => t.includes('go deeper') || t.includes('enter level') || t.includes('next level')),
        hasExtractCode: texts.some(t => t.includes('extract code')),
    };
})()""")

                # Mutation
                if "mutation" in html_lower or (challenge_buttons and challenge_buttons.get("hasTrigger")):
                    hr = self.handlers.handle_mutation(page)
                    for code in hr.codes_found:
                        if code in failed_codes:
                            continue
                        ok, submit_is_trap = fill_and_submit(page, code, step_number, submit_is_trap)
                        if ok:
                            result.success = True
                            result.code_found = code
                            break
                        failed_codes.append(code)
                    if result.success:
                        break

                # Iframe
                if ("iframe" in html_lower and ("level" in html_lower or "nested" in html_lower or "depth" in html_lower or "recursive" in html_lower)) or \
                   (challenge_buttons and (challenge_buttons.get("hasGoDeeper") or challenge_buttons.get("hasExtractCode"))):
                    hr = self.handlers.handle_iframe(page)
                    for code in hr.codes_found:
                        if code in failed_codes:
                            continue
                        ok, submit_is_trap = fill_and_submit(page, code, step_number, submit_is_trap)
                        if ok:
                            result.success = True
                            result.code_found = code
                            break
                        failed_codes.append(code)
                    if result.success:
                        break

                # Split parts
                if "part" in html_lower and ("found" in html_lower or "collect" in html_lower):
                    self.handlers.handle_split_parts(page)

                # Rotating code
                if "rotat" in html_lower and "capture" in html_lower:
                    self.handlers.handle_rotating_code(page)

                # Multi-tab
                if "tab" in html_lower and ("click" in html_lower or "visit" in html_lower):
                    self.handlers.handle_multi_tab(page)

                # Sequence
                if "sequence" in html_lower or ("click" in html_lower and "hover" in html_lower and "type" in html_lower):
                    self.handlers.handle_sequence(page)

                # Video frames
                if "frame" in html_lower and ("navigate" in html_lower or "+1" in html_text or "-1" in html_text):
                    self.handlers.handle_video_frames(page)

                # Deep code extraction
                deep_codes = deep_code_extraction(page, set(failed_codes))
                if deep_codes:
                    logger.info("Step %d: deep codes: %s", step_number, deep_codes[:8])
                    for code in deep_codes[:10]:
                        if code in failed_codes:
                            continue
                        ok, submit_is_trap = fill_and_submit(page, code, step_number, submit_is_trap)
                        if ok:
                            result.success = True
                            result.code_found = code
                            break
                        failed_codes.append(code)
                    if result.success:
                        break

                # Submit-is-trap: animated button
                if submit_is_trap:
                    time.sleep(1.5)
                    fresh_deep = deep_code_extraction(page, set(failed_codes))
                    all_to_try = list(dict.fromkeys(
                        (fresh_deep or []) + (deep_codes or []) + list(codes or []) + list(failed_codes)
                    ))
                    for code in all_to_try[:10]:
                        if _try_animated_button_submit_with_check(page, code, step_number):
                            result.success = True
                            result.code_found = code
                            break
                    if result.success:
                        break

                # Scroll-to-find
                is_scroll_challenge = page.evaluate("""\
(() => {
    const bodyText = (document.body.textContent || '').toLowerCase();
    const hasCanvas = !!document.querySelector('canvas');
    const hasAudio = bodyText.includes('audio challenge') || (bodyText.includes('play audio') && bodyText.includes('complete'));
    const hasDrag = document.querySelectorAll('[draggable="true"]').length >= 3;
    if (hasCanvas || hasAudio || hasDrag) return false;
    const els = document.querySelectorAll('h1, h2, h3, .text-2xl, .text-3xl, .text-xl, .font-bold, .text-lg');
    for (const el of els) {
        const t = (el.textContent || '').toLowerCase();
        if (t.includes('scroll down to find') || t.includes('scroll to find')) return true;
    }
    const mainBox = document.querySelector('.max-w-6xl, .max-w-4xl, .max-w-3xl');
    if (mainBox) {
        const t = (mainBox.textContent || '').toLowerCase();
        if (t.includes('scroll down') && (t.includes('navigation') || t.includes('navigate') || t.includes('nav button'))) return true;
    }
    if (bodyText.includes('keep scrolling') && bodyText.includes('navigation button')) return true;
    if (document.body.scrollHeight > 5000) {
        const sectionDivs = [...document.querySelectorAll('div')].filter(el => {
            const t = (el.textContent || '').trim();
            return t.match(/^Section \\d+/) && t.length > 50;
        });
        if (sectionDivs.length > 10) return true;
    }
    return false;
})()""")
                if is_scroll_challenge and not scroll_attempted:
                    scroll_attempted = True
                    self.handlers.handle_scroll_to_find(page, failed_codes)
                    if check_progress(page.url, step_number):
                        result.success = True
                        break

                # Drag-and-drop inline JS
                page.evaluate("""\
(() => {
    document.querySelectorAll('div, button, a, span').forEach(el => {
        const style = getComputedStyle(el);
        const text = (el.textContent || '').trim();
        if (style.position === 'absolute' || style.position === 'fixed') {
            if (['Click Me!', 'Button!', 'Link!', 'Here!', 'Click Here', 'Click Here!', 'Try This!'].includes(text)) {
                el.style.display = 'none'; el.style.pointerEvents = 'none';
            }
        }
    });
    const pieces = [...document.querySelectorAll('[draggable="true"]')];
    const slots = [...document.querySelectorAll('div')].filter(el => {
        const text = (el.textContent || '').trim();
        const cls = el.getAttribute('class') || '';
        const style = el.getAttribute('style') || '';
        return (text.match(/^Slot \\d+$/) &&
               (cls.includes('dashed') || cls.includes('border-dashed') || style.includes('dashed'))) ||
               (cls.includes('border-dashed') && el.children.length <= 2 && el.offsetWidth > 40);
    });
    const n = Math.min(pieces.length, slots.length, 6);
    for (let i = 0; i < n; i++) {
        try {
            const dt = new DataTransfer();
            dt.setData('text/plain', pieces[i].textContent.trim());
            pieces[i].dispatchEvent(new DragEvent('dragstart', {dataTransfer: dt, bubbles: true, cancelable: true}));
            slots[i].dispatchEvent(new DragEvent('dragenter', {dataTransfer: dt, bubbles: true, cancelable: true}));
            slots[i].dispatchEvent(new DragEvent('dragover', {dataTransfer: dt, bubbles: true, cancelable: true}));
            slots[i].dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true, cancelable: true}));
            pieces[i].dispatchEvent(new DragEvent('dragend', {dataTransfer: dt, bubbles: true, cancelable: true}));
        } catch(e) {}
    }
    document.querySelectorAll('button').forEach(btn => {
        const t = (btn.textContent || '').trim().toLowerCase();
        if ((t.includes('complete') || t.includes('done') || t.includes('verify')) &&
            !t.includes('clear') && btn.offsetParent && !btn.disabled) btn.click();
    });
})()""")
                time.sleep(0.3)

                # Re-extract after all fast-path actions
                html = page.content()
                codes = extract_hidden_codes(html)
                for code in codes:
                    if code in failed_codes:
                        continue
                    ok, submit_is_trap = fill_and_submit(page, code, step_number, submit_is_trap)
                    if ok:
                        result.success = True
                        result.code_found = code
                        break
                    failed_codes.append(code)
                if result.success:
                    break

                # Check progress after fast path
                if check_progress(page.url, step_number):
                    result.success = True
                    break

                # Trap button heuristic
                if failed_codes:
                    trap_count = page.evaluate("""\
(() => {
    const TRAPS = ['proceed', 'continue', 'next step', 'next page', 'next section'];
    return [...document.querySelectorAll('button')].filter(b => {
        const t = (b.textContent || '').trim().toLowerCase();
        return t.length < 40 && TRAPS.some(w => t.includes(w));
    }).length;
})()""")
                    if trap_count >= 8 and not scroll_attempted:
                        scroll_attempted = True
                        self.handlers.handle_scroll_to_find(page, failed_codes)
                        if check_progress(page.url, step_number):
                            result.success = True
                            break

                result.actions_log.append(f"fast_path attempt 0 done")
                continue

            # 4. Detection-based handling for subsequent attempts
            detections = self.detector.detect(page)
            if detections:
                best = detections[0]
                result.challenge_type = best.challenge_type
                logger.info("Step %d attempt %d: detected %s (%.2f)",
                            step_number, attempt, best.challenge_type.name, best.confidence)

                hr = self.handlers.handle(page, best.challenge_type, step_number, failed_codes)
                result.actions_log.extend(hr.actions_log)

                if hr.success:
                    result.success = True
                    if hr.codes_found:
                        result.code_found = hr.codes_found[0]
                    break

                # Try codes from handler
                for code in hr.codes_found:
                    if code in failed_codes:
                        continue
                    ok, submit_is_trap = fill_and_submit(page, code, step_number, submit_is_trap)
                    if ok:
                        result.success = True
                        result.code_found = code
                        break
                    failed_codes.append(code)
                if result.success:
                    break

            # 5. Re-extract after handler actions
            html = page.content()
            new_codes = extract_hidden_codes(html)
            for code in new_codes:
                if code in failed_codes:
                    continue
                ok, submit_is_trap = fill_and_submit(page, code, step_number, submit_is_trap)
                if ok:
                    result.success = True
                    result.code_found = code
                    break
                if submit_is_trap:
                    if _try_animated_button_submit_with_check(page, code, step_number):
                        result.success = True
                        result.code_found = code
                        break
                failed_codes.append(code)
            if result.success:
                break

            # 6. Deep extraction
            if attempt >= 2:
                deep = deep_code_extraction(page, set(failed_codes))
                for code in deep[:5]:
                    if code in failed_codes:
                        continue
                    ok, submit_is_trap = fill_and_submit(page, code, step_number, submit_is_trap)
                    if ok:
                        result.success = True
                        result.code_found = code
                        break
                    if submit_is_trap:
                        if _try_animated_button_submit_with_check(page, code, step_number):
                            result.success = True
                            result.code_found = code
                            break
                    failed_codes.append(code)
                if result.success:
                    break

            # 7. Check progress
            if check_progress(page.url, step_number):
                result.success = True
                break

            # 8. Periodic retries
            if attempt >= 3 and attempt % 3 == 0:
                html_text = page.evaluate("() => document.body.textContent || ''")
                html_lower = html_text.lower()
                if "audio" in html_lower and "play" in html_lower:
                    self.handlers.handle_audio(page)
                if "delayed" in html_lower and "remaining" in html_lower:
                    time.sleep(2.0)
                has_canvas = page.evaluate("() => !!document.querySelector('canvas')")
                if has_canvas:
                    self.handlers.handle_canvas_draw(page)
                # Re-extract
                html = page.content()
                new_codes = extract_hidden_codes(html)
                for code in new_codes:
                    if code in failed_codes:
                        continue
                    ok, submit_is_trap = fill_and_submit(page, code, step_number, submit_is_trap)
                    if ok:
                        result.success = True
                        result.code_found = code
                        break
                    failed_codes.append(code)
                if result.success:
                    break

            # 9. Hide stuck modals at attempt 5
            if attempt == 5:
                from src.solver.challenge_handlers import _hide_stuck_modals
                hidden = _hide_stuck_modals(page)
                if hidden > 0:
                    hr = self.handlers.handle_radio_brute_force(page, step_number)
                    if hr.success:
                        result.success = True
                        break

            time.sleep(0.1)

        result.elapsed_seconds = time.time() - t0
        result.attempts = attempt + 1 if 'attempt' in dir() else 0
        if result.success and not result.challenge_type or result.challenge_type == ChallengeType.UNKNOWN:
            # Try to set from detection
            detections = self.detector.detect(page)
            if detections and detections[0].challenge_type != ChallengeType.UNKNOWN:
                result.challenge_type = detections[0].challenge_type
        return result


def _try_animated_button_submit_with_check(page, code: str, step: int) -> bool:
    """Convenience wrapper: try animated button and check progress."""
    from src.solver.challenge_handlers import _try_animated_button_submit
    if _try_animated_button_submit(page, code, step):
        return True
    return check_progress(page.url, step)
