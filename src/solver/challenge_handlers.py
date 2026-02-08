"""21 challenge handlers + shared utilities for the deterministic solver.

All handlers use sync Playwright — ``page`` is always a sync Page object
passed as parameter (never ``self.browser.page``).

Porting convention from ``agent_solver_to_port.py``:
  * ``async def`` → ``def``
  * ``await page.evaluate(...)`` → ``page.evaluate(...)``
  * ``asyncio.sleep(n)`` → ``time.sleep(n)``
  * ``self.browser.page`` → ``page`` parameter
  * ``self.browser.intercepted_codes`` dropped (no network interception in sync)
"""

from __future__ import annotations

import base64
import logging
import math
import re
import time
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Comment

from src.solver.challenge_detector import ChallengeType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CODE_PATTERN = re.compile(r"\b([A-Z0-9]{6})\b")

FALSE_POSITIVES: set[str] = {
    "DEVICE", "VIEWPORT", "SCRIPT", "BUTTON", "SUBMIT", "ACCEPT",
    "COOKIE", "SCROLL", "HIDDEN", "STYLES", "WINDOW", "SCREEN",
    "CHROME", "WEBKIT", "SAFARI", "MOBILE", "TABLET", "ROBOTS",
    "NOINDEX", "FOLLOW", "NOFOLL", "WIDTHD", "HEIGHT", "MARGIN",
    "FILLER", "MOVING", "LOADED", "REVEAL", "CHOICE", "BEATAE",
    "TEMPOR", "FUGIAT", "ALIQUA", "OPTION", "DIALOG", "ANSWER",
    "SELECT", "DOLORE", "MOLLIT", "VENIAM", "CILLUM", "PLEASE",
    "LABORE", "CONTENT", "SECTION", "HEADER", "FOOTER", "BORDER",
    "COLORS", "IMAGES", "CANCEL", "RETURN", "CHANGE",
    "UPDATE", "DELETE", "CREATE", "SEARCH", "FILTER", "NOTICE",
    "ALERTS", "ERRORS", "STATUS", "RESULT", "OUTPUT", "INPUTS",
    "1500MS", "2500MS", "3500MS", "500PX0", "BEFORE", "AFTER0",
    "APPEAR", "STICKY", "NORMAL", "INLINE", "CENTER", "BOTTOM",
    "SHADOW", "CURSOR", "ZINDEX", "EASING", "ROTATE", "SMOOTH",
    "LAYOUT", "RENDER", "EFFECT", "TOGGLE", "HANDLE", "CUSTOM",
    "PIXELS", "POINTS", "WEIGHT", "SOURCE", "TARGET", "ORIGIN",
    "OBJECT", "STRING", "NUMBER", "PROMPT", "ACCESS", "GLOBAL",
    "EXPORT", "IMPORT", "MODULE", "SHOULD", "UNSAFE", "STRICT",
    "SIGNAL", "STREAM", "BUFFER", "PARSED", "THROWS", "FIELDS",
    "CHOOSE", "LABELS", "CLOSER", "TRICKS",
    "FAKING", "PRIZES", "MODALS", "RADIOS", "DECOYS", "PROCED",
    "FILLED", "PIECES", "SIGNUP", "BLOCKS", "CHARTS", "THINGS",
    "SAMPLE", "VERIFY", "PARAMS", "EVENTS", "CHECKS", "CODING",
    "SINGLE", "DOUBLE", "EXPAND", "UNIQUE", "RECENT", "ACTIVE",
    "RANDOM", "CLOSED", "OPENED", "MARKED", "CALLED", "PASSED",
    "FAILED", "PAUSED", "LISTED", "VALUED", "STORED", "POSTED",
    "COVERS", "TIMERS", "COUNTS", "YELLOW", "SECCND", "BLACKS",
    "WHITES", "GREENS", "SPACES", "SECOND", "MINUTE", "STARTS",
    "MEMORY", "BLOCKS", "REMAIN", "SIMPLE", "NEEDED",
    "EXTEND", "INFORM", "PICKED", "CHOSEN",
    "CANVAS", "STROKE", "DRAWIN", "DRAWAN", "LISTEN", "COMPLT",
    "TIMING", "FRAMES", "CAPTUR", "PUZZLE", "SCROLL",
    "MULTIT", "TABBED", "DECODE", "BASE64",
    "PLAYED", "ESCAPE", "ALMOST", "INSIDE", "SEQUEN", "PROGRE",
    "CLICKM", "SQUARE", "CIRCLE",
    "GESTUR", "SOLVED", "PAGEGO", "MEPICK", "ONETHE",
    "CACHED", "SERVIC", "LAYERS", "LEVELS", "NESTED", "SERVER",
    "SOCKET", "CONNEC", "HEREGO", "IFRAME", "BWRONG",
    "ONNEXT", "MUTATI", "DEEPER", "STEPGO", "1WRONG", "2WRONG", "3WRONG",
    "4WRONG", "5WRONG", "6WRONG", "7WRONG", "8WRONG", "9WRONG", "AWRONG",
    "BWRONG", "CWRONG", "DWRONG", "EWRONG", "FWRONG", "GWRONG", "HWRONG",
    "METHIS", "CLICKM", "WORKER", "REGIST",
}

_LATIN_FP: set[str] = {
    "BEATAE", "LABORE", "DOLORE", "VENIAM", "NOSTRU", "ALIQUA", "EXERCI",
    "TEMPOR", "INCIDI", "LABORI", "MAGNAM", "VOLUPT", "SAPIEN", "FUGIAT",
    "COMMOD", "EXCEPT", "OFFICI", "MOLLIT", "PROIDE", "REPUDI",
}

def is_false_positive(code: str) -> bool:
    """Check if a code is a false positive (static list + dynamic patterns)."""
    if code in FALSE_POSITIVES or code in _LATIN_FP:
        return True
    # Any code ending with "WRONG" is a false positive
    if code.endswith("WRONG"):
        return True
    return False

TRAP_WORDS: list[str] = [
    "proceed", "continue", "next step", "next page", "next section",
    "move on", "go forward", "keep going", "advance", "continue reading",
    "continue journey", "click here", "proceed forward",
]


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class HandlerResult:
    codes_found: list[str] = field(default_factory=list)
    actions_log: list[str] = field(default_factory=list)
    needs_extraction: bool = False
    success: bool = False


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def sort_codes_by_priority(codes) -> list[str]:
    """Sort codes: mixed alpha+digit first (most likely real)."""
    def priority(c):
        has_d = any(ch.isdigit() for ch in c)
        has_a = any(ch.isalpha() for ch in c)
        return (not (has_d and has_a), c)
    return sorted(set(codes), key=priority)


def check_progress(url: str, step: int) -> bool:
    """Check if URL indicates we've moved past *step*."""
    url_lower = url.lower()
    if f"step{step + 1}" in url_lower or f"step-{step + 1}" in url_lower or f"step/{step + 1}" in url_lower:
        return True
    if step == 30 and ("complete" in url_lower or "finish" in url_lower or "done" in url_lower):
        return True
    m = re.search(r"step[/-]?(\d+)", url_lower)
    if m and int(m.group(1)) > step:
        return True
    return False


def clear_popups(page) -> int:
    """Clear blocking popups using deterministic JS. Returns count cleared."""
    return page.evaluate("""\
(() => {
    let cleared = 0;
    const hide = (el) => {
        el.style.display = 'none';
        el.style.pointerEvents = 'none';
        el.style.visibility = 'hidden';
        el.style.zIndex = '-1';
    };
    document.querySelectorAll('.fixed, [class*="absolute"], [class*="z-"]').forEach(el => {
        const text = el.textContent || '';
        if (text.includes('fake') && text.includes('real one')) {
            el.querySelectorAll('button').forEach(btn => {
                const bt = (btn.textContent || '').trim();
                if (!bt.toLowerCase().includes('fake') && bt.length > 0 && bt.length < 30) {
                    btn.click(); cleared++;
                }
            });
        }
        if (text.includes('another way to close') ||
            (text.includes('close button') && text.includes('fake') && !text.includes('real one')) ||
            text.includes('won a prize') || text.includes('amazing deals')) {
            hide(el); cleared++;
        }
        if (text.includes('That close button is fake')) { hide(el); cleared++; }
        if (text.includes('Cookie') || text.includes('cookie')) {
            const btn = [...el.querySelectorAll('button')].find(b => b.textContent.includes('Accept'));
            if (btn) { btn.click(); cleared++; }
        }
        if (text.includes('Limited time offer') || text.includes('Click X to close') ||
            text.includes('popup message')) {
            el.querySelectorAll('button').forEach(btn => btn.click());
            hide(el); cleared++;
        }
        if (text.includes('Click the button to dismiss') || text.includes('interact with this modal')) {
            const btn = el.querySelector('button');
            if (btn) { btn.click(); cleared++; }
        }
        if (text.includes('Wrong Button') || text.includes('Try Again')) {
            const btn = el.querySelector('button');
            if (btn) { btn.click(); cleared++; }
        }
    });
    document.querySelectorAll('.fixed').forEach(el => {
        if (el.classList.contains('bg-black/70') ||
            (el.style.backgroundColor || '').includes('rgba(0, 0, 0')) {
            if (!el.textContent.includes('Step') && !el.querySelector('input[type="radio"]')) {
                el.style.pointerEvents = 'none'; cleared++;
            }
        }
    });
    return cleared;
})()""")


def fill_and_submit(page, code: str, step: int, submit_is_trap: bool = False) -> tuple[bool, bool]:
    """Fill *code* into input, click submit, check URL change.

    Returns ``(success, submit_is_trap)`` where *submit_is_trap* is ``True``
    when a "Wrong Button!" popup was detected (meaning the submit button
    itself is a trap, not that the code is wrong).
    """
    url_before = page.url
    try:
        if submit_is_trap:
            ok = _try_animated_button_submit(page, code, step)
            if ok:
                return True, True
            # Enter key fallback
            page.evaluate(f"""\
(() => {{
    const inp = document.querySelector('input[placeholder*="code" i], input[type="text"]');
    if (inp) {{
        const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
        s.call(inp, '{code}');
        inp.dispatchEvent(new Event('input', {{bubbles: true}}));
        inp.dispatchEvent(new Event('change', {{bubbles: true}}));
        inp.focus();
    }}
}})()""")
            page.keyboard.press("Enter")
            time.sleep(0.3)
            if page.url != url_before:
                logger.info("Code '%s' WORKED (Enter bypass)!", code)
                return True, True
            return False, True

        # Scroll input into view
        page.evaluate("""\
(() => {
    const input = document.querySelector('input[placeholder*="code" i], input[type="text"]');
    if (input) input.scrollIntoView({behavior: 'instant', block: 'center'});
})()""")
        time.sleep(0.1)

        # Clear and type
        inp = page.locator('input[placeholder*="code" i], input[type="text"]').first
        try:
            inp.click(click_count=3, timeout=1000)
        except Exception:
            page.evaluate("""\
(() => {
    const input = document.querySelector('input[placeholder*="code" i], input[type="text"]');
    if (input) { input.focus(); input.select(); }
})()""")
        page.keyboard.press("Backspace")
        time.sleep(0.05)
        page.keyboard.type(code, delay=20)
        time.sleep(0.15)

        # Click submit (avoid trap buttons)
        clicked = page.evaluate("""\
(() => {
    const TRAPS = ['proceed', 'continue', 'next step', 'next page', 'next section',
        'move on', 'go forward', 'keep going', 'advance', 'continue reading',
        'continue journey', 'click here', 'proceed forward'];
    const isTrap = (t) => TRAPS.some(w => t.toLowerCase().includes(w));
    const input = document.querySelector('input[placeholder*="code" i], input[type="text"]');
    if (!input) return false;
    let container = input.parentElement;
    for (let i = 0; i < 4 && container; i++) {
        const btns = container.querySelectorAll('button');
        for (const btn of btns) {
            const t = (btn.textContent || '').trim();
            if (!btn.disabled && !isTrap(t) &&
                (btn.type === 'submit' || t.includes('Submit') || t.includes('Go') || t === '\u2192' || t.length <= 2)) {
                btn.scrollIntoView({behavior: 'instant', block: 'center'});
                btn.click(); return true;
            }
        }
        const safe = [...btns].filter(b => !b.disabled && !isTrap((b.textContent || '').trim()));
        if (safe.length === 1) { safe[0].click(); return true; }
        container = container.parentElement;
    }
    for (const b of document.querySelectorAll('button')) {
        const t = (b.textContent || '').trim();
        if ((t === 'Submit' || t === 'Submit Code') && !b.disabled) { b.click(); return true; }
    }
    return false;
})()""")

        if not clicked:
            page.keyboard.press("Enter")

        time.sleep(0.4)

        # Check for "Wrong Button!" popup
        wrong_button = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    return text.includes('Wrong Button') || text.includes('wrong button');
})()""")
        if wrong_button:
            logger.info("'Wrong Button!' detected — submit button is trap")
            clear_popups(page)
            ok = _try_animated_button_submit(page, code, step)
            if ok:
                return True, True
            # Enter key fallback
            page.evaluate("""\
(() => {
    const inp = document.querySelector('input[placeholder*="code" i], input[type="text"]');
    if (inp) inp.focus();
})()""")
            page.keyboard.press("Enter")
            time.sleep(0.3)
            if page.url != url_before:
                logger.info("Code '%s' WORKED (Enter key)!", code)
                return True, True
            return False, True

        if page.url != url_before:
            logger.info("Code '%s' WORKED!", code)
            return True, submit_is_trap
        else:
            logger.debug("Code '%s' failed", code)
            return False, submit_is_trap
    except Exception as e:
        logger.warning("Fill error: %s", e)
        return False, submit_is_trap


def extract_hidden_codes(html: str) -> list[str]:
    """Extract potential 6-character codes from HTML."""
    codes: set[str] = set()
    soup = BeautifulSoup(html, "html.parser")

    all_text = soup.get_text(separator=" ")
    codes.update(CODE_PATTERN.findall(all_text.upper()))

    for elem in soup.find_all(True):
        for key, value in elem.attrs.items():
            if key.startswith("data-") and isinstance(value, str):
                codes.update(CODE_PATTERN.findall(value.upper()))

    for elem in soup.find_all(True):
        for key, value in elem.attrs.items():
            if key.startswith("aria-") and isinstance(value, str):
                codes.update(CODE_PATTERN.findall(value.upper()))

    for elem in soup.find_all(style=re.compile(r"display:\s*none|visibility:\s*hidden")):
        codes.update(CODE_PATTERN.findall(elem.get_text().upper()))

    for elem in soup.find_all(attrs={"hidden": True}):
        codes.update(CODE_PATTERN.findall(elem.get_text().upper()))

    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        codes.update(CODE_PATTERN.findall(str(comment).upper()))

    for match in re.compile(r"<!--(.*?)-->", re.DOTALL).findall(html):
        codes.update(CODE_PATTERN.findall(match.upper()))

    for meta in soup.find_all("meta"):
        content = meta.get("content", "")
        if isinstance(content, str):
            codes.update(CODE_PATTERN.findall(content.upper()))

    for elem in soup.find_all(attrs={"title": True}):
        title = elem.get("title", "")
        if isinstance(title, str):
            codes.update(CODE_PATTERN.findall(title.upper()))

    b64_pattern = re.compile(r"[A-Za-z0-9+/]{8,}={0,2}")
    raw_text = soup.get_text(separator=" ") + " " + html
    for b64_match in b64_pattern.findall(raw_text):
        try:
            decoded = base64.b64decode(b64_match).decode("utf-8", errors="ignore")
            if decoded:
                codes.update(CODE_PATTERN.findall(decoded.upper()))
        except Exception:
            pass
    for elem in soup.find_all(True):
        for key, value in elem.attrs.items():
            if key.startswith("data-") and isinstance(value, str) and len(value) >= 8:
                try:
                    decoded = base64.b64decode(value).decode("utf-8", errors="ignore")
                    if decoded:
                        codes.update(CODE_PATTERN.findall(decoded.upper()))
                except Exception:
                    pass

    codes = {c for c in codes if not is_false_positive(c)}
    codes = {c for c in codes if not re.match(r"^\d+(?:PX|VH|VW|EM|REM|CH|EX|PC|PT|MM|CM|IN|MS|FR)$", c)}
    codes = {c for c in codes if not c.isdigit()}

    return sort_codes_by_priority(codes)


def deep_code_extraction(page, known_codes: set[str] | None = None) -> list[str]:
    """Extract codes from React fiber, CSS pseudo, shadow DOM, JS vars, iframes."""
    known = set(known_codes or [])
    all_codes: set[str] = set()

    # 1. React Fiber
    react_codes = page.evaluate("""\
(() => {
    const codes = new Set();
    const CODE_RE = /^[A-Z0-9]{6}$/;
    const EMBEDDED_RE = /\\b[A-Z0-9]{6}\\b/g;
    const visited = new WeakSet();
    function extract(val, depth) {
        if (depth > 8 || !val) return;
        const t = typeof val;
        if (t === 'string') {
            if (CODE_RE.test(val)) codes.add(val);
            else if (val.length < 500) { const m = val.match(EMBEDDED_RE); if (m) m.forEach(c => codes.add(c)); }
        } else if (t === 'number') { return;
        } else if (Array.isArray(val)) {
            for (let i = 0; i < Math.min(val.length, 50); i++) extract(val[i], depth + 1);
        } else if (t === 'object') {
            if (visited.has(val)) return; visited.add(val);
            try { const keys = Object.keys(val); for (let i = 0; i < Math.min(keys.length, 100); i++) extract(val[keys[i]], depth + 1); } catch(e) {}
        }
    }
    function walkFiber(fiber, depth) {
        if (!fiber || depth > 60 || visited.has(fiber)) return; visited.add(fiber);
        let hook = fiber.memoizedState;
        for (let i = 0; hook && i < 40; i++, hook = hook.next) {
            extract(hook.memoizedState, 0);
            if (hook.queue) extract(hook.queue.lastRenderedState, 0);
            if (hook.baseState !== undefined) extract(hook.baseState, 0);
        }
        extract(fiber.memoizedProps, 0); extract(fiber.pendingProps, 0);
        if (fiber.stateNode && typeof fiber.stateNode === 'object' && !(fiber.stateNode instanceof HTMLElement)) {
            try { extract(fiber.stateNode.state, 0); } catch(e) {}
            try { extract(fiber.stateNode.props, 0); } catch(e) {}
        }
        walkFiber(fiber.child, depth + 1); walkFiber(fiber.sibling, depth + 1);
    }
    document.querySelectorAll('*').forEach(el => {
        for (const key of Object.keys(el)) {
            if (key.startsWith('__reactFiber$') || key.startsWith('__reactInternalInstance$')) { walkFiber(el[key], 0); break; }
        }
    });
    return [...codes];
})()""")
    all_codes.update(react_codes or [])

    # 2. CSS pseudo-elements & custom properties
    css_codes = page.evaluate("""\
(() => {
    const codes = []; const RE = /[A-Z0-9]{6}/g;
    const els = document.querySelectorAll('*');
    for (let i = 0; i < els.length; i++) {
        for (const pseudo of ['::before', '::after']) {
            try { const c = window.getComputedStyle(els[i], pseudo).content;
                if (c && c !== 'none' && c !== 'normal' && c !== '""') { const m = c.toUpperCase().match(RE); if (m) codes.push(...m); }
            } catch(e) {}
        }
    }
    try { const rs = getComputedStyle(document.documentElement);
        for (let i = 0; i < rs.length; i++) { if (rs[i].startsWith('--')) { const v = rs.getPropertyValue(rs[i]); const m = v.toUpperCase().match(RE); if (m) codes.push(...m); } }
    } catch(e) {}
    return [...new Set(codes)];
})()""")
    all_codes.update(css_codes or [])

    # 3. Shadow DOM
    shadow_codes = page.evaluate("""\
(() => {
    const codes = []; const RE = /\\b[A-Z0-9]{6}\\b/g;
    function walk(root) { if (!root || !root.shadowRoot) return;
        const m = ((root.shadowRoot.textContent||'') + ' ' + (root.shadowRoot.innerHTML||'')).toUpperCase().match(RE);
        if (m) codes.push(...m); root.shadowRoot.querySelectorAll('*').forEach(walk);
    }
    document.querySelectorAll('*').forEach(walk); return [...new Set(codes)];
})()""")
    all_codes.update(shadow_codes or [])

    # 4. JS global vars
    js_codes = page.evaluate("""\
(() => {
    const codes = []; const RE = /\\b[A-Z0-9]{6}\\b/g;
    const candidates = ['__NEXT_DATA__','__APP_DATA__','__CHALLENGE__','__code','__CODE','challengeCode','currentCode','navCode','secretCode','hiddenCode'];
    for (const g of candidates) { try { const v = window[g]; if (!v) continue;
        const s = typeof v === 'string' ? v : JSON.stringify(v).substring(0,10000); const m = s.toUpperCase().match(RE); if (m) codes.push(...m);
    } catch(e) {} }
    for (const k of Object.getOwnPropertyNames(window)) { try { const v = window[k]; if (typeof v === 'string' && /^[A-Z0-9]{6}$/.test(v)) codes.push(v); } catch(e) {} }
    return [...new Set(codes)];
})()""")
    all_codes.update(js_codes or [])

    # 5. Iframe content
    iframe_codes = page.evaluate("""\
(() => {
    const codes = []; const RE = /\\b[A-Z0-9]{6}\\b/g;
    document.querySelectorAll('iframe').forEach(iframe => {
        try { const doc = iframe.contentDocument || iframe.contentWindow.document;
            const m = (doc.body ? doc.body.textContent : '').toUpperCase().match(RE); if (m) codes.push(...m);
        } catch(e) {} });
    return [...new Set(codes)];
})()""")
    all_codes.update(iframe_codes or [])

    # Filter
    all_codes = {c for c in all_codes if not is_false_positive(c)}
    all_codes -= known
    all_codes = {c for c in all_codes if not c.isdigit()}
    all_codes = {c for c in all_codes if not re.match(r"^\d+(?:PX|VH|VW|EM|REM|MS|FR)$", c)}
    return sort_codes_by_priority(all_codes)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_animated_button_submit(page, code: str, step: int) -> bool:
    """Freeze animated elements and click them one-by-one as alternate submit.

    Each animated element is frozen at screen centre, clicked, and then hidden
    if the click did not advance the step — so the *next* element underneath
    becomes the click target.  This avoids the previous bug where all elements
    were stacked at the same z-index and only the topmost one was ever hit.
    """
    try:
        # Restore any elements hidden by a previous call and clear old tags.
        page.evaluate("""\
(() => {
    document.querySelectorAll('[data-anim-btn-idx]').forEach(el => {
        el.style.display = '';
        el.style.pointerEvents = '';
        el.removeAttribute('data-anim-btn-idx');
    });
})()""")

        # Tag each animated element with a unique ID so we can target them
        # individually from Python.
        count = page.evaluate("""\
(() => {
    let idx = 0;
    document.querySelectorAll('*').forEach(el => {
        const style = getComputedStyle(el);
        const cls = el.getAttribute('class') || '';
        const hasAnimation = (style.animation && style.animation !== 'none' &&
            style.animationName !== 'none') ||
            cls.includes('animate-[move') || cls.includes('animate-[bounce');
        const isSmall = el.offsetWidth > 10 && el.offsetWidth < 200 &&
            el.offsetHeight > 10 && el.offsetHeight < 200;
        const isClickable = style.cursor === 'pointer' || cls.includes('cursor-pointer');
        if (hasAnimation && isSmall && isClickable && el.offsetParent) {
            el.setAttribute('data-anim-btn-idx', String(idx++));
        }
    });
    return idx;
})()""")
        if not count:
            return False

        # Fill code once up-front.
        _fill_code_in_input(page, code)
        time.sleep(0.1)

        for i in range(count):
            clear_popups(page)
            # Freeze this specific element at screen centre and click it.
            info = page.evaluate(f"""\
(() => {{
    const el = document.querySelector('[data-anim-btn-idx="{i}"]');
    if (!el || el.style.display === 'none') return null;
    el.style.animation = 'none';
    el.style.position = 'fixed';
    el.style.top = '50%';
    el.style.left = '50%';
    el.style.zIndex = '99999';
    el.style.transform = 'translate(-50%, -50%)';
    const r = el.getBoundingClientRect();
    return {{x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2),
        text: (el.textContent || '').trim().substring(0, 40)}};
}})()""")
            if not info:
                continue
            try:
                page.mouse.click(info["x"], info["y"])
            except Exception:
                pass
            time.sleep(0.3)
            if check_progress(page.url, step):
                logger.info("Animated button '%s' with code '%s' WORKED!", info["text"], code)
                return True
            # Hide this element so the next one is exposed for clicking.
            page.evaluate(f"""\
(() => {{
    const el = document.querySelector('[data-anim-btn-idx="{i}"]');
    if (el) {{ el.style.display = 'none'; el.style.pointerEvents = 'none'; }}
}})()""")

        return False
    except Exception as e:
        logger.warning("Animated button error: %s", e)
        return False


def _try_trap_buttons(page, code: str, step: int) -> bool:
    """Try clicking trap-labeled buttons as real submits (some are actually real).

    When the normal submit shows "Wrong Button!", the trap-labeled buttons
    (proceed, continue, etc.) might actually be the real submit mechanism.
    """
    try:
        count = page.evaluate("""\
(() => {
    const TRAPS = ['proceed', 'continue', 'next step', 'next page', 'next section',
        'move on', 'go forward', 'keep going', 'advance', 'continue journey',
        'click here', 'proceed forward', 'continue reading', 'next', 'go', 'submit code', 'submit'];
    return [...document.querySelectorAll('button, a')].filter(el => {
        const t = (el.textContent || '').trim().toLowerCase();
        return t.length < 40 && TRAPS.some(w => t === w || t.includes(w));
    }).length;
})()""")
        if not count:
            return False

        max_btns = 8 if count > 15 else min(count, 15)
        logger.info("Trying code '%s' with %d/%d trap buttons...", code, max_btns, count)

        for i in range(max_btns):
            clear_popups(page)
            _fill_code_in_input(page, code)
            time.sleep(0.05)

            page.evaluate(f"""\
(() => {{
    const TRAPS = ['proceed', 'continue', 'next step', 'next page', 'next section',
        'move on', 'go forward', 'keep going', 'advance', 'continue journey',
        'click here', 'proceed forward', 'continue reading', 'next', 'go', 'submit code', 'submit'];
    const btns = [...document.querySelectorAll('button, a')].filter(el => {{
        const t = (el.textContent || '').trim().toLowerCase();
        return t.length < 40 && TRAPS.some(w => t === w || t.includes(w));
    }});
    const btn = btns[{i}];
    if (btn) {{
        btn.scrollIntoView({{behavior: 'instant', block: 'center'}});
        btn.click();
    }}
}})()""")
            time.sleep(0.15)
            if check_progress(page.url, step):
                logger.info("Trap button %d with code '%s' WORKED!", i, code)
                return True

        return False
    except Exception as e:
        logger.warning("Trap button error: %s", e)
        return False


def _hide_stuck_modals(page) -> int:
    """Hide radio/option modals blocking the page."""
    return page.evaluate("""\
(() => {
    let hidden = 0;
    document.querySelectorAll('.fixed, [role="dialog"], [class*="modal"]').forEach(el => {
        const text = el.textContent || '';
        if ((text.includes('Please Select') || text.includes('Submit & Continue') ||
             text.includes('Submit and Continue') || text.includes('Select an Option') ||
             text.includes('RADIO MODAL') || text.includes('radio button') ||
             (el.querySelector('input[type="radio"]') && text.includes('Submit')) ||
             (el.querySelector('[role="radio"]') && text.includes('Submit'))) &&
            !el.querySelector('input[type="text"]')) {
            el.style.display = 'none';
            el.style.visibility = 'hidden';
            el.style.pointerEvents = 'none';
            hidden++;
        }
    });
    document.querySelectorAll('.fixed.inset-0').forEach(el => {
        if (el.querySelector('input[type="radio"]') || el.querySelector('[role="radio"]')) {
            el.style.display = 'none'; hidden++;
        }
    });
    return hidden;
})()""")


def force_reset_puzzle(page, failed_codes: list[str]) -> str | None:
    """Force-reset a math puzzle showing cached 'already solved' state.

    Resets React fiber state to re-trigger code generation.
    Ported from reference ``_force_reset_puzzle()``.
    """
    try:
        # Step 1: Find the math answer from the page text
        expr = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    const m = text.match(/(\\d+)\\s*([+\\-*\u00d7\u00f7\\/])\\s*(\\d+)\\s*=\\s*\\?/);
    if (!m) return null;
    const a = parseInt(m[1]), op = m[2], b = parseInt(m[3]);
    switch(op) {
        case '+': return String(a + b);
        case '-': return String(a - b);
        case '*': case '\u00d7': return String(a * b);
        case '/': case '\u00f7': return String(Math.floor(a / b));
        default: return String(a + b);
    }
})()""")
        if not expr:
            return None

        logger.info("Force-reset puzzle: answer=%s", expr)

        # Step 2: Record codes before reset
        codes_before = set(page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    return [...new Set((text.match(/\\b[A-Z0-9]{6}\\b/g) || []))];
})()"""))

        # Step 3: Find puzzle container and reset React state via fiber
        page.evaluate("""\
(() => {
    const visited = new WeakSet();
    const puzzleEls = [...document.querySelectorAll('div')].filter(el => {
        const t = el.textContent || '';
        return t.includes('Puzzle') && (t.includes('solved') || t.includes('Code revealed'))
            && el.offsetParent && el.offsetWidth > 100;
    });
    puzzleEls.sort((a, b) => a.textContent.length - b.textContent.length);
    for (const el of puzzleEls.slice(0, 3)) {
        const fiberKey = Object.keys(el).find(k =>
            k.startsWith('__reactFiber$') || k.startsWith('__reactInternalInstance$'));
        if (!fiberKey) continue;
        let fiber = el[fiberKey];
        let attempts = 0;
        while (fiber && attempts < 30) {
            if (fiber.memoizedState) {
                let state = fiber.memoizedState;
                let stateIdx = 0;
                while (state && stateIdx < 20) {
                    const val = state.memoizedState;
                    if (val === true && state.queue) {
                        const dispatch = state.queue.dispatch;
                        if (dispatch) { try { dispatch(false); } catch(e) {} }
                    }
                    if (typeof val === 'string' && /^[A-Z0-9]{6}$/.test(val) && state.queue) {
                        const dispatch = state.queue.dispatch;
                        if (dispatch) { try { dispatch(''); } catch(e) {} }
                    }
                    if (typeof val === 'number' && val > 0 && val < 100 && state.queue) {
                        const dispatch = state.queue.dispatch;
                        if (dispatch) { try { dispatch(0); } catch(e) {} }
                    }
                    state = state.next;
                    stateIdx++;
                }
            }
            fiber = fiber.return;
            attempts++;
        }
    }
    return true;
})()""")

        time.sleep(0.5)

        # Step 4: Check if input appeared after reset
        has_input = page.evaluate("""\
(() => {
    const inp = document.querySelector('input[type="number"], input[inputmode="numeric"], ' +
        'input[placeholder*="answer" i], input[placeholder*="solution" i]');
    return !!(inp && inp.offsetParent);
})()""")

        if has_input:
            logger.info("Force-reset: puzzle input appeared, entering %s...", expr)
            page.evaluate(f"""\
(() => {{
    const sels = ['input[type="number"]', 'input[inputmode="numeric"]',
        'input[placeholder*="answer" i]', 'input[placeholder*="solution" i]'];
    for (const sel of sels) {{
        const inp = document.querySelector(sel);
        if (inp && inp.offsetParent) {{
            const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
            s.call(inp, '{expr}');
            inp.dispatchEvent(new Event('input', {{bubbles: true}}));
            inp.dispatchEvent(new Event('change', {{bubbles: true}}));
            inp.focus();
            break;
        }}
    }}
}})()""")
            time.sleep(0.2)
            page.keyboard.press("Enter")
            time.sleep(0.5)

            # Click Solve button
            page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        const t = (btn.textContent || '').trim().toLowerCase();
        if ((t === 'solve' || t.includes('check') || t.includes('verify')) && !btn.disabled) {
            btn.click(); break;
        }
    }
})()""")
            time.sleep(0.8)

        # Step 5: Check for new codes
        codes_after = set(page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    return [...new Set((text.match(/\\b[A-Z0-9]{6}\\b/g) || []))];
})()"""))
        new_codes = codes_after - codes_before
        fresh = [c for c in new_codes if not is_false_positive(c)]
        if fresh:
            sorted_fresh = sort_codes_by_priority(fresh)
            logger.info("Force-reset puzzle: fresh code: %s (all: %s)", sorted_fresh[0], sorted_fresh)
            return sorted_fresh[0]

        logger.info("Force-reset puzzle: no new codes generated")
        return None
    except Exception as e:
        logger.warning("Force-reset puzzle error: %s", e)
        return None


def _fill_code_in_input(page, code: str) -> None:
    """Set *code* in the code input via React-compatible setter."""
    page.evaluate(f"""\
(() => {{
    const inp = document.querySelector('input[placeholder*="code" i], input[type="text"]');
    if (inp) {{
        const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
        s.call(inp, '{code}');
        inp.dispatchEvent(new Event('input', {{bubbles: true}}));
        inp.dispatchEvent(new Event('change', {{bubbles: true}}));
    }}
}})()""")


# ---------------------------------------------------------------------------
# ChallengeHandlers class
# ---------------------------------------------------------------------------

class ChallengeHandlers:
    """Dispatch to the right handler method."""

    def handle(
        self,
        page,
        challenge_type: ChallengeType,
        step: int,
        failed_codes: list[str],
        stale_codes: list[str] | None = None,
    ) -> HandlerResult:
        dispatch = {
            ChallengeType.MATH_PUZZLE: lambda: self.handle_math_puzzle(page, failed_codes, stale_codes),
            ChallengeType.SHADOW_DOM: lambda: self.handle_shadow_dom(page),
            ChallengeType.CANVAS_DRAW: lambda: self.handle_canvas_draw(page),
            ChallengeType.AUDIO_CHALLENGE: lambda: self.handle_audio(page),
            ChallengeType.WEBSOCKET: lambda: self.handle_websocket(page),
            ChallengeType.SERVICE_WORKER: lambda: self.handle_service_worker(page),
            ChallengeType.IFRAME_RECURSIVE: lambda: self.handle_iframe(page),
            ChallengeType.MUTATION_OBSERVER: lambda: self.handle_mutation(page),
            ChallengeType.SCROLL_TO_FIND: lambda: self.handle_scroll_to_find(page, failed_codes, step),
            ChallengeType.HOVER_REVEAL: lambda: self.handle_hover_reveal(page),
            ChallengeType.DELAYED_REVEAL: lambda: self.handle_delayed_reveal(page),
            ChallengeType.DRAG_AND_DROP: lambda: self.handle_drag_and_drop(page),
            ChallengeType.KEYBOARD_SEQUENCE: lambda: self.handle_keyboard_sequence(page),
            ChallengeType.RADIO_BRUTE_FORCE: lambda: self.handle_radio_brute_force(page, step),
            ChallengeType.TIMING_CAPTURE: lambda: self.handle_timing_capture(page),
            ChallengeType.SPLIT_PARTS: lambda: self.handle_split_parts(page),
            ChallengeType.ROTATING_CODE: lambda: self.handle_rotating_code(page),
            ChallengeType.MULTI_TAB: lambda: self.handle_multi_tab(page),
            ChallengeType.SEQUENCE_CHALLENGE: lambda: self.handle_sequence(page),
            ChallengeType.VIDEO_FRAMES: lambda: self.handle_video_frames(page),
            ChallengeType.ANIMATED_BUTTON: lambda: self.handle_animated_button(page, failed_codes, step),
            ChallengeType.DOM_EXTRACTION: lambda: self.handle_dom_extraction(page, failed_codes),
        }
        handler = dispatch.get(challenge_type)
        if handler is None:
            return HandlerResult(actions_log=["no handler for " + challenge_type.name])
        try:
            return handler()
        except Exception as e:
            logger.warning("Handler %s error: %s", challenge_type.name, e)
            return HandlerResult(actions_log=[f"error: {e}"])

    # ------------------------------------------------------------------
    # Per-type handlers
    # ------------------------------------------------------------------

    def handle_math_puzzle(self, page, failed_codes: list[str],
                           stale_codes: list[str] | None = None) -> HandlerResult:
        known_bad = set(failed_codes)
        stale_set = set(stale_codes or [])
        codes_before = set(page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    return [...new Set((text.match(/\\b[A-Z0-9]{6}\\b/g) || []))];
})()"""))

        expr = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    const m = text.match(/(\\d+)\\s*([+\\-*\u00d7\u00f7\\/])\\s*(\\d+)\\s*=\\s*\\?/);
    if (!m) return null;
    const a = parseInt(m[1]), op = m[2], b = parseInt(m[3]);
    let answer;
    switch(op) {
        case '+': answer = a + b; break;
        case '-': answer = a - b; break;
        case '*': case '\u00d7': answer = a * b; break;
        case '/': case '\u00f7': answer = Math.floor(a / b); break;
        default: answer = a + b;
    }
    return String(answer);
})()""")
        if not expr:
            return HandlerResult(actions_log=["no math expression found"])

        logger.info("Math puzzle answer: %s", expr)

        # Fill answer
        filled_sel = page.evaluate(f"""\
(() => {{
    const selectors = [
        'input[type="number"]', 'input[inputmode="numeric"]',
        'input[placeholder*="answer" i]', 'input[placeholder*="solution" i]',
        'input[type="text"]:not([placeholder*="code" i])'
    ];
    for (const sel of selectors) {{
        const input = document.querySelector(sel);
        if (input && input.offsetParent) {{
            const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
            s.call(input, '{expr}');
            input.dispatchEvent(new Event('input', {{bubbles: true}}));
            input.dispatchEvent(new Event('change', {{bubbles: true}}));
            input.focus();
            return sel;
        }}
    }}
    return null;
}})()""")
        logger.info("Math: filled input via: %s", filled_sel)
        if not filled_sel:
            # No math input found — can't solve the puzzle.  Don't fall through
            # to pattern matching which would pick up stale/noise codes.
            return HandlerResult(actions_log=["math input not found"], needs_extraction=True)
        time.sleep(0.2)
        page.keyboard.press("Enter")
        time.sleep(0.5)

        # Click Solve button (matches reference: solve, check, verify, submit only)
        for solve_try in range(3):
            solve_clicked = page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        const t = (btn.textContent || '').trim().toLowerCase();
        if ((t === 'solve' || t.includes('check') || t.includes('verify') || t === 'submit')
            && !btn.disabled && btn.offsetParent) {
            btn.click(); return t;
        }
    }
    return null;
})()""")
            if solve_try == 0:
                logger.info("Math: Solve button clicked: %s, codes_before: %s",
                            solve_clicked, sorted(codes_before)[:5])
            time.sleep(0.8)

            codes_after = set(page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    return [...new Set((text.match(/\\b[A-Z0-9]{6}\\b/g) || []))];
})()"""))
            new_codes = codes_after - codes_before
            new_codes = {c for c in new_codes if not is_false_positive(c) and c not in known_bad}
            if solve_try == 0:
                logger.info("Math: codes_after: %s, new_codes: %s",
                            sorted(codes_after)[:5], sorted(new_codes)[:5])
            if new_codes:
                sorted_new = sort_codes_by_priority(new_codes)
                return HandlerResult(codes_found=sorted_new, actions_log=["math solved"], needs_extraction=True)

        # Pattern fallback — separate high-confidence "code revealed" patterns
        # from lower-confidence generic matches. High-confidence codes bypass
        # FALSE_POSITIVES since the page explicitly says "Code revealed: XXX".
        revealed_and_generic = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    const revealed = [];
    const generic = [];
    const revealPatterns = [
        /(?:code(?:\\s+is)?|revealed?)\\s*[:=]\\s*([A-Z0-9]{6})/i,
        /(?:solved|correct|success)[^.]*?\\b([A-Z0-9]{6})\\b/i,
    ];
    for (const p of revealPatterns) { const m = text.match(p); if (m) revealed.push(m[1].toUpperCase()); }
    const genericPatterns = [
        /\\b([A-Z0-9]{6})\\b(?=[^A-Z0-9]*(?:submit|enter|type|input))/i
    ];
    for (const p of genericPatterns) { const m = text.match(p); if (m) generic.push(m[1].toUpperCase()); }
    const successEls = document.querySelectorAll('.text-green-600, .text-green-500, .bg-green-100, .bg-green-50, .text-emerald-600');
    for (const el of successEls) { const t = (el.textContent || '').trim(); const m = t.match(/\\b([A-Z0-9]{6})\\b/); if (m) revealed.push(m[1]); }
    return {revealed: [...new Set(revealed)], generic: [...new Set(generic)]};
})()""")
        revealed_codes = (revealed_and_generic or {}).get("revealed", [])
        generic_codes = (revealed_and_generic or {}).get("generic", [])
        logger.info("Math: pattern fallback revealed=%s generic=%s, known_bad: %s",
                     revealed_codes, generic_codes, sorted(known_bad)[:5])
        # Revealed codes: the page explicitly says "Code revealed: XXX" — trust it.
        # Filter by stale codes (wrong step) but NOT by failed-this-step (trap button).
        fresh_revealed = [c for c in revealed_codes if c not in stale_set and not is_false_positive(c)]
        # Generic codes: filter by both known_bad and FALSE_POSITIVES
        fresh_generic = [c for c in generic_codes if c not in known_bad and not is_false_positive(c)]
        fresh = list(dict.fromkeys(fresh_revealed + fresh_generic))
        if fresh:
            return HandlerResult(codes_found=sort_codes_by_priority(fresh), actions_log=["math pattern"], needs_extraction=True)
        return HandlerResult(actions_log=["math solved, no code found"], needs_extraction=True)

    def handle_shadow_dom(self, page) -> HandlerResult:
        codes: list[str] = []
        log: list[str] = []
        for click_round in range(10):
            result = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    const codeMatch = text.match(/(?:code|Code)[^:]*:\\s*([A-Z0-9]{6})/);
    if (codeMatch) return {done: true, code: codeMatch[1]};
    const progressMatch = text.match(/(\\d+)\\/(\\d+)\\s*(?:levels?|layers?)/i);
    const current = progressMatch ? parseInt(progressMatch[1]) : 0;
    const total = progressMatch ? parseInt(progressMatch[2]) : 3;
    if (current >= total) {
        for (const btn of document.querySelectorAll('button')) {
            const t = (btn.textContent || '').trim().toLowerCase();
            if ((t.includes('reveal') || t.includes('complete') || t.includes('extract')) && !btn.disabled && btn.offsetParent) {
                btn.click(); return {clicked: 'reveal_btn', current, total};
            }
        }
    }
    const candidates = [...document.querySelectorAll('div')].filter(el => {
        const cls = el.getAttribute('class') || '';
        if (!cls.includes('cursor-pointer') && !cls.includes('cursor_pointer')) return false;
        if (!el.offsetParent || el.offsetWidth < 30) return false;
        let directText = '';
        for (const node of el.childNodes) { if (node.nodeType === 3) directText += node.textContent; }
        directText = directText.trim();
        if (!directText) { const first = el.children[0]; if (first && first.tagName !== 'DIV') directText = first.textContent.trim(); }
        return /^(?:Shadow\\s+)?Level\\s+\\d/i.test(directText);
    });
    for (const el of candidates) {
        const cls = el.getAttribute('class') || '';
        const allText = el.textContent || '';
        if (!cls.includes('green') && !allText.startsWith('\u2713') && !allText.includes('\u2713 Shadow') && !allText.includes('\u2713Shadow')) {
            let label = '';
            for (const node of el.childNodes) { if (node.nodeType === 3) label += node.textContent; }
            el.click();
            return {clicked: (label || 'level').trim().substring(0, 30), current, total};
        }
    }
    for (const el of document.querySelectorAll('div[class*="slate"]')) {
        const cls = el.getAttribute('class') || '';
        if (!cls.includes('cursor') && !cls.includes('hover')) continue;
        if (cls.includes('green')) continue;
        if (!el.offsetParent || el.offsetWidth < 30) continue;
        let directLen = 0;
        for (const node of el.childNodes) { if (node.nodeType === 3) directLen += node.textContent.trim().length; }
        if (directLen > 0 && directLen < 50) { el.click(); return {clicked: 'slate-fallback', current, total}; }
    }
    return {clicked: null, current, total};
})()""")
            if not result:
                break
            if result.get("done") and result.get("code"):
                c = result["code"]
                if not is_false_positive(c):
                    codes.append(c)
                    return HandlerResult(codes_found=codes, actions_log=log + ["shadow code found"], success=True)
            if result.get("clicked"):
                log.append(f"shadow: {result['clicked']}")
                time.sleep(0.4)
            else:
                break

        final = page.evaluate("(() => { const m = (document.body.textContent||'').match(/(?:code|Code)[^:]*:\\s*([A-Z0-9]{6})/); return m ? m[1] : null; })()")
        if final and not is_false_positive(final):
            codes.append(final)
        return HandlerResult(codes_found=codes, actions_log=log, needs_extraction=True)

    def handle_canvas_draw(self, page) -> HandlerResult:
        log: list[str] = []
        canvas_info = page.evaluate("""\
(() => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return {found: false};
    canvas.scrollIntoView({behavior: 'instant', block: 'center'});
    const rect = canvas.getBoundingClientRect();
    const text = document.body.textContent.toLowerCase();
    let shape = 'strokes';
    if (text.includes('square')) shape = 'square';
    else if (text.includes('circle')) shape = 'circle';
    else if (text.includes('triangle')) shape = 'triangle';
    else if (text.includes('line')) shape = 'line';
    return {found: true, x: rect.x, y: rect.y, w: rect.width, h: rect.height, shape};
})()""")
        if not canvas_info.get("found"):
            return HandlerResult(actions_log=["no canvas found"])

        cx, cy, cw, ch = canvas_info["x"], canvas_info["y"], canvas_info["w"], canvas_info["h"]
        shape = canvas_info.get("shape", "strokes")
        log.append(f"canvas: drawing {shape}")

        if shape == "square":
            m = 0.2
            corners = [(cx+cw*m, cy+ch*m), (cx+cw*(1-m), cy+ch*m),
                        (cx+cw*(1-m), cy+ch*(1-m)), (cx+cw*m, cy+ch*(1-m)),
                        (cx+cw*m, cy+ch*m)]
            page.mouse.move(corners[0][0], corners[0][1])
            page.mouse.down()
            for corner in corners[1:]:
                page.mouse.move(corner[0], corner[1], steps=15)
                time.sleep(0.05)
            page.mouse.up()
        elif shape == "circle":
            center_x, center_y = cx + cw / 2, cy + ch / 2
            radius = min(cw, ch) * 0.35
            page.mouse.move(center_x + radius, center_y)
            page.mouse.down()
            for i in range(1, 37):
                angle = (2 * math.pi * i) / 36
                page.mouse.move(center_x + radius * math.cos(angle),
                                center_y + radius * math.sin(angle), steps=3)
            page.mouse.up()
        elif shape == "triangle":
            m = 0.2
            corners = [(cx+cw/2, cy+ch*m), (cx+cw*(1-m), cy+ch*(1-m)),
                        (cx+cw*m, cy+ch*(1-m)), (cx+cw/2, cy+ch*m)]
            page.mouse.move(corners[0][0], corners[0][1])
            page.mouse.down()
            for corner in corners[1:]:
                page.mouse.move(corner[0], corner[1], steps=15)
                time.sleep(0.05)
            page.mouse.up()
        else:
            for i in range(4):
                sx = cx + cw * 0.2 + (i * cw * 0.15)
                sy = cy + ch * 0.3 + (i * ch * 0.1)
                ex = cx + cw * 0.5 + (i * cw * 0.1)
                ey = cy + ch * 0.7 - (i * ch * 0.05)
                page.mouse.move(sx, sy)
                page.mouse.down()
                page.mouse.move(ex, ey, steps=10)
                page.mouse.up()
                time.sleep(0.3)

        time.sleep(0.5)
        page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        const t = (btn.textContent || '').trim().toLowerCase();
        if ((t.includes('complete') || t.includes('done') || t.includes('check') ||
             t.includes('verify') || t.includes('reveal')) &&
            !t.includes('clear') && btn.offsetParent && !btn.disabled) { btn.click(); return; }
    }
})()""")
        time.sleep(0.5)
        return HandlerResult(actions_log=log, needs_extraction=True)

    def handle_audio(self, page) -> HandlerResult:
        log: list[str] = []

        # Inject interception hooks for SpeechSynthesis and Audio BEFORE
        # clicking Play.  Without these, __capturedSpeechUtterance and
        # __capturedAudio stay null and the force-end code below does nothing.
        page.evaluate("""\
(() => {
    if (window.__audioFullPatched) return;
    window.__capturedSpeechTexts = [];
    window.__capturedSpeechUtterance = null;
    window.__speechDone = false;
    window.__capturedAudioSrc = null;
    window.__capturedAudio = null;
    window.__audioFullPatched = true;

    // 1. SpeechSynthesis interception
    if (window.speechSynthesis) {
        const origSpeak = window.speechSynthesis.speak.bind(window.speechSynthesis);
        window.speechSynthesis.speak = function(utterance) {
            window.__capturedSpeechTexts.push(utterance.text);
            window.__capturedSpeechUtterance = utterance;
            return origSpeak(utterance);
        };
    }

    // 2. Audio constructor interception
    const OrigAudio = window.Audio;
    window.Audio = function(src) {
        const audio = new OrigAudio(src);
        window.__capturedAudioSrc = src || null;
        window.__capturedAudio = audio;
        return audio;
    };
    window.Audio.prototype = OrigAudio.prototype;

    // 3. HTMLAudioElement.play interception
    const origPlay = HTMLAudioElement.prototype.play;
    HTMLAudioElement.prototype.play = function() {
        window.__capturedAudioSrc = this.src || this.currentSrc;
        window.__capturedAudio = this;
        return origPlay.call(this);
    };

    // 4. URL.createObjectURL interception for blob audio
    const origCreateObjUrl = URL.createObjectURL;
    URL.createObjectURL = function(obj) {
        const url = origCreateObjUrl.call(URL, obj);
        if (obj instanceof Blob && (obj.type.includes('audio') || obj.type === '')) {
            window.__capturedBlobUrl = url;
            window.__capturedBlob = obj;
        }
        return url;
    };
})()""")
        log.append("audio: injected interception hooks")

        play_result = page.evaluate("""\
(() => {
    const btns = [...document.querySelectorAll('button')];
    for (const btn of btns) {
        const text = (btn.textContent || '').trim().toLowerCase();
        if (text.includes('play') && !text.includes('playing') && btn.offsetParent && !btn.disabled) { btn.click(); return 'clicked'; }
    }
    for (const btn of btns) { if ((btn.textContent||'').trim().toLowerCase().includes('playing')) return 'already_playing'; }
    return 'not_found';
})()""")
        if play_result == "not_found":
            return HandlerResult(actions_log=["audio: no play button"])
        log.append(f"audio: {play_result}")
        time.sleep(3.0)

        page.evaluate("""\
(() => {
    if (window.speechSynthesis) window.speechSynthesis.cancel();
    const utt = window.__capturedSpeechUtterance;
    if (utt) {
        try { utt.dispatchEvent(new SpeechSynthesisEvent('end', {utterance: utt})); } catch(e) {
            try { utt.dispatchEvent(new Event('end')); } catch(e2) {}
        }
        if (utt.onend) { try { utt.onend(new Event('end')); } catch(e) {} }
    }
    if (window.__capturedAudio) {
        window.__capturedAudio.pause();
        if (window.__capturedAudio.duration && isFinite(window.__capturedAudio.duration))
            window.__capturedAudio.currentTime = window.__capturedAudio.duration;
        window.__capturedAudio.dispatchEvent(new Event('ended'));
    }
    document.querySelectorAll('audio').forEach(a => {
        a.pause();
        if (a.duration && isFinite(a.duration)) a.currentTime = a.duration;
        a.dispatchEvent(new Event('ended'));
    });
})()""")
        log.append("audio: force-ended speech")

        # Log what buttons exist now
        btn_texts = page.evaluate("""\
(() => {
    return [...document.querySelectorAll('button')]
        .filter(b => b.offsetParent)
        .map(b => ({text: b.textContent.trim().substring(0, 40), disabled: b.disabled}))
        .slice(0, 15);
})()""")
        logger.info("Audio: buttons after force-end: %s", btn_texts)
        time.sleep(1.0)

        # Try clicking enabled Complete/Done/Finish button
        clicked_complete = False
        for _ in range(6):
            clicked = page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        const text = (btn.textContent || '').trim().toLowerCase();
        if ((text.includes('complete') || text.includes('done') || text.includes('finish')) &&
            !text.includes('playing') && btn.offsetParent && !btn.disabled) { btn.click(); return true; }
    }
    return false;
})()""")
            if clicked:
                log.append("audio: clicked complete")
                clicked_complete = True
                break
            time.sleep(0.5)

        if not clicked_complete:
            logger.info("Audio: Complete button not found/clicked — entering force-enable path")

            # Step A: Force speechSynthesis to report not-speaking, then re-dispatch
            # end events. This ensures React's internal checks see audio as finished.
            page.evaluate("""\
(() => {
    // Override speechSynthesis to appear finished
    if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
        try {
            Object.defineProperty(window.speechSynthesis, 'speaking', {get: () => false, configurable: true});
            Object.defineProperty(window.speechSynthesis, 'pending', {get: () => false, configurable: true});
        } catch(e) {}
    }
    // Re-dispatch end events
    const utt = window.__capturedSpeechUtterance;
    if (utt) {
        try { utt.dispatchEvent(new SpeechSynthesisEvent('end', {utterance: utt})); } catch(e) {
            try { utt.dispatchEvent(new Event('end')); } catch(e2) {}
        }
        if (utt.onend) { try { utt.onend(new Event('end')); } catch(e) {} }
    }
    document.querySelectorAll('audio').forEach(a => {
        a.pause();
        if (a.duration && isFinite(a.duration)) a.currentTime = a.duration;
        a.dispatchEvent(new Event('ended'));
    });
})()""")
            time.sleep(0.5)

            # Step B: Try to find and force-enable a disabled Complete button
            force_result = page.evaluate("""\
(() => {
    const keywords = ['complete', 'done', 'finish'];
    for (const btn of document.querySelectorAll('button')) {
        const text = (btn.textContent || '').trim().toLowerCase();
        if (keywords.some(k => text.includes(k)) && !text.includes('playing')) {
            btn.disabled = false;
            btn.removeAttribute('disabled');
            btn.removeAttribute('aria-disabled');
            btn.style.pointerEvents = 'auto';
            btn.style.opacity = '1';
            btn.click();
            return 'force-enabled-clicked';
        }
    }
    return 'none';
})()""")
            log.append(f"audio: force-enable result={force_result}")

            # Step C: If no Complete button exists (button stuck as "Playing..."),
            # use React's dispatch mechanism to flip audio state properly.
            if force_result == 'none':
                logger.info("Audio: no Complete button — forcing React state via dispatch")
                react_result = page.evaluate("""\
(() => {
    function getFiber(el) {
        for (const k of Object.keys(el)) {
            if (k.startsWith('__reactFiber$') || k.startsWith('__reactInternalInstance$'))
                return el[k];
        }
        return null;
    }

    // Find the "Playing..." button to locate the audio component
    let targetBtn = null;
    for (const btn of document.querySelectorAll('button')) {
        const t = (btn.textContent || '').trim().toLowerCase();
        if (t.includes('playing')) { targetBtn = btn; break; }
    }
    if (!targetBtn) return 'no-playing-btn';

    const fiber = getFiber(targetBtn);
    if (!fiber) return 'no-fiber';

    const audioKeys = ['playing','isPlaying','audioPlaying','isSpeaking','speaking'];
    const doneKeys = ['completed','isCompleted','done','isDone','audioComplete','audioCompleted','speechDone','isFinished'];

    // Walk UP from button to find the audio component with boolean hooks.
    // Use queue.dispatch for proper React state updates + re-render.
    let dispatched = 0;
    let node = fiber;
    for (let i = 0; i < 20 && node; i++) {
        if (node.memoizedState && typeof node.memoizedState === 'object' && node.memoizedState.queue !== undefined) {
            // Collect boolean hooks and named-key hooks
            const boolHooks = [];
            let hook = node.memoizedState;
            for (let h = 0; h < 20 && hook; h++) {
                if (typeof hook.memoizedState === 'boolean' && hook.queue && hook.queue.dispatch) {
                    boolHooks.push({val: hook.memoizedState, dispatch: hook.queue.dispatch});
                }
                // Object state with named audio keys
                if (hook.memoizedState && typeof hook.memoizedState === 'object' &&
                    !Array.isArray(hook.memoizedState) && hook.queue && hook.queue.dispatch) {
                    const ms = hook.memoizedState;
                    let newState = Object.assign({}, ms);
                    let changed = false;
                    for (const k of audioKeys) { if (k in ms && ms[k] === true) { newState[k] = false; changed = true; } }
                    for (const k of doneKeys) { if (k in ms && ms[k] === false) { newState[k] = true; changed = true; } }
                    if (changed) { try { hook.queue.dispatch(newState); dispatched++; } catch(e) {} }
                }
                hook = hook.next;
            }
            // 2+ boolean hooks on same component = likely audio (isPlaying + isCompleted)
            if (boolHooks.length >= 2) {
                for (const bh of boolHooks) { try { bh.dispatch(!bh.val); dispatched++; } catch(e) {} }
                if (dispatched > 0) break;
            }
            // Single true boolean = likely isPlaying, flip it
            if (boolHooks.length === 1 && boolHooks[0].val === true) {
                try { boolHooks[0].dispatch(false); dispatched++; } catch(e) {}
            }
        }
        // Class components: use setState
        if (node.stateNode && node.stateNode.setState && node.stateNode.state) {
            const s = node.stateNode.state;
            const update = {};
            let changed = false;
            for (const k of audioKeys) { if (k in s && s[k] === true) { update[k] = false; changed = true; } }
            for (const k of doneKeys) { if (k in s && s[k] === false) { update[k] = true; changed = true; } }
            if (changed) { try { node.stateNode.setState(update); dispatched++; } catch(e) {} }
        }
        if (dispatched > 0) break;
        node = node.return;
    }
    return dispatched > 0 ? 'dispatched-' + dispatched : 'no-audio-state';
})()""")
                log.append(f"audio: react-dispatch result={react_result}")
                logger.info("Audio: React dispatch result: %s", react_result)
                time.sleep(1.5)

            # Step D: After state manipulation, try clicking Complete/Done/Finish again
            for _ in range(6):
                clicked = page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        const text = (btn.textContent || '').trim().toLowerCase();
        if ((text.includes('complete') || text.includes('done') || text.includes('finish'))
            && !text.includes('playing') && btn.offsetParent && !btn.disabled) {
            btn.click(); return true;
        }
    }
    return false;
})()""")
                if clicked:
                    log.append("audio: clicked complete (after state fix)")
                    break
                time.sleep(0.5)

            # Step E: Last resort — click the "Playing..." button itself
            # (sometimes the click handler transitions state regardless)
            else:
                page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        const text = (btn.textContent || '').trim().toLowerCase();
        if (text.includes('playing') && btn.offsetParent) {
            btn.disabled = false;
            btn.removeAttribute('disabled');
            btn.click();
        }
    }
})()""")
                log.append("audio: clicked Playing button as last resort")

        time.sleep(1.0)
        # Log final page state for debugging
        final_btns = page.evaluate("""\
(() => {
    return [...document.querySelectorAll('button')]
        .filter(b => b.offsetParent)
        .map(b => ({text: b.textContent.trim().substring(0, 40), disabled: b.disabled}))
        .slice(0, 10);
})()""")
        logger.info("Audio: final buttons: %s", final_btns)
        return HandlerResult(actions_log=log, needs_extraction=True)

    def handle_websocket(self, page) -> HandlerResult:
        codes: list[str] = []
        log: list[str] = []
        connected = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    if (text.includes('Connected') || text.includes('\u25cf Connected')) return 'already';
    for (const btn of document.querySelectorAll('button')) {
        const t = (btn.textContent || '').trim().toLowerCase();
        if ((t.includes('connect') || t === 'connect') && !btn.disabled && btn.offsetParent) { btn.click(); return 'clicked'; }
    }
    return null;
})()""")
        if not connected:
            return HandlerResult(actions_log=["websocket: no connect button"])
        log.append(f"websocket: {connected}")

        for i in range(25):
            time.sleep(0.5)
            status = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    if (text.match(/\\b[A-Z0-9]{6}\\b/) && (text.includes('code') || text.includes('Code'))) return 'has_code';
    for (const btn of document.querySelectorAll('button')) {
        const t = (btn.textContent || '').trim().toLowerCase();
        if (t.includes('reveal') && !btn.disabled && btn.offsetParent) return 'ready';
    }
    if (text.includes('Ready to reveal') || text.includes('reveal code')) return 'ready';
    if (text.includes('Connected') || text.includes('\u25cf Connected')) return 'connected';
    return 'waiting';
})()""")
            if status in ("has_code", "ready"):
                break
            if status == "connected" and i > 10:
                break

        for _ in range(5):
            clicked = page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        const t = (btn.textContent || '').trim().toLowerCase();
        if ((t.includes('reveal') || t.includes('request') || t.includes('get code') ||
             t.includes('send') || t.includes('extract')) &&
            !t.includes('connect') && !btn.disabled && btn.offsetParent) { btn.click(); return t.substring(0, 30); }
    }
    return null;
})()""")
            if clicked:
                log.append(f"websocket: clicked '{clicked}'")
                time.sleep(0.8)
            else:
                time.sleep(0.5)
                continue
            code = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    for (const el of document.querySelectorAll('.text-green-600, .text-green-700, .bg-green-100, .bg-green-50')) {
        const m = (el.textContent || '').match(/\\b([A-Z0-9]{6})\\b/); if (m) return m[1];
    }
    const m = text.match(/(?:code|Code|CODE)[^:]*?:\\s*([A-Z0-9]{6})/); if (m) return m[1];
    for (const el of document.querySelectorAll('[class*="cyan"], [class*="terminal"], pre, code, [class*="mono"]')) {
        const m2 = (el.textContent || '').match(/\\b([A-Z0-9]{6})\\b/); if (m2) return m2[1];
    }
    return null;
})()""")
            if code and not is_false_positive(code):
                codes.append(code)
                return HandlerResult(codes_found=codes, actions_log=log, success=True)
        return HandlerResult(codes_found=codes, actions_log=log, needs_extraction=True)

    def handle_service_worker(self, page) -> HandlerResult:
        codes: list[str] = []
        log: list[str] = []
        registered = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    if (text.includes('Registered') || text.includes('\u25cf Registered')) return 'already';
    for (const btn of document.querySelectorAll('button')) {
        const t = (btn.textContent || '').trim().toLowerCase();
        if ((t.includes('register') && t.includes('service')) || (t.includes('register') && !btn.disabled && btn.offsetParent)) {
            btn.click(); return 'clicked';
        }
    }
    return null;
})()""")
        if not registered:
            return HandlerResult(actions_log=["service_worker: no register button"])
        log.append(f"service_worker: {registered}")

        for _ in range(15):
            cached = page.evaluate("(() => { const t = document.body.textContent || ''; return t.includes('Cached') || t.includes('cached') || t.includes('\u25cf Cached'); })()")
            if cached:
                break
            time.sleep(0.3)

        page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        const t = (btn.textContent || '').trim().toLowerCase();
        if ((t.includes('retrieve') || t.includes('from cache')) && !btn.disabled && btn.offsetParent) { btn.click(); return true; }
    }
    return false;
})()""")
        time.sleep(0.5)

        code = page.evaluate("""\
(() => {
    for (const el of document.querySelectorAll('.bg-green-100, .bg-green-50, .text-green-600, .text-green-700, .border-green-500')) {
        const m = (el.textContent || '').match(/\\b([A-Z0-9]{6})\\b/); if (m) return m[1];
    }
    const text = document.body.textContent || '';
    const m = text.match(/(?:code\\s+(?:is|retrieved)[^:]*:\\s*)([A-Z0-9]{6})/i); if (m) return m[1].toUpperCase();
    const m2 = text.match(/(?:retrieved|cache)[^.]*?\\b([A-Z0-9]{6})\\b/i); if (m2) return m2[1].toUpperCase();
    return null;
})()""")
        if code and not is_false_positive(code):
            codes.append(code)
        return HandlerResult(codes_found=codes, actions_log=log, needs_extraction=True)

    def _iframe_js_click(self, page, js_find_and_click_expr: str) -> bool:
        """Find element via JS and click it with element.click().
        Works for regular <button> elements. Returns True if click dispatched."""
        try:
            return bool(page.evaluate(js_find_and_click_expr))
        except Exception as e:
            logger.debug("Iframe JS click failed: %s", e)
        return False

    def _iframe_mouse_click(self, page, js_find_expr: str) -> bool:
        """Find element via JS, get bbox, use page.mouse.click for native events.
        Needed for React component divs where JS .click() doesn't trigger handlers."""
        try:
            handle = page.evaluate_handle(js_find_expr)
            box = page.evaluate("""(el) => {
                if (!el || !el.getBoundingClientRect) return null;
                el.scrollIntoView({behavior: 'instant', block: 'center'});
                const r = el.getBoundingClientRect();
                if (r.width === 0 || r.height === 0) return null;
                return {x: r.x + r.width / 2, y: r.y + r.height / 2};
            }""", handle)
            if box:
                page.mouse.click(box["x"], box["y"])
                return True
        except Exception as e:
            logger.debug("Iframe mouse click failed: %s", e)
        return False

    def handle_iframe(self, page) -> HandlerResult:
        logger.info("Iframe handler started")
        codes: list[str] = []
        log: list[str] = []
        extract_attempts = 0
        deepest_click_attempts = 0
        last_clicked_level = -1
        level_div_stuck_count = 0
        prev_current = -1
        for click_round in range(20):
            # Dismiss ALL overlays/popups every round — click close buttons
            # first (updates React state), then CSS-hide remaining overlays.
            page.evaluate("""\
(() => {
    const hide = el => { el.style.display='none'; el.style.pointerEvents='none'; el.style.visibility='hidden'; el.style.zIndex='-1'; };
    // Step 1: Find ALL fixed/absolute overlays with high z-index
    const overlays = [];
    document.querySelectorAll('.fixed, [class*="absolute"], [style*="position: fixed"], [style*="position:fixed"]').forEach(el => {
        const s = getComputedStyle(el);
        const z = parseInt(s.zIndex) || 0;
        const r = el.getBoundingClientRect();
        if (r.width < 50 || r.height < 50) return;
        const t = (el.textContent || '').trim();
        // Skip challenge-essential elements
        if (t.includes('Step') && t.includes('of 30')) return;
        if (el.querySelector('input[type="text"]')) return;
        if (t.includes('Iframe Challenge') || t.includes('Recursive Iframe')) return;
        // Skip level navigation divs
        if (t.includes('Enter Level') && t.includes('border')) return;
        overlays.push({el, z, t: t.substring(0, 100)});
    });
    // Step 2: Click close/dismiss buttons on popup overlays
    let dismissed = 0;
    for (const {el, z, t} of overlays) {
        // Known popup patterns (decoy popups at deepest iframe level)
        const isPopup = t.includes('Wrong Button') || t.includes('Try Again')
            || t.includes('Important Notice') || t.includes('Overlay Notice')
            || t.includes('amazing deals') || t.includes('won a prize')
            || t.includes('Congratulations') || t.includes('Special Offer')
            || t.includes('Act Now') || t.includes('Limited Time')
            || t.includes('Click here for') || t.includes('Subscribe')
            || (z >= 9000 && !t.includes('Level') && !t.includes('Extract')
                && !t.includes('Enter Code') && !t.includes('Submit'));
        if (!isPopup) continue;
        // Click close/dismiss buttons within the popup
        const closeBtns = el.querySelectorAll('button, [role="button"], .cursor-pointer');
        for (const btn of closeBtns) {
            const bt = (btn.textContent || '').trim().toLowerCase();
            if (bt.includes('close') || bt.includes('dismiss') || bt.includes('x')
                || bt.includes('no thanks') || bt.includes('cancel')
                || bt.includes('got it') || bt === 'x' || bt === '\u00d7'
                || bt === '\u2715' || bt === '\u2716') {
                try { btn.click(); dismissed++; } catch(e) {}
            }
        }
        // Also try the last button (often the dismiss action)
        if (closeBtns.length > 0) {
            try { closeBtns[closeBtns.length - 1].click(); dismissed++; } catch(e) {}
        }
        // CSS-hide the popup regardless
        hide(el);
    }
    // Step 3: Make ALL dark backdrop overlays click-through
    document.querySelectorAll('.fixed, [style*="position: fixed"]').forEach(el => {
        const cls = el.className || '';
        const bg = el.style.backgroundColor || '';
        if (cls.includes('bg-black') || bg.includes('rgba(0') || cls.includes('bg-opacity')
            || cls.includes('bg-gray-900') || cls.includes('backdrop')) {
            if (!(el.textContent||'').includes('Step') && !el.querySelector('input')) {
                el.style.pointerEvents = 'none';
                el.style.opacity = '0';
                el.style.zIndex = '-1';
            }
        }
    });
    return dismissed;
})()""")
            page.evaluate("""\
(() => {
    for (const el of document.querySelectorAll('div')) {
        const t = (el.textContent || '').substring(0, 200);
        if (t.includes('Iframe Challenge') || t.includes('Recursive Iframe')) {
            el.scrollIntoView({behavior: 'instant', block: 'center'}); break;
        }
    }
})()""")
            time.sleep(0.2)
            result = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    const codeEls = document.querySelectorAll('.text-green-600, .text-green-700, .bg-green-100, .bg-green-50, .bg-emerald-100, .text-emerald-600, [class*="success"]');
    for (const el of codeEls) {
        const m = (el.textContent || '').match(/\\b([A-Z0-9]{6})\\b/);
        if (m && !['IFRAME','BWRONG','1WRONG','CWRONG'].includes(m[1])) return {done: true, code: m[1]};
    }
    const codeMatch = text.match(/(?:code|Code)[^:]*?:\\s*([A-Z0-9]{6})/);
    if (codeMatch && !['IFRAME','BWRONG','1WRONG','CWRONG'].includes(codeMatch[1])) return {done: true, code: codeMatch[1]};
    const depthMatch = text.match(/(?:depth|level)[^:]*?:\\s*(\\d+)\\s*\\/\\s*(\\d+)/i) ||
                       text.match(/(\\d+)\\s*\\/\\s*(\\d+)\\s*(?:depth|levels?)/i);
    const current = depthMatch ? parseInt(depthMatch[1]) : 0;
    const total = depthMatch ? parseInt(depthMatch[2]) : 4;
    const levelDivs = [];
    for (const div of document.querySelectorAll('div[class*="border-2"][class*="rounded"]')) {
        const cls = div.getAttribute('class') || '';
        const firstText = (div.childNodes[0]?.textContent || '').trim();
        const levelMatch = firstText.match(/(?:Iframe\\s+)?Level\\s+(\\d+)/i);
        if (levelMatch && div.offsetParent && div.offsetWidth > 50) {
            const isComplete = firstText.includes('\u2713') || firstText.includes('\u2714') || cls.includes('emerald');
            levelDivs.push({level: parseInt(levelMatch[1]), complete: isComplete, text: firstText.substring(0, 40)});
        }
    }
    const enterBtns = [];
    let extractBtn = false;
    for (const btn of document.querySelectorAll('button')) {
        if (!btn.offsetParent || btn.disabled) continue;
        const r = btn.getBoundingClientRect();
        if (r.width === 0) continue;
        const t = (btn.textContent || '').trim().toLowerCase();
        const enterMatch = t.match(/enter\\s+level\\s+(\\d+)/i) || t.match(/level\\s+(\\d+)/i);
        if (enterMatch && !t.includes('submit') && !t.includes('extract')) enterBtns.push({level: parseInt(enterMatch[1]), text: t});
        if (t.includes('go deeper') || t.includes('next level') || t.includes('descend')) enterBtns.push({level: current + 1, text: t});
        if (t.includes('extract') || t.includes('get code')) extractBtn = true;
    }
    const atDeepest = text.includes('reached the deepest') || text.includes("You've reached")
        || text.includes('You have reached the deepest');
    return {current, total, levelDivs, enterBtns, extractBtn, atDeepest};
})()""")
            if not result:
                break
            if result.get("done") and result.get("code"):
                c = result["code"]
                if not is_false_positive(c):
                    codes.append(c)
                    return HandlerResult(codes_found=codes, actions_log=log, success=True)

            current = result.get("current", 0)
            total = result.get("total", 4)
            logger.info("Iframe round %d: depth=%d/%d divs=%s btns=%s extract=%s deepest=%s",
                        click_round, current, total,
                        result.get("levelDivs"), result.get("enterBtns"),
                        result.get("extractBtn"), result.get("atDeepest"))

            # PRIORITY 1: Click incomplete level divs.
            # At deepest level, try once to click the deepest div (may need
            # activation), then move to Extract Code.
            level_divs = result.get("levelDivs", [])
            incomplete = [d for d in level_divs if not d.get("complete")]
            at_deepest = result.get("atDeepest") or (total > 0 and current >= total)
            max_stuck = 1 if at_deepest else 3
            if incomplete and level_div_stuck_count < max_stuck:
                incomplete.sort(key=lambda d: d["level"])
                target = incomplete[0]
                if target["level"] == last_clicked_level and current == prev_current:
                    level_div_stuck_count += 1
                else:
                    level_div_stuck_count = 0
                if level_div_stuck_count < max_stuck:
                    # Try Playwright locator click (native browser events)
                    try:
                        loc = page.locator(f"div[class*='border-2'][class*='rounded']").filter(
                            has_text=f"Level {target['level']}"
                        )
                        if loc.count() > 0:
                            loc.first.scroll_into_view_if_needed(timeout=2000)
                            loc.first.click(timeout=2000)
                    except Exception:
                        pass
                    # Also try JS click as backup
                    self._iframe_js_click(page, f"""\
(() => {{
    for (const div of document.querySelectorAll('div[class*="border-2"][class*="rounded"]')) {{
        const t = (div.childNodes[0]?.textContent || '').trim();
        if (t.includes('Level {target["level"]}') && !t.includes('\u2713') && !t.includes('\u2714')) {{
            div.scrollIntoView({{behavior: 'instant', block: 'center'}});
            div.click();
            return true;
        }}
    }}
    return false;
}})()""")
                    log.append(f"iframe: clicked level div {target['level']} ({current}/{total})")
                    last_clicked_level = target["level"]
                    prev_current = current
                    extract_attempts = 0
                    time.sleep(0.4)
                    continue

            # PRIORITY 2: Click Enter Level buttons using JS click
            # (regular <button> elements where JS .click() works fine)
            enter_btns = result.get("enterBtns", [])
            if not at_deepest and enter_btns:
                target_level = current + 1
                enter_btns.sort(key=lambda b: abs(b["level"] - target_level))
                target = enter_btns[0]
                clicked = self._iframe_js_click(page, f"""\
(() => {{
    for (const btn of document.querySelectorAll('button')) {{
        if (!btn.offsetParent || btn.disabled) continue;
        const t = btn.textContent.trim().toLowerCase();
        if (t.includes('{target["text"][:25]}')) {{
            btn.scrollIntoView({{behavior: 'instant', block: 'center'}});
            btn.click();
            return true;
        }}
    }}
    return false;
}})()""")
                if clicked:
                    log.append(f"iframe: js-clicked '{target['text']}' ({current}/{total})")
                    last_clicked_level = -1
                    level_div_stuck_count = 0
                    prev_current = current
                    extract_attempts = 0
                    time.sleep(0.3)
                    continue

            # At deepest level: try to complete the deepest level div via
            # single click + JS click + mouse click (React onClick doesn't
            # fire on dblclick, so we use single click methods instead)
            if incomplete and at_deepest and deepest_click_attempts < 3:
                deepest_click_attempts += 1
                target = incomplete[0]
                logger.info("Iframe: deepest level %d click attempt %d/3", target['level'], deepest_click_attempts)
                # Method 1: Playwright single click
                try:
                    loc = page.locator(f"div[class*='border-2'][class*='rounded']").filter(
                        has_text=f"Level {target['level']}"
                    )
                    if loc.count() > 0:
                        loc.first.scroll_into_view_if_needed(timeout=2000)
                        loc.first.click(timeout=2000)
                        log.append(f"iframe: clicked level {target['level']} (deepest attempt {deepest_click_attempts})")
                except Exception:
                    pass
                # Method 2: JS click + React event dispatch
                page.evaluate(f"""\
(() => {{
    for (const div of document.querySelectorAll('div[class*="border-2"][class*="rounded"]')) {{
        const t = (div.childNodes[0]?.textContent || '').trim();
        if (t.includes('Level {target["level"]}') && !t.includes('\u2713') && !t.includes('\u2714')) {{
            div.scrollIntoView({{behavior: 'instant', block: 'center'}});
            div.click();
            div.dispatchEvent(new MouseEvent('click', {{bubbles:true, cancelable:true, view:window}}));
            // Also invoke React onClick via fiber
            const fk = Object.keys(div).find(k => k.startsWith('__reactFiber$') || k.startsWith('__reactProps$'));
            if (fk) {{
                const obj = div[fk];
                const onClick = obj.onClick || obj.memoizedProps?.onClick || obj.pendingProps?.onClick;
                if (typeof onClick === 'function') {{
                    try {{ onClick({{preventDefault:()=>{{}},stopPropagation:()=>{{}},target:div,currentTarget:div}}); }} catch(e) {{}}
                }}
            }}
            return true;
        }}
    }}
    return false;
}})()""")
                # Method 3: Mouse click at coordinates
                try:
                    rect = page.evaluate(f"""\
(() => {{
    for (const div of document.querySelectorAll('div[class*="border-2"][class*="rounded"]')) {{
        const t = (div.childNodes[0]?.textContent || '').trim();
        if (t.includes('Level {target["level"]}') && !t.includes('\u2713') && !t.includes('\u2714')) {{
            const r = div.getBoundingClientRect();
            return {{x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2)}};
        }}
    }}
    return null;
}})()""")
                    if rect:
                        page.mouse.click(rect["x"], rect["y"])
                except Exception:
                    pass
                time.sleep(0.5)

            # PRIORITY 3: At deepest level — catch the moving button
            # Debug: on first extract attempt, dump the DOM structure
            if at_deepest and extract_attempts == 0:
                dom_info = page.evaluate("""\
(() => {
    const info = {};
    // All buttons with details
    info.buttons = [...document.querySelectorAll('button')].map(btn => {
        const r = btn.getBoundingClientRect();
        const s = getComputedStyle(btn);
        return {
            text: btn.textContent.trim().substring(0, 50),
            cls: (btn.getAttribute('class') || '').substring(0, 100),
            pos: s.position,
            disabled: btn.disabled,
            visible: btn.offsetParent !== null || s.position === 'fixed',
            rect: {x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height)},
            pe: s.pointerEvents
        };
    }).filter(b => b.rect.w > 0);
    // Incomplete level div innerHTML
    for (const div of document.querySelectorAll('div[class*="border-2"][class*="rounded"]')) {
        const t = (div.childNodes[0]?.textContent || '').trim();
        if (t.includes('Level') && !t.includes('\u2713') && !t.includes('\u2714')) {
            info.levelDiv = {text: t, cls: (div.getAttribute('class')||'').substring(0,100),
                html: div.innerHTML.substring(0, 500)};
        }
    }
    // Elements with pointer-events-auto class
    info.pointerEventsAuto = [...document.querySelectorAll('.pointer-events-auto, [class*="pointer-events-auto"]')]
        .map(el => ({tag: el.tagName, text: (el.textContent||'').trim().substring(0,50),
            cls: (el.getAttribute('class')||'').substring(0,80)}));
    // Elements with animation
    info.animated = [];
    document.querySelectorAll('*').forEach(el => {
        const s = getComputedStyle(el);
        if (s.animation && s.animation !== 'none' && s.animationName !== 'none') {
            const r = el.getBoundingClientRect();
            info.animated.push({tag: el.tagName, text: (el.textContent||'').trim().substring(0,50),
                anim: s.animationName, rect: {x:Math.round(r.x),y:Math.round(r.y),w:Math.round(r.width),h:Math.round(r.height)}});
        }
    });
    return info;
})()""")
                logger.debug("Iframe DOM at deepest level: %s", dom_info)

            if at_deepest and (result.get("extractBtn") or extract_attempts > 0):
                extract_attempts += 1

                # Skip animated button detection — they're ALL decoys at the
                # deepest level (trap buttons like "Next Step", "Click Here!",
                # "Continue").  Go straight to Extract Code.

                # On first attempt, hide all animated decoys so they don't
                # intercept clicks on Extract Code.
                if extract_attempts == 1:
                    hidden = page.evaluate("""\
(() => {
    let count = 0;
    const hide = el => { el.style.display='none'; el.style.pointerEvents='none'; el.style.visibility='hidden'; el.style.zIndex='-1'; };
    document.querySelectorAll('*').forEach(el => {
        const s = getComputedStyle(el);
        const cls = el.getAttribute('class') || '';
        const hasAnim = (s.animation && s.animation !== 'none' && s.animationName !== 'none') ||
            cls.includes('animate-');
        const r = el.getBoundingClientRect();
        const isSmall = r.width > 10 && r.width < 300 && r.height > 10 && r.height < 200;
        const t = (el.textContent||'').trim().toLowerCase();
        if (hasAnim && isSmall && !t.includes('extract') && !t.includes('get code')) {
            hide(el); count++;
        }
    });
    // Dismiss ALL popup overlays — click close buttons, then CSS-hide
    document.querySelectorAll('.fixed, [class*="absolute"], [style*="position: fixed"], [style*="position:fixed"]').forEach(el => {
        const t = (el.textContent||'').trim();
        const r = el.getBoundingClientRect();
        if (r.width < 30 || r.height < 30) return;
        // Skip challenge-essential elements
        if ((t.includes('Step') && t.includes('of 30')) || el.querySelector('input[type="text"]')) return;
        if (t.includes('Iframe Challenge') || t.includes('Recursive Iframe')) return;
        if (t.includes('Extract Code') || t.includes('Get Code')) return;
        if (t.includes('Enter Level') && el.querySelector('[class*="border-2"]')) return;
        // Click close/dismiss buttons first (updates React state)
        const btns = el.querySelectorAll('button, [role="button"]');
        for (const btn of btns) {
            const bt = (btn.textContent||'').trim().toLowerCase();
            if (bt.includes('close') || bt.includes('dismiss') || bt === 'x'
                || bt === '\u00d7' || bt === '\u2715' || bt.includes('no thanks')
                || bt.includes('cancel') || bt.includes('got it') || bt.includes('ok')) {
                try { btn.click(); count++; } catch(e) {}
            }
        }
        hide(el);
    });
    // Also hide high-z-index floating decoys by text pattern
    document.querySelectorAll('div, a, span').forEach(el => {
        const s = getComputedStyle(el);
        if ((s.position === 'absolute' || s.position === 'fixed') && parseInt(s.zIndex) > 10) {
            const t = (el.textContent||'').trim();
            const decoyTexts = ['Click Here!','Try This!','Button!','Link!','Click Me!','Here!',
                'Moving!','Next Step','Continue','Move On','Important Notice','amazing deals',
                'won a prize','Congratulations','Special Offer','Act Now'];
            for (const d of decoyTexts) { if (t.includes(d)) { hide(el); break; } }
        }
    });
    return count;
})()""")
                    logger.info("Iframe: hid %d animated decoys at deepest level", hidden)
                    time.sleep(0.3)

                    # Try interacting with elements INSIDE the deepest level div
                    # to trigger completion (click buttons, links, spans within it)
                    inner_clicks = page.evaluate("""\
(() => {
    const clicks = [];
    for (const div of document.querySelectorAll('div[class*="border-2"][class*="rounded"]')) {
        const t = (div.childNodes[0]?.textContent || '').trim();
        if (!t.includes('Level') || t.includes('\u2713') || t.includes('\u2714')) continue;
        // Click all interactive children inside this level div
        const children = div.querySelectorAll('button, a, [role="button"], [onclick], [class*="cursor-pointer"]');
        for (const child of children) {
            const ct = (child.textContent||'').trim();
            if (ct.toLowerCase().includes('extract') || ct.toLowerCase().includes('get code')) continue;
            try { child.click(); clicks.push(ct.substring(0, 30)); } catch(e) {}
        }
        // Also click the div itself
        try { div.click(); clicks.push('div-self'); } catch(e) {}
    }
    return clicks;
})()""")
                    if inner_clicks:
                        logger.info("Iframe: clicked %d inner elements: %s", len(inner_clicks), inner_clicks[:5])
                    time.sleep(0.5)

                if extract_attempts > 8:
                    # Fiber search as fallback after 8 extract attempts
                    fiber_code = page.evaluate("""\
(() => {
    const codes = new Set();
    const fp = new Set(['IFRAME','BWRONG','1WRONG','CWRONG','STEPGO','WORKER','LOADED','DWRONG','EWRONG','FWRONG','GWRONG','HWRONG','METHIS','REGIST','PAGEGO']);
    function searchFiber(fiber, depth) {
        if (!fiber || depth > 50) return;
        let state = fiber.memoizedState;
        for (let i = 0; i < 20 && state; i++) {
            const val = state.memoizedState;
            if (typeof val === 'string') {
                const m = val.match(/^[A-Z0-9]{6}$/);
                if (m && !fp.has(m[0]) && !m[0].endsWith('WRONG')) codes.add(m[0]);
            }
            if (typeof val === 'object' && val) {
                for (const v of Object.values(val)) {
                    if (typeof v === 'string') {
                        const m2 = v.match(/^[A-Z0-9]{6}$/);
                        if (m2 && !fp.has(m2[0]) && !m2[0].endsWith('WRONG')) codes.add(m2[0]);
                    }
                }
            }
            state = state.next;
        }
        const props = fiber.memoizedProps || fiber.pendingProps;
        if (props) {
            for (const v of Object.values(props)) {
                if (typeof v === 'string') {
                    const m = v.match(/^[A-Z0-9]{6}$/);
                    if (m && !fp.has(m[0]) && !m[0].endsWith('WRONG')) codes.add(m[0]);
                }
            }
        }
        if (fiber.child) searchFiber(fiber.child, depth + 1);
        if (fiber.sibling) searchFiber(fiber.sibling, depth + 1);
    }
    const root = document.querySelector('#root');
    const key = Object.keys(root).find(k => k.startsWith('__reactFiber$') || k.startsWith('__reactContainer$'));
    if (key) searchFiber(root[key], 0);
    return [...codes];
})()""")
                    if fiber_code:
                        logger.info("Iframe: React fiber codes: %s", fiber_code)
                        codes.extend(fiber_code)
                        return HandlerResult(codes_found=codes, actions_log=log, success=True)
                    logger.info("Iframe: Extract clicked %dx + no animated/fiber codes, bailing", extract_attempts)
                    break

                # After 3 extract attempts, try React dispatch to force level completion
                if extract_attempts == 3:
                    logger.info("Iframe: trying React dispatch to force deepest level completion")
                    page.evaluate("""\
(() => {
    function getFiber(el) {
        for (const k of Object.keys(el)) {
            if (k.startsWith('__reactFiber$') || k.startsWith('__reactInternalInstance$'))
                return el[k];
        }
        return null;
    }
    // Find the deepest incomplete level div
    let targetDiv = null;
    for (const div of document.querySelectorAll('div[class*="border-2"][class*="rounded"]')) {
        const t = (div.childNodes[0]?.textContent || '').trim();
        if (t.includes('Level') && !t.includes('\u2713') && !t.includes('\u2714'))
            targetDiv = div;
    }
    if (!targetDiv) return;
    const fiber = getFiber(targetDiv);
    if (!fiber) return;
    // Walk up to find component with hooks
    let node = fiber;
    for (let i = 0; i < 20 && node; i++) {
        if (node.memoizedState && typeof node.memoizedState === 'object') {
            let hook = node.memoizedState;
            for (let h = 0; h < 20 && hook; h++) {
                if (typeof hook.memoizedState === 'boolean' && hook.queue && hook.queue.dispatch) {
                    // Toggle: if false (likely isComplete), set to true
                    if (hook.memoizedState === false) {
                        try { hook.queue.dispatch(true); } catch(e) {}
                    }
                }
                if (hook.memoizedState && typeof hook.memoizedState === 'object' &&
                    !Array.isArray(hook.memoizedState) && hook.queue && hook.queue.dispatch) {
                    const ms = hook.memoizedState;
                    const keys = ['complete','completed','isComplete','extracted','done','hasCode','codeExtracted'];
                    let newState = Object.assign({}, ms);
                    let changed = false;
                    for (const k of keys) { if (k in ms && ms[k] === false) { newState[k] = true; changed = true; } }
                    if (changed) { try { hook.queue.dispatch(newState); } catch(e) {} }
                }
                hook = hook.next;
            }
        }
        node = node.return;
    }
})()""")
                    time.sleep(1.0)

                # Force-complete the deepest level via React fiber if it's still
                # incomplete — the Extract Code button checks this state.
                if incomplete:
                    page.evaluate("""\
(() => {
    function getFiber(el) {
        for (const k of Object.keys(el)) {
            if (k.startsWith('__reactFiber$') || k.startsWith('__reactInternalInstance$'))
                return el[k];
        }
        return null;
    }
    // Find all incomplete level divs and force-complete them
    for (const div of document.querySelectorAll('div[class*="border-2"][class*="rounded"]')) {
        const t = (div.childNodes[0]?.textContent || '').trim();
        if (!t.includes('Level') || t.includes('\u2713') || t.includes('\u2714')) continue;
        // Click it natively first
        div.click();
        // Then walk the React fiber to flip boolean state hooks
        const fiber = getFiber(div);
        if (!fiber) continue;
        let node = fiber;
        for (let i = 0; i < 25 && node; i++) {
            if (node.memoizedState && typeof node.memoizedState === 'object' && node.memoizedState.queue !== undefined) {
                let hook = node.memoizedState;
                for (let h = 0; h < 20 && hook; h++) {
                    if (typeof hook.memoizedState === 'boolean' && hook.memoizedState === false
                        && hook.queue && hook.queue.dispatch) {
                        try { hook.queue.dispatch(true); } catch(e) {}
                    }
                    hook = hook.next;
                }
            }
            node = node.return;
        }
    }
})()""")
                    time.sleep(0.5)

                # Aggressively dismiss ALL popups and overlays before Extract
                # Code — click Close buttons first (to update React state),
                # then CSS-hide everything blocking the Extract Code button.
                dismissed = page.evaluate("""\
(() => {
    const hide = el => { el.style.display='none'; el.style.pointerEvents='none'; el.style.visibility='hidden'; el.style.zIndex='-1'; };
    let dismissed = 0;
    // Hide elements we froze at center
    document.querySelectorAll('[data-iframe-anim]').forEach(el => hide(el));
    // Find and dismiss ALL popup overlays by clicking their close buttons
    const allPositioned = document.querySelectorAll('.fixed, [class*="absolute"], [style*="position: fixed"], [style*="position:fixed"]');
    for (const el of allPositioned) {
        const t = (el.textContent || '').trim();
        const s = getComputedStyle(el);
        const z = parseInt(s.zIndex) || 0;
        const r = el.getBoundingClientRect();
        // Skip tiny/invisible elements
        if (r.width < 30 || r.height < 30) continue;
        // Skip challenge-essential elements
        if (t.includes('Step') && t.includes('of 30')) continue;
        if (el.querySelector('input[type="text"]')) continue;
        if (t.includes('Iframe Challenge') || t.includes('Recursive Iframe')) continue;
        if (t.includes('Extract Code') || t.includes('Get Code')) continue;
        // Skip level navigation content
        if (t.includes('Enter Level') && el.querySelector('[class*="border-2"]')) continue;
        // This is a popup/overlay — try clicking close buttons
        const btns = el.querySelectorAll('button, [role="button"], .cursor-pointer');
        for (const btn of btns) {
            const bt = (btn.textContent || '').trim().toLowerCase();
            if (bt.includes('close') || bt.includes('dismiss') || bt === 'x'
                || bt === '\u00d7' || bt === '\u2715' || bt === '\u2716'
                || bt.includes('no thanks') || bt.includes('cancel')
                || bt.includes('got it') || bt.includes('ok')) {
                try { btn.click(); dismissed++; } catch(e) {}
            }
        }
        // CSS-hide it
        hide(el);
        dismissed++;
    }
    // Also hide ALL high-z-index elements that could block clicks
    document.querySelectorAll('*').forEach(el => {
        const s = getComputedStyle(el);
        const z = parseInt(s.zIndex) || 0;
        if (z >= 100 && (s.position === 'fixed' || s.position === 'absolute')) {
            const t = (el.textContent || '').trim();
            if (!t.includes('Step') && !t.includes('Extract') && !t.includes('Get Code')
                && !t.includes('Enter Code') && !t.includes('Submit')
                && !el.querySelector('input[type="text"]')) {
                hide(el);
            }
        }
    });
    // Hide floating decoy elements by text pattern
    document.querySelectorAll('div, a, span, button').forEach(el => {
        const s = getComputedStyle(el);
        if (s.position === 'absolute' || s.position === 'fixed') {
            const t = (el.textContent||'').trim();
            const decoyTexts = ['Click Here!','Try This!','Button!','Link!','Click Me!','Here!',
                'Moving!','Next Step','Continue','Move On','Go Forward','Keep Going',
                'Important Notice','amazing deals','won a prize','Congratulations',
                'Special Offer','Act Now','Limited Time','Subscribe'];
            for (const d of decoyTexts) {
                if (t.includes(d)) { hide(el); break; }
            }
        }
    });
    return dismissed;
})()""")
                if dismissed:
                    logger.info("Iframe: dismissed %d popups/overlays before Extract Code", dismissed)
                time.sleep(0.3)

                # Fix pointer-events on the Extract Code button's ancestor chain
                # Parent containers may have pointer-events:none that blocks clicks
                page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        const t = btn.textContent.trim().toLowerCase();
        if (!t.includes('extract') && !t.includes('get code')) continue;
        // Walk up the DOM tree and ensure pointer-events is enabled
        let el = btn;
        while (el && el !== document.body) {
            const s = getComputedStyle(el);
            if (s.pointerEvents === 'none') {
                el.style.pointerEvents = 'auto';
            }
            el = el.parentElement;
        }
        return true;
    }
    return false;
})()""")

                # Try clicking Extract Code with multiple methods:
                # 1) React fiber onClick invocation
                # 2) JS click + dispatchEvent
                # 3) Playwright locator click (native browser events)
                # 4) page.mouse.click at exact coordinates
                btn_rect = page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        if (!btn.offsetParent && getComputedStyle(btn).position !== 'fixed') continue;
        if (btn.disabled) continue;
        const t = btn.textContent.trim().toLowerCase();
        if (t.includes('extract') || t.includes('get code')) {
            btn.scrollIntoView({behavior: 'instant', block: 'center'});
            const r = btn.getBoundingClientRect();
            return {x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2),
                    w: r.width, h: r.height, text: btn.textContent.trim()};
        }
    }
    return null;
})()""")
                if btn_rect:
                    # Method 1: Direct React onClick handler invocation via fiber
                    # Walk up the fiber tree to find onClick (may be on a parent)
                    page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        const t = btn.textContent.trim().toLowerCase();
        if (!t.includes('extract') && !t.includes('get code')) continue;
        const syntheticEvent = {preventDefault:()=>{},stopPropagation:()=>{},
            target:btn, currentTarget:btn, nativeEvent:new MouseEvent('click')};
        // Try __reactProps$ first (direct props on DOM node)
        const pk = Object.keys(btn).find(k => k.startsWith('__reactProps$'));
        if (pk && typeof btn[pk]?.onClick === 'function') {
            try { btn[pk].onClick(syntheticEvent); } catch(e) {}
        }
        // Walk up the fiber tree to find onClick handler
        const fk = Object.keys(btn).find(k => k.startsWith('__reactFiber$'));
        if (fk) {
            let fiber = btn[fk];
            for (let i = 0; i < 15 && fiber; i++) {
                const props = fiber.memoizedProps || fiber.pendingProps;
                if (props && typeof props.onClick === 'function') {
                    try { props.onClick(syntheticEvent); } catch(e) {}
                    break;
                }
                fiber = fiber.return;
            }
        }
        // Also try standard click
        btn.click();
        // Also try dispatchEvent with full MouseEvent
        try {
            btn.dispatchEvent(new MouseEvent('click', {bubbles:true, cancelable:true, view:window}));
        } catch(e) {}
        return;
    }
})()""")
                    time.sleep(0.5)
                    try:
                        # Method 2: mouse click at coordinates
                        page.mouse.click(btn_rect["x"], btn_rect["y"])
                        log.append(f"iframe: clicked extract at ({btn_rect['x']},{btn_rect['y']})")
                    except Exception:
                        pass
                    # Method 3: Playwright locator
                    try:
                        loc = page.locator("button").filter(has_text="Extract Code")
                        if loc.count() > 0:
                            loc.first.click(timeout=2000, force=True)
                    except Exception:
                        pass
                else:
                    try:
                        loc = page.locator("button").filter(has_text="Extract Code")
                        if loc.count() > 0:
                            loc.first.scroll_into_view_if_needed(timeout=2000)
                            loc.first.click(timeout=2000, force=True)
                            log.append(f"iframe: force-clicked extract ({current}/{total})")
                        else:
                            loc2 = page.locator("button").filter(has_text="Get Code")
                            if loc2.count() > 0:
                                loc2.first.click(timeout=2000, force=True)
                    except Exception as e:
                        logger.debug("Iframe: extract click failed: %s", e)

                # Post-click analysis: check what changed
                time.sleep(1.0)
                post_click = page.evaluate("""\
(() => {
    const info = {};
    // Check the dashed-border container for any code
    for (const el of document.querySelectorAll('[class*="dashed"]')) {
        info.dashedContent = el.textContent.trim().substring(0, 200);
    }
    // Check deepest level div for new content
    for (const div of document.querySelectorAll('div[class*="border-2"][class*="rounded"]')) {
        const t = (div.childNodes[0]?.textContent || '').trim();
        if (t.includes('Level') && !t.includes('\u2713') && !t.includes('\u2714')) {
            info.levelHtml = div.innerHTML.substring(0, 500);
            info.levelComplete = false;
        }
        if (t.includes('Level') && (t.includes('\u2713') || t.includes('\u2714'))) {
            info.lastCompleteLevel = t.substring(0, 40);
        }
    }
    if (!info.hasOwnProperty('levelComplete')) info.levelComplete = true;
    // Check ALL elements with green/success styling for revealed codes
    info.greenText = [];
    for (const el of document.querySelectorAll('[class*="green"], [class*="emerald"], [class*="success"]')) {
        const t = (el.textContent||'').trim();
        if (t.length > 0 && t.length < 200) info.greenText.push(t.substring(0, 100));
    }
    // Check data attributes for hidden codes
    info.dataAttrs = [];
    document.querySelectorAll('[data-code], [data-secret], [data-value]').forEach(el => {
        for (const attr of ['data-code', 'data-secret', 'data-value']) {
            const v = el.getAttribute(attr);
            if (v) info.dataAttrs.push(v);
        }
    });
    // Check aria-labels
    document.querySelectorAll('[aria-label]').forEach(el => {
        const v = el.getAttribute('aria-label') || '';
        const m = v.match(/\\b[A-Z0-9]{6}\\b/);
        if (m) { if (!info.ariaCode) info.ariaCode = []; info.ariaCode.push(m[0]); }
    });
    // Check for ANY 6-char code anywhere
    const allText = document.body.textContent || '';
    const allCodes = [...(allText.match(/\\b[A-Z0-9]{6}\\b/g) || [])];
    const fp = new Set(['IFRAME','BWRONG','1WRONG','CWRONG','STEPGO','WORKER','LOADED','DWRONG','EWRONG','FWRONG','GWRONG','HWRONG','METHIS','REGIST','BUTTON','SUBMIT','TOGGLE']);
    info.allCodes = allCodes.filter(c => !fp.has(c) && !c.endsWith('WRONG')).slice(0, 10);
    // Check Extract Code button's React fiber for state
    for (const btn of document.querySelectorAll('button')) {
        if (!(btn.textContent||'').toLowerCase().includes('extract')) continue;
        const fiberKey = Object.keys(btn).find(k => k.startsWith('__reactFiber$'));
        if (!fiberKey) continue;
        let node = btn[fiberKey];
        info.fiberState = [];
        for (let i = 0; i < 15 && node; i++) {
            let hook = node.memoizedState;
            const hookVals = [];
            for (let h = 0; h < 10 && hook; h++) {
                const ms = hook.memoizedState;
                if (ms !== null && ms !== undefined) {
                    const type = typeof ms;
                    if (type === 'string' || type === 'number' || type === 'boolean') hookVals.push(ms);
                    else if (Array.isArray(ms)) hookVals.push('[array:' + ms.length + ']');
                    else if (type === 'object') { try { hookVals.push(JSON.stringify(ms).substring(0, 100)); } catch(e) { hookVals.push('[object]'); } }
                }
                hook = hook.next;
            }
            if (hookVals.length > 0) info.fiberState.push({depth: i, vals: hookVals});
            node = node.return;
        }
        break;
    }
    return info;
})()""")
                logger.info("Iframe: post-extract analysis (attempt %d): %s", extract_attempts, post_click)

                # Extract codes from fiberState if present
                if post_click and post_click.get("fiberState"):
                    import re as _re
                    _fp_set = {'IFRAME','BWRONG','1WRONG','CWRONG','STEPGO','WORKER','LOADED',
                               'DWRONG','EWRONG','FWRONG','GWRONG','HWRONG','METHIS','REGIST',
                               'BUTTON','SUBMIT','TOGGLE','PAGEGO'}
                    for fs_entry in post_click["fiberState"]:
                        for v in fs_entry.get("vals", []):
                            if isinstance(v, str) and _re.match(r'^[A-Z0-9]{6}$', v) and v not in _fp_set and not v.endswith('WRONG'):
                                logger.info("Iframe: found code %s in fiberState at depth %d", v, fs_entry.get("depth", -1))
                                codes.append(v)
                    if codes:
                        # Try direct submission from within the handler — the
                        # caller's fill_and_submit may fail due to overlays.
                        url_before = page.url
                        for fiber_code in codes:
                            submitted = page.evaluate(f"""\
(() => {{
    const code = '{fiber_code}';
    const inp = document.querySelector('input[placeholder*="code" i], input[type="text"]');
    if (!inp) return {{error: 'no input'}};
    // React-compatible value setting
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    setter.call(inp, code);
    inp.dispatchEvent(new Event('input', {{bubbles: true}}));
    inp.dispatchEvent(new Event('change', {{bubbles: true}}));
    // Try clicking Submit Code button
    let clicked = false;
    for (const btn of document.querySelectorAll('button')) {{
        const t = (btn.textContent || '').trim();
        if ((t === 'Submit' || t === 'Submit Code') && !btn.disabled) {{
            btn.click(); clicked = true; break;
        }}
    }}
    // Also check near the input
    if (!clicked) {{
        let container = inp.parentElement;
        for (let i = 0; i < 6 && container; i++) {{
            for (const btn of container.querySelectorAll('button')) {{
                const t = (btn.textContent || '').trim();
                if (t.includes('Submit') && !btn.disabled) {{
                    btn.click(); clicked = true; break;
                }}
            }}
            if (clicked) break;
            container = container.parentElement;
        }}
    }}
    return {{filled: true, clicked, val: inp.value, disabled: document.querySelector('button[disabled]')?.textContent?.trim()}};
}})()""")
                            logger.info("Iframe: direct submit result for '%s': %s", fiber_code, submitted)
                            time.sleep(0.5)
                            # Also try Enter key
                            try:
                                page.keyboard.press("Enter")
                            except Exception:
                                pass
                            time.sleep(1.0)
                            if page.url != url_before:
                                logger.info("Iframe: code '%s' accepted (URL changed)", fiber_code)
                                return HandlerResult(codes_found=codes, actions_log=log, success=True)
                        return HandlerResult(codes_found=codes, actions_log=log, success=True)

                # Take a screenshot on 3rd failed attempt for visual debugging
                if extract_attempts == 3:
                    try:
                        ss_path = f"/tmp/iframe_deepest_step{current}_{int(time.time())}.png"
                        page.screenshot(path=ss_path)
                        logger.info("Iframe: screenshot saved to %s", ss_path)
                    except Exception as e:
                        logger.debug("Iframe: screenshot failed: %s", e)

                time.sleep(0.5)
                continue
            logger.info("Iframe: nothing to click (round %d, %d/%d)", click_round, current, total)
            break

        # Final extraction: try high-confidence patterns, then all candidates
        # (Python-side is_false_positive filters noise)
        final_candidates = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    const results = [];
    const m = text.match(/(?:code|Code|extracted)[^:]*?:\\s*([A-Z0-9]{6})/);
    if (m) results.push(m[1]);
    for (const el of document.querySelectorAll('[class*="dashed"], [class*="green"], [class*="emerald"]')) {
        const t = (el.textContent || '').trim();
        const cm = t.match(/\\b([A-Z0-9]{6})\\b/); if (cm && t.length < 200) results.push(cm[1]);
    }
    const allCodes = text.match(/\\b[A-Z0-9]{6}\\b/g) || [];
    for (const c of allCodes) { if (!results.includes(c) && !c.endsWith('WRONG')) results.push(c); }
    return [...new Set(results)];
})()""") or []
        for fc in final_candidates:
            if not is_false_positive(fc):
                codes.append(fc)
                break  # Take the first non-FP code
        return HandlerResult(codes_found=codes, actions_log=log, needs_extraction=True)

    def handle_mutation(self, page) -> HandlerResult:
        codes: list[str] = []
        log: list[str] = []
        page.evaluate("""\
(() => {
    for (const el of document.querySelectorAll('div, h2, h3, p')) {
        const t = (el.textContent || '').substring(0, 200);
        if (t.includes('Mutation Challenge') || t.includes('Trigger Mutation')) {
            el.scrollIntoView({behavior: 'instant', block: 'center'}); break;
        }
    }
})()""")
        time.sleep(0.2)
        for _ in range(12):
            result = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    const codeMatch = text.match(/(?:code|Code)[^:]*:\\s*([A-Z0-9]{6})/);
    if (codeMatch) return {done: true, code: codeMatch[1]};
    let current = 0, total = 5;
    const m = text.match(/(\\d+)\\s*\\/\\s*(\\d+)\\s*(?:mutations?|triggered|complete)/i) ||
              text.match(/Mutations?[^:]*?:\\s*(\\d+)\\s*\\/\\s*(\\d+)/i) ||
              text.match(/triggered[^:]*?:\\s*(\\d+)\\s*\\/\\s*(\\d+)/i);
    if (m) { current = parseInt(m[1]); total = parseInt(m[2]); }
    return {current, total};
})()""")
            if not result:
                break
            if result.get("done") and result.get("code"):
                c = result["code"]
                if not is_false_positive(c):
                    codes.append(c)
                    return HandlerResult(codes_found=codes, actions_log=log, success=True)
            current = result.get("current", 0)
            total = result.get("total", 5)
            if current >= total:
                page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        const t = (btn.textContent || '').trim().toLowerCase();
        if (t.includes('complete') && !btn.disabled && btn.offsetParent) { btn.click(); return; }
    }
})()""")
                log.append(f"mutation: complete ({current}/{total})")
                time.sleep(0.5)
                continue
            page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        const t = (btn.textContent || '').trim().toLowerCase();
        if (t.includes('trigger') && !btn.disabled && btn.offsetParent) { btn.click(); return; }
    }
})()""")
            log.append(f"mutation: trigger ({current}/{total})")
            time.sleep(0.3)

        final = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    const m = text.match(/(?:code|Code)[^:]*:\\s*([A-Z0-9]{6})/); if (m) return m[1];
    for (const el of document.querySelectorAll('.text-green-600, .bg-green-100, .bg-rose-100, [class*="success"]')) {
        const cm = (el.textContent || '').match(/\\b([A-Z0-9]{6})\\b/); if (cm) return cm[1];
    }
    return null;
})()""")
        if final and not is_false_positive(final):
            codes.append(final)
        return HandlerResult(codes_found=codes, actions_log=log, needs_extraction=True)

    def handle_scroll_to_find(self, page, known_codes: list[str],
                              step_number: int = 0) -> HandlerResult:
        """Handle scroll-to-find challenges with progress checking after every action."""
        log: list[str] = []
        sorted_codes = sort_codes_by_priority(known_codes) if known_codes else []
        SCROLL_TIMEOUT = 25
        scroll_start = time.time()

        def _expired():
            return time.time() - scroll_start > SCROLL_TIMEOUT

        def _progress():
            return check_progress(page.url, step_number) if step_number else False

        if sorted_codes:
            _fill_code_in_input(page, sorted_codes[0])

        # Phase 0: mouse.wheel scroll + progress checking at each position
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.2)
        prev_scroll_y = 0
        start = time.time()
        phase0_codes: set[str] = set()
        while time.time() - start < 10:
            page.mouse.wheel(0, 800)
            time.sleep(0.10)
            if _progress():
                log.append("scroll: phase 0 auto-nav")
                return HandlerResult(actions_log=log, success=True)
            # Accumulate codes during scroll (virtualized content)
            try:
                vp_codes = page.evaluate("""() => {
                    const text = document.body.innerText || '';
                    return (text.match(/\\b[A-Z0-9]{6}\\b/g) || []).filter((v,i,a) => a.indexOf(v) === i);
                }""")
                phase0_codes.update(vp_codes or [])
            except Exception:
                pass
            cur_y = page.evaluate("() => window.scrollY")
            if cur_y <= prev_scroll_y and prev_scroll_y > 100:
                for _ in range(5):
                    page.mouse.wheel(0, 800)
                    time.sleep(0.15)
                    if _progress():
                        log.append("scroll: phase 0 auto-nav at bottom")
                        return HandlerResult(actions_log=log, success=True)
                break
            prev_scroll_y = cur_y

        # Try accumulated codes from Phase 0
        p0_new = [c for c in phase0_codes if not is_false_positive(c)
                  and c not in _LATIN_FP and not c.isdigit()
                  and c not in (known_codes or [])
                  and not re.match(r'^\d+(?:PX|VH|VW|EM|REM|MS|FR)$', c)]
        if p0_new:
            logger.info("Scroll phase 0 accumulated codes: %s", p0_new[:10])
            for code in sort_codes_by_priority(p0_new)[:5]:
                ok, _ = fill_and_submit(page, code, step_number)
                if ok:
                    return HandlerResult(actions_log=log, success=True, codes_found=[code])

        # Phase 0-deep: React + CSS + shadow DOM extraction
        if not _expired():
            deep_codes = deep_code_extraction(page, set(known_codes or []))
            if deep_codes:
                logger.info("Scroll phase 0-deep: %s", deep_codes[:8])
                for code in deep_codes[:10]:
                    ok, _ = fill_and_submit(page, code, step_number)
                    if ok:
                        return HandlerResult(actions_log=log, success=True, codes_found=[code])

        # Phase 0-slow: incremental scroll with pauses
        if not _expired():
            if sorted_codes:
                _fill_code_in_input(page, sorted_codes[0])
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(0.2)
            total_h = page.evaluate("() => document.body.scrollHeight")
            slow_start = time.time()
            prev_y = 0
            for pos in range(0, total_h + 400, 400):
                if time.time() - slow_start > 8 or _expired():
                    break
                page.mouse.wheel(0, 400)
                time.sleep(0.25)
                if _progress():
                    log.append("scroll: phase 0-slow auto-nav")
                    return HandlerResult(actions_log=log, success=True)
                cur_y = page.evaluate("() => window.scrollY")
                if cur_y <= prev_y and prev_y > 100:
                    break
                prev_y = cur_y

        # Phase 0a: scrollable containers
        if not _expired():
            containers = page.evaluate("""\
(() => {
    const results = [];
    const els = document.querySelectorAll('div, section, main, article');
    for (const el of els) {
        if (el.closest('.fixed')) continue;
        const s = window.getComputedStyle(el);
        const overflow = s.overflow + s.overflowY;
        if (!(overflow.includes('auto') || overflow.includes('scroll'))) continue;
        if (el.scrollHeight <= el.clientHeight + 10) continue;
        const r = el.getBoundingClientRect();
        if (r.width < 100 || r.height < 100) continue;
        results.push({
            x: Math.round(r.x + r.width/2),
            y: Math.round(r.y + r.height/2),
            scrollable: el.scrollHeight - el.clientHeight
        });
    }
    return results;
})()""")
            for cont in (containers or [])[:3]:
                page.mouse.move(cont['x'], cont['y'])
                time.sleep(0.05)
                scroll_remaining = cont['scrollable']
                while scroll_remaining > 0:
                    page.mouse.wheel(0, 500)
                    scroll_remaining -= 500
                    time.sleep(0.10)
                    if _progress():
                        log.append("scroll: container scroll worked")
                        return HandlerResult(actions_log=log, success=True)

        # Phase 0b: keyboard scroll
        if not _expired():
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(0.1)
            page.keyboard.press("End")
            time.sleep(0.5)
            if _progress():
                log.append("scroll: End key worked")
                return HandlerResult(actions_log=log, success=True)
            for _ in range(80):
                if _expired():
                    break
                page.keyboard.press("PageDown")
                time.sleep(0.06)
                if _progress():
                    log.append("scroll: PageDown worked")
                    return HandlerResult(actions_log=log, success=True)

        # Phase 0c: synthetic events
        if not _expired():
            page.evaluate("""\
(() => {
    window.scrollTo(0, document.body.scrollHeight);
    for (const target of [window, document, document.documentElement, document.body]) {
        target.dispatchEvent(new Event('scroll', {bubbles: true}));
        target.dispatchEvent(new WheelEvent('wheel', {deltaY: 500, bubbles: true}));
    }
})()""")
            time.sleep(0.3)
            if _progress():
                log.append("scroll: synthetic events worked")
                return HandlerResult(actions_log=log, success=True)

        # Phase 1: scroll + click safe buttons with progress checks
        if not _expired():
            total_h = page.evaluate("() => document.body.scrollHeight")
            for pos in range(0, total_h + 800, 800):
                if _expired():
                    break
                page.evaluate(f"window.scrollTo(0, {pos})")
                time.sleep(0.05)
                btn_results = page.evaluate("""\
(() => {
    const SAFE_WORDS = ['next', 'submit', 'go', '\u2192', 'navigate', 'enter'];
    const btns = [...document.querySelectorAll('button, a')];
    const results = [];
    for (const btn of btns) {
        const rect = btn.getBoundingClientRect();
        if (rect.top < -10 || rect.top > window.innerHeight + 10) continue;
        if (!btn.offsetParent || btn.disabled) continue;
        if (btn.closest('.fixed')) continue;
        const t = (btn.textContent || '').trim();
        const tl = t.toLowerCase();
        if (tl.length > 40 || tl.length === 0) continue;
        if (t === '\u00d7' || t === 'X' || t === '\u2715') continue;
        if (SAFE_WORDS.some(w => tl === w || (tl.includes(w) && tl.length < 15))) {
            results.push({text: t, idx: btns.indexOf(btn)});
        }
    }
    return results;
})()""")
                for btn in btn_results:
                    clear_popups(page)
                    page.evaluate(f"(idx) => [...document.querySelectorAll('button, a')][idx]?.click()", btn["idx"])
                    time.sleep(0.1)
                    if _progress():
                        log.append(f"scroll: safe button '{btn['text']}' worked")
                        return HandlerResult(actions_log=log, success=True)

        # Phase 2: outlier buttons
        if not _expired():
            outlier_result = page.evaluate("""\
(() => {
    const btns = [...document.querySelectorAll('button')].filter(b => {
        if (!b.offsetParent || b.disabled || b.closest('.fixed')) return false;
        const t = b.textContent.trim();
        return t.length > 0 && t.length < 40 && t !== '\u00d7' && t !== 'X' && t !== '\u2715';
    });
    if (btns.length < 5) return null;
    const freq = {};
    btns.forEach(b => { const label = b.textContent.trim().toLowerCase(); freq[label] = (freq[label] || 0) + 1; });
    const outliers = btns.filter(b => { const label = b.textContent.trim().toLowerCase(); return freq[label] <= 2; });
    return outliers.map((b, i) => ({text: b.textContent.trim(), idx: [...document.querySelectorAll('button')].indexOf(b)}));
})()""")
            if outlier_result:
                for btn in outlier_result:
                    clear_popups(page)
                    page.evaluate(f"""\
(idx) => {{
    const btn = document.querySelectorAll('button')[idx];
    if (btn) {{ btn.scrollIntoView({{behavior: 'instant', block: 'center'}}); btn.click(); }}
}}""", btn["idx"])
                    time.sleep(0.12)
                    if _progress():
                        log.append(f"scroll: outlier '{btn['text']}' worked")
                        return HandlerResult(actions_log=log, success=True)

        # Phase 3: full page button scan
        if not _expired():
            if sorted_codes:
                _fill_code_in_input(page, sorted_codes[0])
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(0.05)
            total_h = page.evaluate("() => document.body.scrollHeight")
            for pos in range(0, total_h + 1000, 1000):
                page.mouse.wheel(0, 1000)
                time.sleep(0.05)

            all_btns = page.evaluate("""\
(() => {
    const btns = [...document.querySelectorAll('button, a')].filter(el => {
        if (el.disabled || el.closest('.fixed')) return false;
        const t = (el.textContent || '').trim();
        return t.length > 0 && t.length < 40 && t !== '\u00d7' && t !== 'X' && t !== '\u2715';
    });
    return btns.map((b, i) => ({text: b.textContent.trim(), idx: i}));
})()""")
            batch_size = 5
            for start_idx in range(0, len(all_btns), batch_size):
                if _expired():
                    break
                end_idx = min(start_idx + batch_size, len(all_btns))
                page.evaluate(f"""\
(() => {{
    const start = {start_idx}, end = {end_idx};
    const clearP = () => {{
        document.querySelectorAll('.fixed').forEach(el => {{
            const text = el.textContent || '';
            if (text.includes('Wrong Button') || text.includes('Try Again') ||
                text.includes('another way') || text.includes('fake') ||
                text.includes('won a prize') || text.includes('popup message') ||
                text.includes('Click the button to dismiss')) {{
                const btn = el.querySelector('button');
                if (btn) btn.click();
                el.style.display = 'none';
                el.style.pointerEvents = 'none';
            }}
        }});
    }};
    const allBtns = [...document.querySelectorAll('button, a')].filter(el => {{
        if (el.disabled || el.closest('.fixed')) return false;
        const t = (el.textContent || '').trim();
        return t.length > 0 && t.length < 40 && t !== '\u00d7' && t !== 'X' && t !== '\u2715';
    }});
    for (let i = start; i < Math.min(end, allBtns.length); i++) {{
        clearP();
        allBtns[i].scrollIntoView({{behavior: 'instant', block: 'center'}});
        allBtns[i].click();
    }}
}})()""")
                time.sleep(0.1)
                if _progress():
                    log.append("scroll: phase 3 batch worked")
                    return HandlerResult(actions_log=log, success=True)

        # Phase 4: Playwright mouse click through visible buttons during scroll
        if not _expired():
            if sorted_codes:
                _fill_code_in_input(page, sorted_codes[0])
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(0.1)
            total_h = page.evaluate("() => document.body.scrollHeight")
            scroll_pos = 0
            phase4_start = time.time()
            while scroll_pos < total_h + 800:
                if time.time() - phase4_start > 12 or _expired():
                    break
                page.mouse.wheel(0, 800)
                scroll_pos += 800
                time.sleep(0.1)
                visible_btns = page.evaluate("""\
(() => {
    const sel = 'button, a, [role="button"], [class*="cursor-pointer"], [onclick]';
    const els = [...document.querySelectorAll(sel)].filter(el => {
        if (el.closest('.fixed') || el.disabled) return false;
        const rect = el.getBoundingClientRect();
        if (rect.top < -10 || rect.top > window.innerHeight + 10) return false;
        if (rect.width < 10 || rect.height < 10) return false;
        const t = (el.textContent || '').trim();
        if (t.length === 0 || t.length > 60) return false;
        if (t === '\u00d7' || t === 'X' || t === '\u2715') return false;
        return true;
    });
    return els.map(el => {
        const rect = el.getBoundingClientRect();
        return { text: (el.textContent || '').trim().substring(0, 40),
                 x: Math.round(rect.x + rect.width / 2),
                 y: Math.round(rect.y + rect.height / 2) };
    });
})()""")
                for btn in (visible_btns or []):
                    clear_popups(page)
                    try:
                        page.mouse.click(btn['x'], btn['y'])
                        time.sleep(0.05)
                    except Exception:
                        pass
                if _progress():
                    log.append(f"scroll: phase 4 worked at scroll {scroll_pos}px")
                    return HandlerResult(actions_log=log, success=True)

        # Phase 5: React onClick elements (divs, spans that aren't buttons)
        if not _expired():
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(0.05)
            total_h = page.evaluate("() => document.body.scrollHeight")
            for scroll_pos in range(0, total_h + 1200, 1200):
                if _expired():
                    break
                page.mouse.wheel(0, 1200)
                time.sleep(0.08)
                react_btns = page.evaluate("""\
(() => {
    const results = [];
    const els = document.querySelectorAll('div, span, p, li, td, section');
    for (const el of els) {
        if (el.closest('.fixed')) continue;
        const rect = el.getBoundingClientRect();
        if (rect.top < -10 || rect.top > window.innerHeight + 10) continue;
        if (rect.width < 20 || rect.height < 15) continue;
        const propsKey = Object.keys(el).find(k => k.startsWith('__reactProps$'));
        if (propsKey && el[propsKey] && el[propsKey].onClick) {
            const t = (el.textContent || '').trim();
            if (t === '\u00d7' || t === 'X' || t.length > 60) continue;
            if (el.querySelector('button, a')) continue;
            results.push({ x: Math.round(rect.x + rect.width / 2),
                           y: Math.round(rect.y + rect.height / 2) });
        }
    }
    return results;
})()""")
                for btn in (react_btns or []):
                    clear_popups(page)
                    try:
                        page.mouse.click(btn['x'], btn['y'])
                    except Exception:
                        pass
                if _progress():
                    log.append("scroll: React onClick worked")
                    return HandlerResult(actions_log=log, success=True)

        page.evaluate("window.scrollTo(0, 0)")
        log.append("scroll: all phases complete, no progress")
        return HandlerResult(actions_log=log, needs_extraction=True)

    def handle_hover_reveal(self, page) -> HandlerResult:
        target = page.evaluate("""\
(() => {
    const decoys = ['Click Me!', 'Button!', 'Link!', 'Here!', 'Click Here!', 'Try This!'];
    document.querySelectorAll('div, button, span').forEach(el => {
        const style = getComputedStyle(el);
        if ((style.position === 'absolute' || style.position === 'fixed') && decoys.includes(el.textContent.trim())) {
            el.style.display = 'none';
        }
    });
    const candidates = [...document.querySelectorAll('[class*="cursor-pointer"]')].filter(el =>
        el.offsetParent && el.offsetWidth > 50 && el.offsetHeight > 30 &&
        !el.closest('.fixed:not(:has(input[type="text"]))'));
    if (candidates.length === 0) {
        const bordered = [...document.querySelectorAll('div')].filter(el => {
            const cls = el.getAttribute('class') || '';
            return cls.includes('border-2') && cls.includes('rounded') && el.offsetParent && el.offsetWidth > 50;
        });
        if (bordered.length > 0) candidates.push(...bordered);
    }
    if (candidates.length === 0) return null;
    const el = candidates[0];
    el.scrollIntoView({behavior: 'instant', block: 'center'});
    const rect = el.getBoundingClientRect();
    const opts = {bubbles: true, clientX: rect.x + rect.width/2, clientY: rect.y + rect.height/2};
    el.dispatchEvent(new MouseEvent('mouseenter', opts));
    el.dispatchEvent(new MouseEvent('mouseover', opts));
    return {x: rect.x + rect.width/2, y: rect.y + rect.height/2};
})()""")
        if target:
            page.mouse.move(target["x"], target["y"])
            time.sleep(1.5)
            return HandlerResult(actions_log=["hover: hovered 1.5s"], needs_extraction=True)
        return HandlerResult(actions_log=["hover: no target found"], needs_extraction=True)

    def handle_delayed_reveal(self, page) -> HandlerResult:
        codes: list[str] = []
        for i in range(25):
            result = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    const codeMatch = text.match(/(?:code|Code|challenge code)[^:]*:\\s*([A-Z0-9]{6})/i);
    if (codeMatch) return {code: codeMatch[1]};
    for (const el of document.querySelectorAll('.text-green-600, .bg-green-100, .bg-blue-100, .bg-purple-100')) {
        const m = (el.textContent || '').match(/\\b([A-Z0-9]{6})\\b/); if (m) return {code: m[1]};
    }
    const timerMatch = text.match(/(\\d+\\.?\\d*)\\s*s(?:econds?)?\\s*remaining/i);
    const remaining = timerMatch ? parseFloat(timerMatch[1]) : null;
    const done = text.includes('revealed') || text.includes('Complete') || text.includes('100%');
    return {remaining, done};
})()""")
            if result and result.get("code"):
                c = result["code"]
                if not is_false_positive(c):
                    codes.append(c)
                    return HandlerResult(codes_found=codes, actions_log=["delayed: code found"])
            if result and result.get("done"):
                time.sleep(0.3)
                continue
            if result and result.get("remaining") is not None and result["remaining"] < 0.5:
                time.sleep(0.6)
                continue
            time.sleep(0.4)

        final = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    const matches = text.match(/\\b([A-Z0-9]{6})\\b/g) || [];
    const known = new Set(['FILLER', 'SUBMIT', 'BUTTON', 'SCROLL', 'REVEAL']);
    return matches.find(m => !known.has(m) && /[0-9]/.test(m)) || null;
})()""")
        if final and not is_false_positive(final):
            codes.append(final)
        return HandlerResult(codes_found=codes, actions_log=["delayed: waited"], needs_extraction=True)

    def handle_drag_and_drop(self, page) -> HandlerResult:
        log: list[str] = []
        # JS DragEvent dispatch
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
        log.append("drag: JS events dispatched")

        # Mouse fallback
        fill_count = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    const match = text.match(/(\\d+)\\/(\\d+)\\s*filled/);
    return match ? parseInt(match[1]) : -1;
})()""")
        if fill_count >= 0 and fill_count < 6:
            log.append("drag: mouse fallback")
            for round_num in range(6):
                state = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    const match = text.match(/(\\d+)\\/(\\d+)\\s*filled/);
    const filled = match ? parseInt(match[1]) : 0;
    if (filled >= 6) return {filled, done: true};
    const emptySlots = [...document.querySelectorAll('div')].filter(el => {
        const t = (el.textContent || '').trim();
        return t.match(/^Slot \\d+$/) &&
               ((el.getAttribute('class') || '').includes('dashed') || (el.getAttribute('style') || '').includes('dashed'));
    }).map(el => { el.scrollIntoView({behavior: 'instant', block: 'center'}); const rect = el.getBoundingClientRect();
        return {x: rect.x + rect.width/2, y: rect.y + rect.height/2}; });
    const dropZones = [...document.querySelectorAll('[class*="border-dashed"]')];
    const dropZoneSet = new Set(dropZones);
    const pieces = [...document.querySelectorAll('[draggable="true"]')].filter(el => {
        let parent = el.parentElement;
        while (parent) { if (dropZoneSet.has(parent)) return false; parent = parent.parentElement; }
        return el.offsetParent !== null;
    }).map(el => { const rect = el.getBoundingClientRect();
        return {x: rect.x + rect.width/2, y: rect.y + rect.height/2, text: el.textContent.trim()}; });
    return {filled, done: false, emptySlots, pieces: pieces.slice(0, 6)};
})()""")
                if state.get("done") or state.get("filled", 0) >= 6:
                    page.evaluate("""\
(() => { document.querySelectorAll('button').forEach(btn => {
    const t = (btn.textContent || '').trim().toLowerCase();
    if ((t.includes('complete') || t.includes('done') || t.includes('verify')) &&
        !t.includes('clear') && btn.offsetParent && !btn.disabled) btn.click();
}); })()""")
                    break
                slots = state.get("emptySlots", [])
                pieces = state.get("pieces", [])
                if not slots or not pieces:
                    break
                piece = pieces[0]
                slot = slots[0]
                page.mouse.move(piece["x"], piece["y"])
                page.mouse.down()
                time.sleep(0.05)
                page.mouse.move(slot["x"], slot["y"], steps=15)
                time.sleep(0.05)
                page.mouse.up()
                time.sleep(0.3)

        return HandlerResult(actions_log=log, needs_extraction=True)

    def handle_keyboard_sequence(self, page) -> HandlerResult:
        html_text = page.evaluate("() => document.body.textContent || ''")
        keys = re.findall(r"((?:Control|Shift|Alt|Meta)\+[A-Za-z0-9])", html_text)
        seen: set[str] = set()
        unique_keys: list[str] = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                unique_keys.append(k)
        if not unique_keys:
            return HandlerResult(actions_log=["keyboard: no keys found"])
        page.evaluate("() => document.body.focus()")
        for k in unique_keys:
            page.keyboard.press(k)
            time.sleep(0.3)
        return HandlerResult(actions_log=[f"keyboard: pressed {unique_keys}"], needs_extraction=True)

    def handle_radio_brute_force(self, page, step: int) -> HandlerResult:
        # Scroll modal containers
        page.evaluate("""\
(() => {
    document.querySelectorAll('[class*="overflow-y"], [class*="overflow-auto"], [class*="max-h"]').forEach(el => {
        if (el.scrollHeight > el.clientHeight) el.scrollTop = el.scrollHeight;
    });
    document.querySelectorAll('.fixed').forEach(modal => {
        modal.querySelectorAll('*').forEach(el => {
            if (el.scrollHeight > el.clientHeight + 10) el.scrollTop = el.scrollHeight;
        });
    });
})()""")
        time.sleep(0.2)

        # Mouse wheel scroll inside modal
        modal_center = page.evaluate("""\
(() => {
    const modal = [...document.querySelectorAll('.fixed')].find(el =>
        el.textContent.includes('Please Select') || el.textContent.includes('Submit & Continue'));
    if (!modal) return null;
    const rect = modal.getBoundingClientRect();
    return {x: rect.x + rect.width/2, y: rect.y + rect.height/2};
})()""")
        if modal_center:
            page.mouse.move(modal_center["x"], modal_center["y"])
            for _ in range(5):
                page.mouse.wheel(0, 500)
                time.sleep(0.05)
            time.sleep(0.1)

        count = page.evaluate("""\
(() => {
    let opts = document.querySelectorAll('input[type="radio"]');
    if (opts.length > 0) return {count: opts.length, type: 'native'};
    opts = document.querySelectorAll('[role="radio"]');
    if (opts.length > 0) return {count: opts.length, type: 'role'};
    const submitBtn = [...document.querySelectorAll('button')].find(b =>
        b.textContent.includes('Submit & Continue') || b.textContent.includes('Submit and Continue'));
    if (!submitBtn) return {count: 0, type: 'none'};
    let modal = submitBtn.parentElement;
    while (modal && modal !== document.body) {
        if (modal.querySelector('[class*="overflow"]') || modal.querySelector('[class*="max-h"]') ||
            modal.classList.contains('fixed')) break;
        modal = modal.parentElement;
    }
    if (!modal || modal === document.body) modal = submitBtn.closest('div[class*="bg-white"], div[class*="rounded"]');
    if (!modal) return {count: 0, type: 'none'};
    const cards = [...modal.querySelectorAll('[class*="cursor-pointer"], [class*="border"][class*="rounded"]')].filter(el => {
        const text = el.textContent.trim();
        return text.length > 0 && text.length < 80 && !text.includes('Submit') && !text.includes('Section');
    });
    return {count: cards.length, type: 'custom'};
})()""")

        radio_count = count.get("count", 0) if isinstance(count, dict) else 0
        if radio_count == 0:
            has_text = page.evaluate("""\
(() => {
    const text = (document.body.textContent || '').toLowerCase();
    return text.includes('please select an option') && text.includes('submit');
})()""")
            if has_text:
                radio_count = page.evaluate("""\
(() => {
    const modal = [...document.querySelectorAll('.fixed')].find(el =>
        el.textContent.includes('Please Select') || el.textContent.includes('Submit & Continue'));
    if (!modal) return 0;
    return [...modal.querySelectorAll('div[class*="border"], div[class*="cursor"], label')].filter(el => {
        const t = el.textContent.trim();
        return t.length > 0 && t.length < 80 && !t.includes('Submit') && el.offsetParent;
    }).length;
})()""")

        if radio_count == 0:
            return HandlerResult(actions_log=["radio: no options found"])

        effective_count = min(radio_count, 15)
        for i in range(effective_count):
            page.evaluate("""\
(idx) => {
    let options = [...document.querySelectorAll('input[type="radio"]')];
    if (options.length === 0) options = [...document.querySelectorAll('[role="radio"]')];
    if (options.length === 0) {
        const submitBtn = [...document.querySelectorAll('button')].find(b =>
            b.textContent.includes('Submit & Continue') || b.textContent.includes('Submit and Continue'));
        if (!submitBtn) return;
        let modal = submitBtn.parentElement;
        while (modal && modal !== document.body) {
            if (modal.querySelector('[class*="overflow"]') || modal.classList.contains('fixed')) break;
            modal = modal.parentElement;
        }
        if (!modal || modal === document.body) modal = submitBtn.closest('.fixed') || submitBtn.closest('div[class*="bg-white"]');
        if (modal) {
            options = [...modal.querySelectorAll('[class*="cursor-pointer"], [class*="border"][class*="rounded"], label')].filter(el => {
                const t = el.textContent.trim();
                return t.length > 0 && t.length < 80 && !t.includes('Submit') && !t.includes('Section');
            });
        }
    }
    const opt = options[idx]; if (!opt) return;
    opt.click();
    const innerRadio = opt.querySelector('input[type="radio"]'); if (innerRadio) innerRadio.click();
    const card = opt.closest('label, [class*="cursor-pointer"]'); if (card && card !== opt) card.click();
    const sub = [...document.querySelectorAll('button')].find(b => b.textContent.includes('Submit'));
    if (sub) sub.click();
}""", i)
            time.sleep(0.15)
            if check_progress(page.url, step):
                return HandlerResult(actions_log=[f"radio: option {i+1}/{radio_count} correct"], success=True)

        _hide_stuck_modals(page)
        return HandlerResult(actions_log=[f"radio: all {effective_count} options tried"])

    def handle_timing_capture(self, page) -> HandlerResult:
        for _ in range(5):
            page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        const t = (btn.textContent || '').trim().toLowerCase();
        if (t.includes('capture') && btn.offsetParent && !btn.disabled) { btn.click(); return true; }
    }
    return false;
})()""")
            time.sleep(1.0)
        return HandlerResult(actions_log=["timing: clicked capture"], needs_extraction=True)

    def handle_split_parts(self, page) -> HandlerResult:
        log: list[str] = []
        for _ in range(10):
            result = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    const foundMatch = text.match(/(\\d+)\\/(\\d+)\\s*found/);
    const found = foundMatch ? parseInt(foundMatch[1]) : 0;
    const total = foundMatch ? parseInt(foundMatch[2]) : 4;
    if (found >= total) return {found, total, clicked: 0, done: true};
    let clicked = 0;
    document.querySelectorAll('div').forEach(el => {
        const style = getComputedStyle(el);
        const cls = el.getAttribute('class') || '';
        const elText = (el.textContent || '').trim();
        if (!(style.position === 'absolute' || cls.includes('absolute'))) return;
        if (!elText.match(/Part\\s*\\d/i)) return;
        if (el.offsetWidth < 10) return;
        const bg = style.backgroundColor;
        if (bg.includes('134') || bg.includes('green') || cls.includes('bg-green')) return;
        el.scrollIntoView({behavior: 'instant', block: 'center'});
        el.click(); clicked++;
    });
    return {found, total, clicked, done: false};
})()""")
            if result.get("done"):
                log.append("split_parts: all collected")
                break
            if result.get("clicked", 0) == 0:
                page.evaluate("() => window.scrollBy(0, 400)")
            time.sleep(0.5)
        return HandlerResult(actions_log=log, needs_extraction=True)

    def handle_rotating_code(self, page) -> HandlerResult:
        for _ in range(15):
            state = page.evaluate("""\
(() => {
    const btns = [...document.querySelectorAll('button')];
    let done = 0, required = 3;
    for (const btn of btns) {
        const t = (btn.textContent || '').trim();
        const m = t.match(/[Cc]apture.*?(\\d+)\\/(\\d+)/);
        if (m) { done = parseInt(m[1]); required = parseInt(m[2]); break; }
    }
    return {done, required, complete: done >= required};
})()""")
            if state.get("complete"):
                return HandlerResult(actions_log=["rotating: complete"], needs_extraction=True)
            clicked = page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        const t = (btn.textContent || '').trim().toLowerCase();
        if (t.includes('capture') && btn.offsetParent && !btn.disabled) { btn.click(); return true; }
    }
    return false;
})()""")
            if not clicked:
                break
            time.sleep(1.0)
        return HandlerResult(actions_log=["rotating: done"], needs_extraction=True)

    def handle_multi_tab(self, page) -> HandlerResult:
        for _ in range(3):
            result = page.evaluate("""\
(() => {
    const btns = [...document.querySelectorAll('button')];
    const tabBtns = btns.filter(b => {
        const t = (b.textContent || '').trim().toLowerCase();
        return (t.includes('tab') || t.match(/^\\d+$/)) && b.offsetParent;
    });
    for (const btn of tabBtns) btn.click();
    return tabBtns.length;
})()""")
            time.sleep(0.5)
        return HandlerResult(actions_log=["multi_tab: clicked tabs"], needs_extraction=True)

    def handle_sequence(self, page) -> HandlerResult:
        log: list[str] = []
        # Click "Click Me"
        page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        const t = (btn.textContent || '').trim().toLowerCase();
        if (t.includes('click me') && btn.offsetParent && !btn.disabled) { btn.click(); return; }
    }
})()""")
        time.sleep(0.3)
        log.append("sequence: clicked")

        # Hover
        hover_info = page.evaluate("""\
(() => {
    const els = [...document.querySelectorAll('div, span, p')];
    let best = null;
    for (const el of els) {
        const t = (el.textContent || '').trim().toLowerCase();
        if ((t === 'hover over this area' || t.includes('hover over')) && el.offsetParent) {
            if (!best || el.textContent.length < best.textContent.length) best = el;
        }
    }
    if (best) {
        best.scrollIntoView({behavior: 'instant', block: 'center'});
        const rect = best.getBoundingClientRect();
        return {x: rect.x + rect.width/2, y: rect.y + rect.height/2};
    }
    return null;
})()""")
        if hover_info:
            page.mouse.move(hover_info["x"], hover_info["y"])
            time.sleep(0.5)
            page.evaluate(f"""\
(() => {{
    const el = document.elementFromPoint({hover_info['x']}, {hover_info['y']});
    if (el) {{
        el.dispatchEvent(new MouseEvent('mouseenter', {{bubbles: true}}));
        el.dispatchEvent(new MouseEvent('mouseover', {{bubbles: true}}));
    }}
}})()""")
            time.sleep(0.8)
        log.append("sequence: hovered")

        # Type
        page.evaluate("""\
(() => {
    const inputs = [...document.querySelectorAll('input[type="text"], input:not([type]), textarea')];
    const inp = inputs.find(i => {
        const ph = (i.placeholder || '').toLowerCase();
        return !ph.includes('code') && i.offsetParent && i.type !== 'number' && i.type !== 'hidden';
    });
    if (inp) {
        inp.scrollIntoView({behavior: 'instant', block: 'center'}); inp.focus();
        const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
        s.call(inp, 'hello world');
        inp.dispatchEvent(new Event('input', {bubbles: true}));
        inp.dispatchEvent(new Event('change', {bubbles: true}));
    }
})()""")
        time.sleep(0.3)
        log.append("sequence: typed")

        # Scroll box
        scroll_info = page.evaluate("""\
(() => {
    for (const el of document.querySelectorAll('div, textarea')) {
        const style = getComputedStyle(el);
        const isScrollable = style.overflow === 'auto' || style.overflow === 'scroll' ||
            style.overflowY === 'auto' || style.overflowY === 'scroll';
        if (isScrollable && el.scrollHeight > el.clientHeight + 10 &&
            el.offsetParent && el.clientHeight < 400 && el.clientHeight > 30) {
            el.scrollIntoView({behavior: 'instant', block: 'center'});
            el.scrollTop = el.scrollHeight;
            const rect = el.getBoundingClientRect();
            return {x: rect.x + rect.width/2, y: rect.y + rect.height/2};
        }
    }
    return null;
})()""")
        if scroll_info:
            page.mouse.move(scroll_info["x"], scroll_info["y"])
            page.mouse.wheel(0, 300)
        time.sleep(0.3)
        log.append("sequence: scrolled")

        # Click Complete
        page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        const t = (btn.textContent || '').trim().toLowerCase();
        if (t.includes('complete') && btn.offsetParent && !btn.disabled) { btn.click(); return; }
    }
})()""")
        time.sleep(0.5)
        return HandlerResult(actions_log=log, needs_extraction=True)

    def handle_video_frames(self, page) -> HandlerResult:
        state = page.evaluate("""\
(() => {
    const text = document.body.textContent || '';
    const targetMatch = text.match(/(?:frame|Frame)\\s+(\\d+)/g);
    let targetFrame = null;
    if (targetMatch) {
        for (const m of targetMatch) {
            const num = parseInt(m.match(/\\d+/)[0]);
            if (num > 0 && num < 100) { targetFrame = num; break; }
        }
    }
    const currentMatch = text.match(/Frame\\s+(\\d+)\\/(\\d+)/);
    const currentFrame = currentMatch ? parseInt(currentMatch[1]) : 0;
    return {targetFrame, currentFrame};
})()""")
        target = state.get("targetFrame")
        if target is None:
            return HandlerResult(actions_log=["video: no target frame"])

        for _ in range(5):
            page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        if (btn.textContent.trim() === '+1' && btn.offsetParent) { btn.click(); return; }
    }
})()""")
            time.sleep(0.3)

        for _ in range(20):
            current = page.evaluate("""\
(() => {
    const m = (document.body.textContent || '').match(/Frame\\s+(\\d+)\\//);
    return m ? parseInt(m[1]) : 0;
})()""")
            if current == target:
                break
            diff = target - current
            btn_text = "+10" if diff >= 10 else "-10" if diff <= -10 else "+1" if diff > 0 else "-1"
            page.evaluate(f"""\
(() => {{
    for (const btn of document.querySelectorAll('button')) {{
        if (btn.textContent.trim() === '{btn_text}' && btn.offsetParent) {{ btn.click(); return; }}
    }}
}})()""")
            time.sleep(0.2)

        time.sleep(0.5)
        page.evaluate("""\
(() => {
    for (const btn of document.querySelectorAll('button')) {
        const t = (btn.textContent || '').trim().toLowerCase();
        if ((t.includes('complete') || t.includes('done') || t.includes('reveal')) &&
            btn.offsetParent && !btn.disabled) { btn.click(); return; }
    }
})()""")
        time.sleep(0.5)
        return HandlerResult(actions_log=[f"video: navigated to frame {target}"], needs_extraction=True)

    def handle_animated_button(self, page, codes: list[str], step: int) -> HandlerResult:
        log: list[str] = []
        for code in codes[:10]:
            if _try_animated_button_submit(page, code, step):
                return HandlerResult(codes_found=[code], actions_log=["animated: worked"], success=True)
        log.append("animated: no code worked")
        return HandlerResult(actions_log=log)

    def handle_dom_extraction(self, page, failed_codes: list[str]) -> HandlerResult:
        html = page.content()
        codes = extract_hidden_codes(html)
        fresh = [c for c in codes if c not in failed_codes]
        deep = deep_code_extraction(page, set(failed_codes))
        all_codes = list(dict.fromkeys(fresh + deep))
        return HandlerResult(codes_found=all_codes, actions_log=["dom: extracted"], needs_extraction=False)
