"""Detect which of 21 challenge types is on the current page.

Uses a single page.evaluate() call to collect DOM signals, then maps
them to ranked ChallengeType results in Python.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ChallengeType(Enum):
    DOM_EXTRACTION = "dom_extraction"
    MATH_PUZZLE = "math_puzzle"
    KEYBOARD_SEQUENCE = "keyboard_sequence"
    RADIO_BRUTE_FORCE = "radio_brute_force"
    SCROLL_TO_FIND = "scroll_to_find"
    HOVER_REVEAL = "hover_reveal"
    TIMING_CAPTURE = "timing_capture"
    MULTI_TAB = "multi_tab"
    SEQUENCE_CHALLENGE = "sequence_challenge"
    VIDEO_FRAMES = "video_frames"
    SPLIT_PARTS = "split_parts"
    ROTATING_CODE = "rotating_code"
    DELAYED_REVEAL = "delayed_reveal"
    DRAG_AND_DROP = "drag_and_drop"
    ANIMATED_BUTTON = "animated_button"
    SHADOW_DOM = "shadow_dom"
    CANVAS_DRAW = "canvas_draw"
    AUDIO_CHALLENGE = "audio_challenge"
    WEBSOCKET = "websocket"
    SERVICE_WORKER = "service_worker"
    IFRAME_RECURSIVE = "iframe_recursive"
    MUTATION_OBSERVER = "mutation_observer"
    UNKNOWN = "unknown"


@dataclass
class DetectionResult:
    challenge_type: ChallengeType
    confidence: float
    signals: dict[str, bool] = field(default_factory=dict)


# Single JS snippet that collects all detection signals from the page.
_DETECT_SIGNALS_JS = """\
(() => {
    const bodyText = (document.body.textContent || '').toLowerCase();
    const bodyLen = bodyText.length;
    const hasCanvas = !!document.querySelector('canvas');
    const hasDraggable = document.querySelectorAll('[draggable="true"]').length;
    const hasRadio = document.querySelectorAll('input[type="radio"]').length;
    const hasRoleRadio = document.querySelectorAll('[role="radio"]').length;
    const hasShadowRoot = [...document.querySelectorAll('*')].some(el => el.shadowRoot);
    const hasIframe = document.querySelectorAll('iframe').length;
    const pageHeight = document.body.scrollHeight;
    const viewportHeight = window.innerHeight;

    const btns = [...document.querySelectorAll('button')].filter(b => b.offsetParent && !b.disabled);
    const btnTexts = btns.map(b => (b.textContent || '').trim().toLowerCase());

    const hasSubmitContinue = btnTexts.some(t => t.includes('submit & continue') || t.includes('submit and continue'));
    const hasCaptureBtn = btnTexts.some(t => t.includes('capture'));
    const hasPlayBtn = btnTexts.some(t => t.includes('play') && !t.includes('playing'));
    const hasConnectBtn = btnTexts.some(t => t === 'connect' || t.includes('connect to'));
    const hasRegisterBtn = btnTexts.some(t => t.includes('register'));
    const hasTriggerBtn = btnTexts.some(t => t.includes('trigger'));
    const hasRevealBtn = btnTexts.some(t => t.includes('reveal'));
    const hasCompleteBtn = btnTexts.some(t => t.includes('complete') || t.includes('done'));
    const hasSolveBtn = btnTexts.some(t => t === 'solve' || t.includes('check') || t.includes('verify'));
    const hasTabBtns = btnTexts.filter(t => t.includes('tab') || t.match(/^\\d+$/)).length;
    const hasExtractBtn = btnTexts.some(t => t.includes('extract'));
    const hasEnterLevelBtn = btnTexts.some(t => t.includes('enter level') || t.includes('go deeper') || t.includes('next level'));
    const hasClickMeBtn = btnTexts.some(t => t.includes('click me'));
    const hasFrameNav = btnTexts.some(t => t === '+1' || t === '-1' || t === '+10' || t === '-10');

    const hasAnimated = (() => {
        let count = 0;
        document.querySelectorAll('*').forEach(el => {
            const style = getComputedStyle(el);
            const cls = el.getAttribute('class') || '';
            if ((style.animation && style.animation !== 'none' && style.animationName !== 'none') ||
                cls.includes('animate-[move') || cls.includes('animate-[bounce')) count++;
        });
        return count;
    })();

    const scrollContainers = [...document.querySelectorAll('div, section')].filter(el => {
        const s = getComputedStyle(el);
        return (s.overflow + s.overflowY).match(/auto|scroll/) && el.scrollHeight > el.clientHeight + 10;
    }).length;

    const sectionDivCount = [...document.querySelectorAll('div')].filter(el => {
        const t = (el.textContent || '').trim();
        return t.match(/^Section \\d+/) && t.length > 30;
    }).length;

    const trapBtnCount = (() => {
        const TRAPS = ['proceed', 'continue', 'next step', 'next page', 'next section'];
        return btns.filter(b => {
            const t = (b.textContent || '').trim().toLowerCase();
            return t.length < 40 && TRAPS.some(w => t.includes(w));
        }).length;
    })();

    const hasPartElements = bodyText.includes('part') && (bodyText.includes('found') || bodyText.includes('collect'));
    const hasTimer = !!(bodyText.match(/\\d+\\.?\\d*\\s*s(?:econds?)?\\s*remaining/i));
    const hasMathExpr = !!(bodyText.match(/\\d+\\s*[+\\-*×÷\\/]\\s*\\d+\\s*=\\s*\\?/));
    const hasKeyboardSeq = bodyText.includes('keyboard sequence') || (bodyText.includes('press') && bodyText.includes('keys'));
    const hasHoverText = bodyText.includes('hover') && (bodyText.includes('reveal') || bodyText.includes('code'));
    const hasScrollInstruction = bodyText.includes('scroll down to find') || bodyText.includes('scroll to find') ||
        (bodyText.includes('keep scrolling') && bodyText.includes('navigation'));
    const hasShadowText = bodyText.includes('shadow') && (bodyText.includes('layer') || bodyText.includes('level') || bodyText.includes('nested'));
    const hasWebsocketText = bodyText.includes('websocket') || (bodyText.includes('connect') && bodyText.includes('server'));
    const hasServiceWorkerText = bodyText.includes('service worker') || (bodyText.includes('register') && bodyText.includes('cache'));
    const hasIframeText = bodyText.includes('iframe') && (bodyText.includes('level') || bodyText.includes('nested') || bodyText.includes('depth') || bodyText.includes('recursive'));
    const hasMutationText = bodyText.includes('mutation');
    const hasAudioText = bodyText.includes('audio') && (bodyText.includes('play') || bodyText.includes('listen'));
    const hasCanvasText = bodyText.includes('draw') || bodyText.includes('canvas') || bodyText.includes('stroke');
    const hasRotatingText = bodyText.includes('rotat') && bodyText.includes('capture');
    const hasSequenceText = bodyText.includes('sequence') || (bodyText.includes('click') && bodyText.includes('hover') && bodyText.includes('type'));
    const hasDelayedText = bodyText.includes('delayed') && (bodyText.includes('reveal') || bodyText.includes('remaining') || bodyText.includes('wait'));
    const hasVideoText = bodyText.includes('frame') && (bodyText.includes('navigate') || bodyText.includes('+1'));
    const hasMultiTabText = bodyText.includes('tab') && (bodyText.includes('click') || bodyText.includes('visit'));

    return {
        bodyLen, hasCanvas, hasDraggable, hasRadio, hasRoleRadio,
        hasShadowRoot, hasIframe, pageHeight, viewportHeight,
        hasSubmitContinue, hasCaptureBtn, hasPlayBtn, hasConnectBtn,
        hasRegisterBtn, hasTriggerBtn, hasRevealBtn, hasCompleteBtn,
        hasSolveBtn, hasTabBtns, hasExtractBtn, hasEnterLevelBtn,
        hasClickMeBtn, hasFrameNav, hasAnimated, scrollContainers,
        sectionDivCount, trapBtnCount, hasPartElements, hasTimer,
        hasMathExpr, hasKeyboardSeq, hasHoverText, hasScrollInstruction,
        hasShadowText, hasWebsocketText, hasServiceWorkerText,
        hasIframeText, hasMutationText, hasAudioText, hasCanvasText,
        hasRotatingText, hasSequenceText, hasDelayedText, hasVideoText,
        hasMultiTabText,
    };
})()
"""


class ChallengeDetector:
    """Detects which challenge type is present on the current page."""

    def detect(self, page) -> list[DetectionResult]:
        """Collect signals from the page and rank challenge types.

        Args:
            page: Playwright sync Page object.

        Returns:
            List of DetectionResult sorted by confidence (highest first).
        """
        try:
            signals = page.evaluate(_DETECT_SIGNALS_JS)
        except Exception as e:
            logger.warning("Detection signal collection failed: %s", e)
            return [DetectionResult(ChallengeType.UNKNOWN, 0.0)]

        results: list[DetectionResult] = []

        def _add(ct: ChallengeType, conf: float, **kw: bool) -> None:
            results.append(DetectionResult(ct, min(conf, 1.0), dict(kw)))

        s = signals  # shorthand

        # --- Math puzzle ---
        if s.get("hasMathExpr") or s.get("hasSolveBtn"):
            conf = 0.5
            if s.get("hasMathExpr"):
                conf += 0.4
            if s.get("hasSolveBtn"):
                conf += 0.1
            _add(ChallengeType.MATH_PUZZLE, conf, math=True, solve_btn=bool(s.get("hasSolveBtn")))

        # --- Keyboard sequence ---
        if s.get("hasKeyboardSeq"):
            _add(ChallengeType.KEYBOARD_SEQUENCE, 0.9, keyboard=True)

        # --- Radio brute force ---
        if s.get("hasRadio") or s.get("hasRoleRadio") or s.get("hasSubmitContinue"):
            conf = 0.4
            if s.get("hasRadio") or s.get("hasRoleRadio"):
                conf += 0.3
            if s.get("hasSubmitContinue"):
                conf += 0.3
            _add(ChallengeType.RADIO_BRUTE_FORCE, conf, radio=True)

        # --- Scroll to find ---
        if s.get("hasScrollInstruction") or (s.get("pageHeight", 0) > 5000 and s.get("sectionDivCount", 0) > 10):
            conf = 0.5
            if s.get("hasScrollInstruction"):
                conf += 0.4
            if s.get("trapBtnCount", 0) >= 8:
                conf += 0.1
            _add(ChallengeType.SCROLL_TO_FIND, conf, scroll=True)

        # --- Hover reveal ---
        if s.get("hasHoverText"):
            _add(ChallengeType.HOVER_REVEAL, 0.85, hover=True)

        # --- Timing capture ---
        if s.get("hasTimer") or (s.get("hasCaptureBtn") and not s.get("hasRotatingText")):
            conf = 0.5
            if s.get("hasTimer"):
                conf += 0.3
            if s.get("hasCaptureBtn"):
                conf += 0.2
            _add(ChallengeType.TIMING_CAPTURE, conf, timing=True)

        # --- Multi-tab ---
        if s.get("hasMultiTabText") and s.get("hasTabBtns", 0) >= 2:
            _add(ChallengeType.MULTI_TAB, 0.85, multi_tab=True)

        # --- Sequence challenge ---
        if s.get("hasSequenceText") or (s.get("hasClickMeBtn") and s.get("hasHoverText")):
            _add(ChallengeType.SEQUENCE_CHALLENGE, 0.85, sequence=True)

        # --- Video frames ---
        if s.get("hasVideoText") and s.get("hasFrameNav"):
            _add(ChallengeType.VIDEO_FRAMES, 0.9, video=True)

        # --- Split parts ---
        if s.get("hasPartElements"):
            _add(ChallengeType.SPLIT_PARTS, 0.8, parts=True)

        # --- Rotating code ---
        if s.get("hasRotatingText"):
            _add(ChallengeType.ROTATING_CODE, 0.85, rotating=True)

        # --- Delayed reveal ---
        if s.get("hasDelayedText") or s.get("hasTimer"):
            conf = 0.4
            if s.get("hasDelayedText"):
                conf += 0.4
            if s.get("hasTimer"):
                conf += 0.2
            _add(ChallengeType.DELAYED_REVEAL, conf, delayed=True)

        # --- Drag and drop ---
        if s.get("hasDraggable", 0) >= 3:
            _add(ChallengeType.DRAG_AND_DROP, 0.9, draggable=True)

        # --- Animated button ---
        if s.get("hasAnimated", 0) > 0:
            _add(ChallengeType.ANIMATED_BUTTON, 0.7, animated=True)

        # --- Shadow DOM ---
        if s.get("hasShadowText") or s.get("hasShadowRoot"):
            conf = 0.5
            if s.get("hasShadowText"):
                conf += 0.3
            if s.get("hasShadowRoot"):
                conf += 0.2
            _add(ChallengeType.SHADOW_DOM, conf, shadow=True)

        # --- Canvas draw ---
        if s.get("hasCanvas") and s.get("hasCanvasText"):
            _add(ChallengeType.CANVAS_DRAW, 0.9, canvas=True)

        # --- Audio challenge ---
        if s.get("hasAudioText") and s.get("hasPlayBtn"):
            _add(ChallengeType.AUDIO_CHALLENGE, 0.9, audio=True)
        elif s.get("hasAudioText"):
            _add(ChallengeType.AUDIO_CHALLENGE, 0.6, audio=True)

        # --- WebSocket ---
        if s.get("hasWebsocketText") or (s.get("hasConnectBtn") and not s.get("hasServiceWorkerText")):
            conf = 0.5
            if s.get("hasWebsocketText"):
                conf += 0.4
            if s.get("hasConnectBtn"):
                conf += 0.1
            _add(ChallengeType.WEBSOCKET, conf, websocket=True)

        # --- Service Worker ---
        if s.get("hasServiceWorkerText") or (s.get("hasRegisterBtn") and "cache" in str(signals)):
            conf = 0.5
            if s.get("hasServiceWorkerText"):
                conf += 0.4
            if s.get("hasRegisterBtn"):
                conf += 0.1
            _add(ChallengeType.SERVICE_WORKER, conf, service_worker=True)

        # --- Iframe recursive ---
        if s.get("hasIframeText") or (s.get("hasEnterLevelBtn") and s.get("hasExtractBtn")):
            conf = 0.5
            if s.get("hasIframeText"):
                conf += 0.3
            if s.get("hasEnterLevelBtn"):
                conf += 0.1
            if s.get("hasExtractBtn"):
                conf += 0.1
            _add(ChallengeType.IFRAME_RECURSIVE, conf, iframe=True)

        # --- Mutation observer ---
        if s.get("hasMutationText") or s.get("hasTriggerBtn"):
            conf = 0.4
            if s.get("hasMutationText"):
                conf += 0.4
            if s.get("hasTriggerBtn"):
                conf += 0.2
            _add(ChallengeType.MUTATION_OBSERVER, conf, mutation=True)

        # --- DOM extraction (fallback — always present at low confidence) ---
        _add(ChallengeType.DOM_EXTRACTION, 0.3)

        # Always include UNKNOWN as final fallback.
        _add(ChallengeType.UNKNOWN, 0.1)

        results.sort(key=lambda r: r.confidence, reverse=True)

        # Log when only fallback detectors fired — helps identify
        # unrecognized challenge types that need new detectors.
        if results and results[0].challenge_type == ChallengeType.DOM_EXTRACTION:
            active_signals = {k: v for k, v in s.items()
                              if v and v is not False and v != 0}
            logger.info("No specific challenge detected. Active signals: %s",
                        active_signals)

        return results
