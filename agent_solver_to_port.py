"""Agent-based challenge solver using Gemini 3 vision models.

Instead of hardcoding 20+ challenge types with heuristics, this solver
uses Gemini 3 Flash/Pro to SEE the page and REASON about what to do.
The AI agent handles the tricky challenges (scroll reveals, hover codes,
trap buttons) while deterministic JS handles popups and code submission.
"""
import asyncio
import re
import time
import base64

from browser import BrowserController
from agent_vision import AgentVision, ActionType
from dom_parser import extract_hidden_codes
from metrics import MetricsTracker


class AgentChallengeSolver:
    def __init__(self, api_key: str, timeout: int = 300):
        self.api_key = api_key
        self.browser = BrowserController()
        self.vision = AgentVision(api_key)
        self.metrics = MetricsTracker()
        self.current_step = 0
        self.keep_browser_open = False
        self.submit_is_trap = False  # Set when "Wrong Button!" detected
        self.timeout = timeout

    async def run(self, start_url: str, headless: bool = False) -> dict:
        """Run through all 30 challenges."""
        await self.browser.start(start_url, headless=headless)

        try:
            await asyncio.sleep(2)
            print("Clicking START button...", flush=True)
            await self.browser.click_by_text("START")
            await asyncio.sleep(1)

            run_start = time.time()
            self.run_start = run_start
            for step in range(1, 31):
                self.current_step = step
                self.metrics.start_challenge(step)
                step_start = time.time()
                elapsed = step_start - run_start
                print(f"\n{'='*60}", flush=True)
                print(f"  STEP {step}/30  (elapsed: {elapsed:.1f}s)", flush=True)
                print(f"{'='*60}", flush=True)

                # Soft timeout: stop starting new steps near the time limit
                if elapsed > self.timeout - 10:
                    print(f"  SOFT TIMEOUT: {elapsed:.0f}s elapsed, stopping step loop", flush=True)
                    self.metrics.end_challenge(step, success=False, error="Soft timeout")
                    break

                success = await self._solve_step(step)

                step_time = time.time() - step_start
                status = "PASSED" if success else "FAILED"
                print(f"  [{step_time:.1f}s] Step {step} {status}", flush=True)

                if not success:
                    self.metrics.end_challenge(step, success=False, error="Max attempts reached")
        finally:
            self.metrics.print_summary()
            if not self.keep_browser_open:
                await self.browser.stop()
            else:
                print("\n[KEEP-OPEN] Browser left open for manual debugging.", flush=True)
                print("[KEEP-OPEN] Press Ctrl+C to close.", flush=True)

        return self.metrics.get_summary()

    async def _solve_step(self, step: int) -> bool:
        """Solve a single challenge step using the agent loop."""
        total_tin = 0
        total_tout = 0
        failed_codes: list[str] = []
        action_history: list[str] = []
        max_attempts = 15
        scroll_attempted = False  # Prevent running scroll-to-find multiple times per step
        self.submit_is_trap = False  # Reset per step

        # Wait for React to render
        await self._wait_for_content()

        for attempt in range(max_attempts):
            # Global time budget check - leave 15s buffer for cleanup
            global_elapsed = time.time() - getattr(self, 'run_start', time.time())
            if global_elapsed > self.timeout - 15:
                print(f"  TIMEOUT RISK: {global_elapsed:.0f}s elapsed, skipping remaining attempts", flush=True)
                break

            # 1. Check if already progressed
            url = await self.browser.get_url()
            if self._check_progress(url, step):
                self.metrics.end_challenge(step, True, total_tin, total_tout)
                print(f"  >>> PASSED <<<", flush=True)
                return True

            # 2. Clear popups (fast, deterministic JS)
            cleared = await self._clear_popups()
            if cleared > 0:
                print(f"  Cleared {cleared} popups", flush=True)
                await asyncio.sleep(0.2)

            # 3. Attempt 0: fast path - DOM extraction + basic interactions
            if attempt == 0:
                # Scroll to trigger scroll-reveal challenges
                await self.browser.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(0.3)
                await self.browser.page.evaluate("window.scrollTo(0, 0)")
                await asyncio.sleep(0.2)
                # Scroll again (some need 500px+)
                await self.browser.page.evaluate("window.scrollTo(0, 1000)")
                await asyncio.sleep(0.3)

                # Click Reveal Code buttons + challenge-specific buttons
                await self.browser.page.evaluate("""() => {
                    document.querySelectorAll('button').forEach(btn => {
                        const t = btn.textContent.toLowerCase();
                        if ((t.includes('reveal') || t.includes('accept') ||
                             t.includes('register') || t.includes('retrieve') ||
                             t.includes('connect') || t.includes('trigger')) && 
                            btn.offsetParent && !btn.disabled) {
                            btn.click();
                        }
                    });
                }""")
                await asyncio.sleep(0.3)

                # Click "click here to reveal" elements multiple times
                for _ in range(5):
                    clicked = await self.browser.page.evaluate("""() => {
                        let clicked = 0;
                        document.querySelectorAll('div, p, span').forEach(el => {
                            const text = el.textContent || '';
                            if (text.includes('click here') && text.includes('to reveal')) {
                                el.click();
                                clicked++;
                            }
                        });
                        return clicked;
                    }""")
                    if clicked == 0:
                        break
                    await asyncio.sleep(0.2)

                # Scroll modal containers
                await self.browser.page.evaluate("""() => {
                    document.querySelectorAll('[class*="overflow-y"], [class*="overflow-auto"], [class*="max-h"]').forEach(el => {
                        if (el.scrollHeight > el.clientHeight) el.scrollTop = el.scrollHeight;
                    });
                }""")

                # Try DOM codes
                html = await self.browser.get_html()
                codes = extract_hidden_codes(html)
                if codes:
                    print(f"  DOM codes: {codes}", flush=True)
                    for code in codes:
                        if code in failed_codes:
                            continue
                        if await self._fill_and_submit(code, step):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED <<<", flush=True)
                            return True
                        failed_codes.append(code)

                # Check for radio modal - brute force (handles native + custom)
                if await self._brute_force_radio(step):
                    self.metrics.end_challenge(step, True, total_tin, total_tout)
                    print(f"  >>> PASSED <<<", flush=True)
                    return True

                # Handle keyboard sequences
                html_text = await self.browser.page.evaluate("() => document.body.textContent || ''")
                if 'keyboard sequence' in html_text.lower() or ('press' in html_text.lower() and 'keys' in html_text.lower()):
                    keys = re.findall(r'((?:Control|Shift|Alt|Meta)\+[A-Za-z0-9])', html_text)
                    seen = set()
                    unique_keys = []
                    for k in keys:
                        if k not in seen:
                            seen.add(k)
                            unique_keys.append(k)
                    if unique_keys:
                        print(f"  Keyboard sequence: {unique_keys}", flush=True)
                        await self.browser.page.evaluate("() => document.body.focus()")
                        for k in unique_keys:
                            await self.browser.page.keyboard.press(k)
                            await asyncio.sleep(0.3)

                # Handle math puzzles
                if 'puzzle' in html_text.lower() and ('= ?' in html_text or '=?' in html_text):
                    math_code = await self._try_math_puzzle(failed_codes)
                    if math_code and math_code not in failed_codes:
                        if await self._fill_and_submit(math_code, step):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED <<<", flush=True)
                            return True
                        failed_codes.append(math_code)

                    # Force-reset puzzle if showing stale "already solved" state
                    if 'solved' in html_text.lower() and 'code revealed' in html_text.lower():
                        fresh_code = await self._force_reset_puzzle()
                        if fresh_code and fresh_code not in failed_codes:
                            print(f"  Force-reset puzzle code: {fresh_code}", flush=True)
                            if await self._fill_and_submit(fresh_code, step):
                                self.metrics.end_challenge(step, True, total_tin, total_tout)
                                print(f"  >>> PASSED (force-reset puzzle) <<<", flush=True)
                                return True
                            failed_codes.append(fresh_code)

                    # Even if math code was stale, solving puzzle may have unlocked something
                    # Try deep extraction after puzzle solve (React state may have changed)
                    post_math_codes = await self._deep_code_extraction(failed_codes)
                    if post_math_codes:
                        print(f"  Post-math deep codes: {post_math_codes[:5]}", flush=True)
                        for code in post_math_codes[:5]:
                            if code in failed_codes:
                                continue
                            if await self._fill_and_submit(code, step):
                                self.metrics.end_challenge(step, True, total_tin, total_tout)
                                print(f"  >>> PASSED (post-math deep) <<<", flush=True)
                                return True
                            failed_codes.append(code)

                # Handle timing/capture challenges
                if 'capture' in html_text.lower() and ('timing' in html_text.lower() or 'second' in html_text.lower()):
                    for _ in range(5):
                        await self.browser.page.evaluate("""() => {
                            const btns = [...document.querySelectorAll('button')];
                            for (const btn of btns) {
                                const t = (btn.textContent || '').trim().toLowerCase();
                                if (t.includes('capture') && btn.offsetParent && !btn.disabled) {
                                    btn.click(); return true;
                                }
                            }
                            return false;
                        }""")
                        await asyncio.sleep(1.0)

                # Handle hover challenge
                if 'hover' in html_text.lower() and ('reveal' in html_text.lower() or 'code' in html_text.lower()):
                    # Find hover targets and hover for 1.5s
                    target = await self.browser.page.evaluate("""() => {
                        // Remove floating decoys first
                        const decoys = ['Click Me!', 'Button!', 'Link!', 'Here!', 'Click Here!', 'Try This!'];
                        document.querySelectorAll('div, button, span').forEach(el => {
                            const style = getComputedStyle(el);
                            if ((style.position === 'absolute' || style.position === 'fixed') && decoys.includes(el.textContent.trim())) {
                                el.style.display = 'none';
                            }
                        });
                        // Find hover target
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
                        // Dispatch hover events
                        const opts = {bubbles: true, clientX: rect.x + rect.width/2, clientY: rect.y + rect.height/2};
                        el.dispatchEvent(new MouseEvent('mouseenter', opts));
                        el.dispatchEvent(new MouseEvent('mouseover', opts));
                        return {x: rect.x + rect.width/2, y: rect.y + rect.height/2};
                    }""")
                    if target:
                        await self.browser.page.mouse.move(target['x'], target['y'])
                        await asyncio.sleep(1.5)
                        print(f"  Hovered target for 1.5s", flush=True)

                # Handle "I Remember" buttons (memory challenge)
                await self.browser.page.evaluate("""() => {
                    document.querySelectorAll('button').forEach(btn => {
                        const t = (btn.textContent || '').trim().toLowerCase();
                        if (t.includes('i remember') && btn.offsetParent && !btn.disabled) btn.click();
                    });
                }""")

                # Handle audio challenge (headless Chromium has no TTS voices)
                if 'audio' in html_text.lower() and ('play' in html_text.lower() or 'listen' in html_text.lower()):
                    await self._try_audio_challenge()

                # Handle canvas drawing challenge
                has_canvas = await self.browser.page.evaluate("() => !!document.querySelector('canvas')")
                if has_canvas and ('draw' in html_text.lower() or 'canvas' in html_text.lower() or 'stroke' in html_text.lower()):
                    await self._try_canvas_challenge()

                # Handle Service Worker challenge
                if 'service worker' in html_text.lower() or ('register' in html_text.lower() and 'cache' in html_text.lower()):
                    sw_code = await self._try_service_worker_challenge()
                    if sw_code and sw_code not in failed_codes:
                        print(f"  Service Worker code: {sw_code}", flush=True)
                        if await self._submit_code_with_fallbacks(sw_code, step):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED (service worker) <<<", flush=True)
                            return True
                        failed_codes.append(sw_code)

                # Handle Shadow DOM challenge (click through layers)
                if 'shadow' in html_text.lower() and ('layer' in html_text.lower() or 'level' in html_text.lower() or 'nested' in html_text.lower()):
                    shadow_code = await self._try_shadow_dom_challenge()
                    if shadow_code and shadow_code not in failed_codes:
                        print(f"  Shadow DOM code: {shadow_code}", flush=True)
                        if await self._submit_code_with_fallbacks(shadow_code, step):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED (shadow DOM) <<<", flush=True)
                            return True
                        failed_codes.append(shadow_code)

                # Handle WebSocket challenge (connect + receive)
                if 'websocket' in html_text.lower() or ('connect' in html_text.lower() and 'server' in html_text.lower()):
                    ws_code = await self._try_websocket_challenge()
                    if ws_code and ws_code not in failed_codes:
                        print(f"  WebSocket code: {ws_code}", flush=True)
                        if await self._submit_code_with_fallbacks(ws_code, step):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED (websocket) <<<", flush=True)
                            return True
                        failed_codes.append(ws_code)

                # Handle Delayed Reveal challenge (wait for timer)
                if 'delayed' in html_text.lower() and ('reveal' in html_text.lower() or 'remaining' in html_text.lower() or 'wait' in html_text.lower()):
                    delay_code = await self._try_delayed_reveal()
                    if delay_code and delay_code not in failed_codes:
                        print(f"  Delayed Reveal code: {delay_code}", flush=True)
                        if await self._submit_code_with_fallbacks(delay_code, step):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED (delayed reveal) <<<", flush=True)
                            return True
                        failed_codes.append(delay_code)

                # Re-evaluate html_text for late-loading challenges
                html_text = await self.browser.page.evaluate("() => document.body.textContent || ''")
                html_lower = html_text.lower()

                # Also detect challenges by their buttons (text may be in modal/hidden)
                challenge_buttons = await self.browser.page.evaluate("""() => {
                    const btns = [...document.querySelectorAll('button')].filter(b => b.offsetParent && !b.disabled);
                    const texts = btns.map(b => b.textContent.trim().toLowerCase());
                    return {
                        hasTrigger: texts.some(t => t.includes('trigger mutation') || t.includes('trigger')),
                        hasGoDeeper: texts.some(t => t.includes('go deeper') || t.includes('enter level') || t.includes('next level')),
                        hasExtractCode: texts.some(t => t.includes('extract code')),
                        hasRegisterSW: texts.some(t => t.includes('register service')),
                        hasConnect: texts.some(t => t === 'connect' || t.includes('connect to')),
                    };
                }""")

                # Handle Mutation challenge (click trigger button N times)
                if 'mutation' in html_lower or 'trigger mutation' in html_lower or (challenge_buttons and challenge_buttons.get('hasTrigger')):
                    print(f"  Detected: Mutation challenge", flush=True)
                    mut_code = await self._try_mutation_challenge()
                    if mut_code and mut_code not in failed_codes:
                        print(f"  Mutation code: {mut_code}", flush=True)
                        if await self._submit_code_with_fallbacks(mut_code, step):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED (mutation) <<<", flush=True)
                            return True
                        failed_codes.append(mut_code)

                # Handle Recursive Iframe challenge (navigate nested levels)
                if ('iframe' in html_lower and ('level' in html_lower or 'nested' in html_lower or 'depth' in html_lower or 'recursive' in html_lower)) or \
                   (challenge_buttons and (challenge_buttons.get('hasGoDeeper') or challenge_buttons.get('hasExtractCode'))):
                    print(f"  Detected: Iframe challenge", flush=True)
                    iframe_code = await self._try_iframe_challenge()
                    if iframe_code and iframe_code not in failed_codes:
                        print(f"  Iframe code: {iframe_code}", flush=True)
                        if await self._submit_code_with_fallbacks(iframe_code, step):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED (iframe) <<<", flush=True)
                            return True
                        failed_codes.append(iframe_code)

                # Handle split parts challenge
                if 'part' in html_text.lower() and ('found' in html_text.lower() or 'collect' in html_text.lower()):
                    await self._try_split_parts()

                # Handle rotating code challenge
                if 'rotat' in html_text.lower() and 'capture' in html_text.lower():
                    await self._try_rotating_code()

                # Handle multi-tab challenge
                if 'tab' in html_text.lower() and ('click' in html_text.lower() or 'visit' in html_text.lower()):
                    await self._try_multi_tab()

                # Handle sequence challenge (click, hover, type, scroll)
                if 'sequence' in html_text.lower() or ('click' in html_text.lower() and 'hover' in html_text.lower() and 'type' in html_text.lower()):
                    await self._try_sequence_challenge()

                # Handle video frames challenge
                if 'frame' in html_text.lower() and ('navigate' in html_text.lower() or '+1' in html_text or '-1' in html_text):
                    await self._try_video_challenge()

                # Deep code extraction (React state, CSS, shadow DOM, JS vars, network)
                deep_codes = await self._deep_code_extraction(failed_codes)
                if deep_codes:
                    print(f"  Deep extraction codes: {deep_codes[:8]}", flush=True)
                    for code in deep_codes[:10]:
                        if code in failed_codes:
                            continue
                        if await self._fill_and_submit(code, step):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED (deep extraction) <<<", flush=True)
                            return True
                        failed_codes.append(code)

                # === Handle "Submit is trap" / animated button challenges ===
                # If we detected that the normal submit button is a trap, try alternatives
                if self.submit_is_trap:
                    trap_elapsed = time.time() - getattr(self, 'run_start', time.time())
                    if trap_elapsed > self.timeout - 25:
                        print(f"  Submit-is-trap: skipping, {trap_elapsed:.0f}s elapsed", flush=True)
                    else:
                        print(f"  Submit is trap - trying animated button with all codes...", flush=True)
                        # Wait briefly for delayed content blocks to load
                        await asyncio.sleep(1.5)
                        # Re-extract codes after content loaded
                        fresh_deep = await self._deep_code_extraction(failed_codes)
                        all_codes_to_try = list(dict.fromkeys(
                            (fresh_deep or []) + (deep_codes or []) + list(codes or []) + list(failed_codes)
                        ))
                        # Try each code with animated button
                        for code in all_codes_to_try[:10]:
                            if await self._try_animated_button_submit(code, step):
                                self.metrics.end_challenge(step, True, total_tin, total_tout)
                                print(f"  >>> PASSED (animated button) <<<", flush=True)
                                return True
                        # Also try brute-forcing with common patterns
                        body_text = await self.browser.page.evaluate("() => document.body.textContent || ''")
                        # Extract ALL 6-char codes visible on page
                        all_visible = await self.browser.page.evaluate("""() => {
                            const text = document.body.innerText || '';
                            return [...new Set((text.match(/\\b[A-Z0-9]{6}\\b/g) || []))];
                        }""")
                        from dom_parser import FALSE_POSITIVES
                        fresh_visible = [c for c in all_visible if c not in FALSE_POSITIVES]
                        for code in fresh_visible[:10]:
                            if await self._try_animated_button_submit(code, step):
                                self.metrics.end_challenge(step, True, total_tin, total_tout)
                                print(f"  >>> PASSED (animated+visible) <<<", flush=True)
                                return True
                        # When main submit is trap, try "trap" buttons as potential real submits
                        print(f"  Trying trap buttons as real submits...", flush=True)
                        if await self._try_trap_buttons(step, all_codes_to_try[:3]):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED (trap button was real) <<<", flush=True)
                            return True

                # Handle "Scroll Down to Find Navigation" - check headers, main container, and page structure
                is_scroll_challenge = await self.browser.page.evaluate("""() => {
                    // Counter-check: if other challenge types are clearly present, skip scroll detection
                    const bodyText = (document.body.textContent || '').toLowerCase();
                    const hasCanvas = !!document.querySelector('canvas');
                    const hasAudio = bodyText.includes('audio challenge') || (bodyText.includes('play audio') && bodyText.includes('complete'));
                    const hasDrag = document.querySelectorAll('[draggable="true"]').length >= 3;
                    if (hasCanvas || hasAudio || hasDrag) return false;

                    // Check heading/prominent text elements for scroll instruction
                    const els = document.querySelectorAll('h1, h2, h3, .text-2xl, .text-3xl, .text-xl, .font-bold, .text-lg');
                    for (const el of els) {
                        const t = (el.textContent || '').toLowerCase();
                        if (t.includes('scroll down to find') || t.includes('scroll to find')) return true;
                    }
                    // Check the main challenge box
                    const mainBox = document.querySelector('.max-w-6xl, .max-w-4xl, .max-w-3xl');
                    if (mainBox) {
                        const t = (mainBox.textContent || '').toLowerCase();
                        if (t.includes('scroll down') && (t.includes('navigation') || t.includes('navigate') || t.includes('nav button'))) return true;
                    }
                    // Check body text for scroll-related instructions
                    if (bodyText.includes('keep scrolling') && bodyText.includes('navigation button')) return true;
                    // Structural check: many sections with filler text + scroll height > 5000px
                    if (document.body.scrollHeight > 5000) {
                        const sections = document.querySelectorAll('[class*="section"], [class*="Section"]');
                        const sectionDivs = [...document.querySelectorAll('div')].filter(el => {
                            const t = (el.textContent || '').trim();
                            return t.match(/^Section \\d+/) && t.length > 50;
                        });
                        if (sections.length > 10 || sectionDivs.length > 10) return true;
                    }
                    return false;
                }""")
                if is_scroll_challenge and not scroll_attempted:
                    scroll_attempted = True
                    all_codes = list(codes) + list(failed_codes) if codes else list(failed_codes)
                    if await self._try_scroll_to_find_nav(all_codes):
                        url = await self.browser.get_url()
                        if self._check_progress(url, step):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED <<<", flush=True)
                            return True

                # Handle "Delayed Reveal" - short fallback wait (main handler runs earlier)
                has_timer = await self.browser.page.evaluate("""() => {
                    const text = document.body.textContent || '';
                    return !!(text.match(/\\d+\\.?\\d*\\s*s(?:econds?)?\\s*remaining/i) ||
                              text.match(/delayed\\s+reveal/i));
                }""")
                if has_timer:
                    await asyncio.sleep(1.5)
                    print(f"  Waited 1.5s for delayed reveal (fallback)", flush=True)

                # Extract codes right after math puzzle and delayed reveal (before scroll changes page)
                html_fresh = await self.browser.get_html()
                fresh_codes = extract_hidden_codes(html_fresh)
                for code in fresh_codes:
                    if code in failed_codes:
                        continue
                    if await self._fill_and_submit(code, step):
                        self.metrics.end_challenge(step, True, total_tin, total_tout)
                        print(f"  >>> PASSED <<<", flush=True)
                        return True
                    failed_codes.append(code)

                # Hide floating decoy elements that obstruct drag area
                await self.browser.page.evaluate("""() => {
                    document.querySelectorAll('div, button, a, span').forEach(el => {
                        const style = getComputedStyle(el);
                        const text = (el.textContent || '').trim();
                        if (style.position === 'absolute' || style.position === 'fixed') {
                            if (['Click Me!', 'Button!', 'Link!', 'Here!', 'Click Here', 'Click Here!', 'Try This!'].includes(text)) {
                                el.style.display = 'none';
                                el.style.pointerEvents = 'none';
                            }
                        }
                    });
                }""")
                # Handle drag-and-drop via JS events
                await self.browser.page.evaluate("""() => {
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
                    // Click Complete/Done button
                    document.querySelectorAll('button').forEach(btn => {
                        const t = (btn.textContent || '').trim().toLowerCase();
                        if ((t.includes('complete') || t.includes('done') || t.includes('verify')) &&
                            !t.includes('clear') && btn.offsetParent && !btn.disabled) btn.click();
                    });
                }""")
                await asyncio.sleep(0.3)

                # Playwright mouse-based drag-and-drop fallback (if JS events didn't fill all slots)
                fill_count = await self.browser.page.evaluate("""() => {
                    const text = document.body.textContent || '';
                    const match = text.match(/(\\d+)\\/(\\d+)\\s*filled/);
                    return match ? parseInt(match[1]) : -1;
                }""")
                if fill_count >= 0 and fill_count < 6:
                    await self._try_mouse_drag_and_drop()

                # Re-extract codes after all fast path actions
                html = await self.browser.get_html()
                codes = extract_hidden_codes(html)
                for code in codes:
                    if code in failed_codes:
                        continue
                    if await self._fill_and_submit(code, step):
                        self.metrics.end_challenge(step, True, total_tin, total_tout)
                        print(f"  >>> PASSED <<<", flush=True)
                        return True
                    failed_codes.append(code)

                # Check progress after fast path
                url = await self.browser.get_url()
                if self._check_progress(url, step):
                    self.metrics.end_challenge(step, True, total_tin, total_tout)
                    print(f"  >>> PASSED <<<", flush=True)
                    return True

                # If all extracted codes failed and there are many trap buttons, try scroll-to-find
                if failed_codes:
                    trap_count = await self.browser.page.evaluate("""() => {
                        const TRAPS = ['proceed', 'continue', 'next step', 'next page', 'next section'];
                        return [...document.querySelectorAll('button')].filter(b => {
                            const t = (b.textContent || '').trim().toLowerCase();
                            return t.length < 40 && TRAPS.some(w => t.includes(w));
                        }).length;
                    }""")
                    if trap_count >= 8 and not scroll_attempted:
                        scroll_attempted = True
                        print(f"  {trap_count} trap buttons detected, trying scroll-to-find...", flush=True)
                        # Many trap buttons strongly suggests a scroll-to-find challenge
                        if await self._try_scroll_to_find_nav(list(failed_codes), deep_scroll=True):
                            url = await self.browser.get_url()
                            if self._check_progress(url, step):
                                self.metrics.end_challenge(step, True, total_tin, total_tout)
                                print(f"  >>> PASSED <<<", flush=True)
                                return True

                action_history.append("fast_path: scrolled, clicked reveals, tried DOM codes, handled specials")
                continue

            # 4. AI Agent: take screenshot, ask Gemini what to do
            print(f"  [attempt {attempt+1}] Asking Gemini...", flush=True)

            # Clear popups and hide stuck modals before screenshot
            await self._clear_popups()
            if attempt >= 2:
                await self._hide_stuck_modals()

            # Alternate scroll position for variety
            if attempt % 3 == 0:
                await self.browser.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            elif attempt % 3 == 1:
                await self.browser.page.evaluate("window.scrollTo(0, 0)")
            else:
                await self.browser.page.evaluate("window.scrollTo(0, 500)")
            await asyncio.sleep(0.3)

            screenshot = await self.browser.screenshot()
            html = await self.browser.get_html()
            dom_codes = extract_hidden_codes(html)

            action, tin, tout = self.vision.analyze(
                screenshot_bytes=screenshot,
                html_snippet=html[:6000],
                step=step,
                attempt=attempt,
                dom_codes=dom_codes,
                failed_codes=failed_codes,
                history=action_history,
            )
            total_tin += tin
            total_tout += tout

            # 5. If the agent found a code, try it immediately
            if action.code_found:
                code = action.code_found.upper().strip()
                if len(code) == 6 and code not in failed_codes:
                    print(f"  Agent found code: {code}", flush=True)
                    if await self._fill_and_submit(code, step):
                        self.metrics.end_challenge(step, True, total_tin, total_tout)
                        print(f"  >>> PASSED <<<", flush=True)
                        return True
                    # If submit is trap, also try animated button
                    if self.submit_is_trap:
                        if await self._try_animated_button_submit(code, step):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED (agent+animated) <<<", flush=True)
                            return True
                    failed_codes.append(code)

            # 6. Execute the agent's suggested action
            action_desc = await self._execute_action(action)
            action_history.append(action_desc)

            # 7. After action, try DOM codes again (action may have revealed new ones)
            await asyncio.sleep(0.3)
            html = await self.browser.get_html()
            new_codes = extract_hidden_codes(html)
            for code in new_codes:
                if code in failed_codes:
                    continue
                print(f"  New code after action: {code}", flush=True)
                if await self._fill_and_submit(code, step):
                    self.metrics.end_challenge(step, True, total_tin, total_tout)
                    print(f"  >>> PASSED <<<", flush=True)
                    return True
                # Animated button fallback when submit is trap
                if self.submit_is_trap:
                    if await self._try_animated_button_submit(code, step):
                        self.metrics.end_challenge(step, True, total_tin, total_tout)
                        print(f"  >>> PASSED (animated after action) <<<", flush=True)
                        return True
                failed_codes.append(code)

            # 7b. Deep extraction after action (React state may have changed)
            if attempt >= 2:
                deep = await self._deep_code_extraction(failed_codes)
                for code in deep[:5]:
                    if code in failed_codes:
                        continue
                    print(f"  Deep code after action: {code}", flush=True)
                    if await self._fill_and_submit(code, step):
                        self.metrics.end_challenge(step, True, total_tin, total_tout)
                        print(f"  >>> PASSED (deep) <<<", flush=True)
                        return True
                    if self.submit_is_trap:
                        if await self._try_animated_button_submit(code, step):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED (deep+animated) <<<", flush=True)
                            return True
                    failed_codes.append(code)

            # 8. Check progress
            url = await self.browser.get_url()
            if self._check_progress(url, step):
                self.metrics.end_challenge(step, True, total_tin, total_tout)
                print(f"  >>> PASSED <<<", flush=True)
                return True

            # 9. Every 4th attempt, try trap buttons with all known codes
            if attempt >= 4 and attempt % 4 == 0 and failed_codes:
                if await self._try_trap_buttons(step, failed_codes):
                    self.metrics.end_challenge(step, True, total_tin, total_tout)
                    print(f"  >>> PASSED <<<", flush=True)
                    return True

            # 10. Re-run key fast path handlers periodically
            if attempt >= 3 and attempt % 3 == 0:
                # Re-try scroll-to-find-nav, audio, canvas, etc.
                html_text = await self.browser.page.evaluate("() => document.body.textContent || ''")
                is_scroll_ch = await self.browser.page.evaluate("""() => {
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
                }""")
                if is_scroll_ch and not scroll_attempted:
                    scroll_attempted = True
                    if await self._try_scroll_to_find_nav(list(failed_codes)):
                        url = await self.browser.get_url()
                        if self._check_progress(url, step):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED <<<", flush=True)
                            return True
                if 'audio' in html_text.lower() and 'play' in html_text.lower():
                    await self._try_audio_challenge()
                if 'delayed' in html_text.lower() and 'remaining' in html_text.lower():
                    await asyncio.sleep(2.0)
                has_canvas = await self.browser.page.evaluate("() => !!document.querySelector('canvas')")
                if has_canvas:
                    await self._try_canvas_challenge()
                # Re-try drag-and-drop
                fill_count = await self.browser.page.evaluate("""() => {
                    const text = document.body.textContent || '';
                    const match = text.match(/(\\d+)\\/(\\d+)\\s*filled/);
                    return match ? parseInt(match[1]) : -1;
                }""")
                if fill_count >= 0 and fill_count < 6:
                    await self._try_mouse_drag_and_drop()
                # Re-try mutation challenge
                html_text = await self.browser.page.evaluate("() => document.body.textContent || ''")
                html_lower = html_text.lower()
                if 'mutation' in html_lower or 'trigger mutation' in html_lower:
                    mut_code = await self._try_mutation_challenge()
                    if mut_code and mut_code not in failed_codes:
                        print(f"  Mutation code: {mut_code}", flush=True)
                        if await self._submit_code_with_fallbacks(mut_code, step):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED (mutation) <<<", flush=True)
                            return True
                        failed_codes.append(mut_code)
                # Re-try iframe challenge
                if 'iframe' in html_lower and ('level' in html_lower or 'nested' in html_lower or 'depth' in html_lower):
                    iframe_code = await self._try_iframe_challenge()
                    if iframe_code and iframe_code not in failed_codes:
                        print(f"  Iframe code: {iframe_code}", flush=True)
                        if await self._submit_code_with_fallbacks(iframe_code, step):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED (iframe) <<<", flush=True)
                            return True
                        failed_codes.append(iframe_code)
                # Re-try shadow DOM challenge
                if 'shadow' in html_lower and ('level' in html_lower or 'layer' in html_lower):
                    shadow_code = await self._try_shadow_dom_challenge()
                    if shadow_code and shadow_code not in failed_codes:
                        if await self._submit_code_with_fallbacks(shadow_code, step):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED (shadow DOM retry) <<<", flush=True)
                            return True
                        failed_codes.append(shadow_code)
                # Re-try websocket challenge
                if 'websocket' in html_lower or ('connect' in html_lower and 'server' in html_lower):
                    ws_code = await self._try_websocket_challenge()
                    if ws_code and ws_code not in failed_codes:
                        if await self._submit_code_with_fallbacks(ws_code, step):
                            self.metrics.end_challenge(step, True, total_tin, total_tout)
                            print(f"  >>> PASSED (websocket retry) <<<", flush=True)
                            return True
                        failed_codes.append(ws_code)
                # Re-extract codes
                html = await self.browser.get_html()
                new_codes = extract_hidden_codes(html)
                for code in new_codes:
                    if code in failed_codes:
                        continue
                    if await self._submit_code_with_fallbacks(code, step):
                        self.metrics.end_challenge(step, True, total_tin, total_tout)
                        print(f"  >>> PASSED <<<", flush=True)
                        return True
                    failed_codes.append(code)

            # 11. Hide stuck modals after attempt 5 (they've been tried)
            if attempt == 5:
                hidden = await self._hide_stuck_modals()
                if hidden > 0:
                    print(f"  Hidden {hidden} stuck modals", flush=True)
                    # Re-try brute force radio (new options may be revealed)
                    if await self._brute_force_radio(step):
                        self.metrics.end_challenge(step, True, total_tin, total_tout)
                        print(f"  >>> PASSED <<<", flush=True)
                        return True

            await asyncio.sleep(0.1)

        return False

    # ── Deterministic helpers (no AI cost) ──────────────────────────────

    async def _wait_for_content(self) -> bool:
        """Wait for React SPA to render meaningful content."""
        for _ in range(10):
            html = await self.browser.get_html()
            if len(html) > 1000 and ("button" in html.lower() or "input" in html.lower()):
                return True
            await asyncio.sleep(0.5)
        return False

    async def _submit_code_with_fallbacks(self, code: str, step: int) -> bool:
        """Try to submit code: fill_and_submit first, then animated button if trap."""
        if await self._fill_and_submit(code, step):
            return True
        if self.submit_is_trap:
            if await self._try_animated_button_submit(code, step):
                return True
        return False

    def _check_progress(self, url: str, step: int) -> bool:
        """Check if URL indicates we've moved past current step."""
        url_lower = url.lower()
        if f"step{step + 1}" in url_lower or f"step-{step + 1}" in url_lower or f"step/{step + 1}" in url_lower:
            return True
        if step == 30 and ("complete" in url_lower or "finish" in url_lower or "done" in url_lower):
            return True
        match = re.search(r"step[/-]?(\d+)", url_lower)
        if match and int(match.group(1)) > step:
            return True
        return False

    @staticmethod
    def _sort_codes_by_priority(codes):
        """Sort codes: mixed alpha+digit first (most likely real), then all-alpha/all-digit."""
        def priority(c):
            has_d = any(ch.isdigit() for ch in c)
            has_a = any(ch.isalpha() for ch in c)
            return (not (has_d and has_a), c)
        return sorted(set(codes), key=priority)

    async def _deep_code_extraction(self, known_codes: list[str] | None = None) -> list[str]:
        """Extract codes from React fiber state, CSS pseudo-elements, shadow DOM, JS vars, network."""
        known = set(known_codes or [])
        all_codes = set()

        # 1. React Fiber Tree Walking
        react_codes = await self.browser.page.evaluate("""() => {
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
        }""")
        all_codes.update(react_codes or [])

        # 2. CSS pseudo-element content & custom properties
        css_codes = await self.browser.page.evaluate("""() => {
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
        }""")
        all_codes.update(css_codes or [])

        # 3. Shadow DOM traversal
        shadow_codes = await self.browser.page.evaluate("""() => {
            const codes = []; const RE = /\\b[A-Z0-9]{6}\\b/g;
            function walk(root) { if (!root || !root.shadowRoot) return;
                const m = ((root.shadowRoot.textContent||'') + ' ' + (root.shadowRoot.innerHTML||'')).toUpperCase().match(RE);
                if (m) codes.push(...m); root.shadowRoot.querySelectorAll('*').forEach(walk);
            }
            document.querySelectorAll('*').forEach(walk); return [...new Set(codes)];
        }""")
        all_codes.update(shadow_codes or [])

        # 4. JavaScript global variable scan
        js_codes = await self.browser.page.evaluate("""() => {
            const codes = []; const RE = /\\b[A-Z0-9]{6}\\b/g;
            const candidates = ['__NEXT_DATA__','__APP_DATA__','__CHALLENGE__','__code','__CODE','challengeCode','currentCode','navCode','secretCode','hiddenCode'];
            for (const g of candidates) { try { const v = window[g]; if (!v) continue;
                const s = typeof v === 'string' ? v : JSON.stringify(v).substring(0,10000); const m = s.toUpperCase().match(RE); if (m) codes.push(...m);
            } catch(e) {} }
            for (const k of Object.getOwnPropertyNames(window)) { try { const v = window[k]; if (typeof v === 'string' && /^[A-Z0-9]{6}$/.test(v)) codes.push(v); } catch(e) {} }
            return [...new Set(codes)];
        }""")
        all_codes.update(js_codes or [])

        # 5. Network intercepted codes
        all_codes.update(self.browser.intercepted_codes)

        # 6. Iframe content
        iframe_codes = await self.browser.page.evaluate("""() => {
            const codes = []; const RE = /\\b[A-Z0-9]{6}\\b/g;
            document.querySelectorAll('iframe').forEach(iframe => {
                try { const doc = iframe.contentDocument || iframe.contentWindow.document;
                    const m = (doc.body ? doc.body.textContent : '').toUpperCase().match(RE); if (m) codes.push(...m);
                } catch(e) {} });
            return [...new Set(codes)];
        }""")
        all_codes.update(iframe_codes or [])

        # Filter false positives
        from dom_parser import FALSE_POSITIVES
        LATIN = {'BEATAE','LABORE','DOLORE','VENIAM','NOSTRU','ALIQUA','EXERCI',
                 'TEMPOR','INCIDI','LABORI','MAGNAM','VOLUPT','SAPIEN','FUGIAT',
                 'COMMOD','EXCEPT','OFFICI','MOLLIT','PROIDE','REPUDI','FILLER',
                 'SCROLL','HIDDEN','BUTTON','SUBMIT','OPTION','CHOICE','REVEAL',
                 'PUZZLE','CANVAS','STROKE','SECOND','MEMORY','LOADED','BLOCKS',
                 'CHANGE','DELETE','CREATE','SEARCH','FILTER','NOTICE','STATUS',
                 'RESULT','OUTPUT','INPUTS','BEFORE','LAYOUT','RENDER','EFFECT',
                 'TOGGLE','HANDLE','CUSTOM','STRING','NUMBER','PROMPT','GLOBAL',
                 'MODULE','SHOULD','COOKIE','MOVING','FILLED','PIECES','VERIFY',
                 'DEVICE','SCREEN','MOBILE','TABLET','SELECT','PLEASE','SIMPLE',
                 'NEEDED','EXTEND','RANDOM','ACTIVE','PLAYED','ESCAPE','ALMOST',
                 'INSIDE','SOLVED','CENTER','BOTTOM','SHADOW','CURSOR','ROTATE',
                 'COLORS','IMAGES','CANCEL','RETURN','UPDATE','ALERTS','ERRORS'}
        all_codes -= FALSE_POSITIVES
        all_codes -= LATIN
        all_codes -= known
        all_codes = {c for c in all_codes if not c.isdigit()}
        all_codes = {c for c in all_codes if not re.match(r'^\d+(?:PX|VH|VW|EM|REM|MS|FR)$', c)}
        return self._sort_codes_by_priority(all_codes)

    async def _clear_popups(self) -> int:
        """Clear blocking popups using deterministic JS. Returns count cleared."""
        return await self.browser.page.evaluate("""() => {
            let cleared = 0;
            const hide = (el) => {
                el.style.display = 'none';
                el.style.pointerEvents = 'none';
                el.style.visibility = 'hidden';
                el.style.zIndex = '-1';
            };

            document.querySelectorAll('.fixed, [class*="absolute"], [class*="z-"]').forEach(el => {
                const text = el.textContent || '';

                // Popup with real dismiss button (fake one labeled, real one isn't)
                if (text.includes('fake') && text.includes('real one')) {
                    el.querySelectorAll('button').forEach(btn => {
                        const bt = (btn.textContent || '').trim();
                        if (!bt.toLowerCase().includes('fake') && bt.length > 0 && bt.length < 30) {
                            btn.click();
                            cleared++;
                        }
                    });
                }

                // Popup where ALL close buttons are fake
                if (text.includes('another way to close') ||
                    (text.includes('close button') && text.includes('fake') && !text.includes('real one')) ||
                    text.includes('won a prize') || text.includes('amazing deals')) {
                    hide(el);
                    cleared++;
                }

                // "That close button is fake!" warnings
                if (text.includes('That close button is fake')) {
                    hide(el);
                    cleared++;
                }

                // Cookie consent
                if (text.includes('Cookie') || text.includes('cookie')) {
                    const btn = [...el.querySelectorAll('button')].find(b => b.textContent.includes('Accept'));
                    if (btn) { btn.click(); cleared++; }
                }

                // Limited time offer / Click X to close
                if (text.includes('Limited time offer') || text.includes('Click X to close') ||
                    text.includes('popup message')) {
                    el.querySelectorAll('button').forEach(btn => btn.click());
                    hide(el);
                    cleared++;
                }

                // "Click the button to dismiss" modals
                if (text.includes('Click the button to dismiss') || text.includes('interact with this modal')) {
                    const btn = el.querySelector('button');
                    if (btn) { btn.click(); cleared++; }
                }

                // "Wrong Button" modals
                if (text.includes('Wrong Button') || text.includes('Try Again')) {
                    const btn = el.querySelector('button');
                    if (btn) { btn.click(); cleared++; }
                }
            });

            // Disable bg-black/70 overlays
            document.querySelectorAll('.fixed').forEach(el => {
                if (el.classList.contains('bg-black/70') ||
                    (el.style.backgroundColor || '').includes('rgba(0, 0, 0')) {
                    if (!el.textContent.includes('Step') && !el.querySelector('input[type="radio"]')) {
                        el.style.pointerEvents = 'none';
                        cleared++;
                    }
                }
            });

            return cleared;
        }""")

    async def _fill_and_submit(self, code: str, step: int) -> bool:
        """Fill code into input, click submit, check if URL changed."""
        url_before = await self.browser.get_url()

        try:
            # If we already know submit is a trap, go straight to animated button
            if self.submit_is_trap:
                if await self._try_animated_button_submit(code, step):
                    return True
                # Try Enter key
                await self.browser.page.evaluate(f"""() => {{
                    const inp = document.querySelector('input[placeholder*="code" i], input[type="text"]');
                    if (inp) {{
                        const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
                        s.call(inp, '{code}');
                        inp.dispatchEvent(new Event('input', {{bubbles: true}}));
                        inp.dispatchEvent(new Event('change', {{bubbles: true}}));
                        inp.focus();
                    }}
                }}""")
                await self.browser.page.keyboard.press("Enter")
                await asyncio.sleep(0.3)
                url_after = await self.browser.get_url()
                if url_after != url_before:
                    print(f"    Code '{code}' WORKED (Enter bypass)!", flush=True)
                    return True
                return False

            # Scroll input into view
            await self.browser.page.evaluate("""() => {
                const input = document.querySelector('input[placeholder*="code" i], input[type="text"]');
                if (input) input.scrollIntoView({behavior: 'instant', block: 'center'});
            }""")
            await asyncio.sleep(0.1)

            # Clear and type
            inp = self.browser.page.locator('input[placeholder*="code" i], input[type="text"]').first
            try:
                await inp.click(click_count=3, timeout=1000)
            except Exception:
                await self.browser.page.evaluate("""() => {
                    const input = document.querySelector('input[placeholder*="code" i], input[type="text"]');
                    if (input) { input.focus(); input.select(); }
                }""")
            await self.browser.page.keyboard.press("Backspace")
            await asyncio.sleep(0.05)
            await self.browser.page.keyboard.type(code, delay=20)
            await asyncio.sleep(0.15)

            # Click submit (avoid trap buttons)
            clicked = await self.browser.page.evaluate("""() => {
                const TRAPS = ['proceed', 'continue', 'next step', 'next page', 'next section',
                    'move on', 'go forward', 'keep going', 'advance', 'continue reading',
                    'continue journey', 'click here', 'proceed forward'];
                const isTrap = (t) => TRAPS.some(w => t.toLowerCase().includes(w));

                const input = document.querySelector('input[placeholder*="code" i], input[type="text"]');
                if (!input) return false;

                // Search in parent containers
                let container = input.parentElement;
                for (let i = 0; i < 4 && container; i++) {
                    const btns = container.querySelectorAll('button');
                    for (const btn of btns) {
                        const t = (btn.textContent || '').trim();
                        if (!btn.disabled && !isTrap(t) &&
                            (btn.type === 'submit' || t.includes('Submit') || t.includes('Go') || t === '→' || t.length <= 2)) {
                            btn.scrollIntoView({behavior: 'instant', block: 'center'});
                            btn.click();
                            return true;
                        }
                    }
                    // Single non-trap button in container
                    const safe = [...btns].filter(b => !b.disabled && !isTrap((b.textContent || '').trim()));
                    if (safe.length === 1) { safe[0].click(); return true; }
                    container = container.parentElement;
                }
                // Fallback: exact "Submit" or "Submit Code"
                for (const b of document.querySelectorAll('button')) {
                    const t = (b.textContent || '').trim();
                    if ((t === 'Submit' || t === 'Submit Code') && !b.disabled) { b.click(); return true; }
                }
                return false;
            }""")

            if not clicked:
                await self.browser.page.keyboard.press("Enter")

            await asyncio.sleep(0.4)

            # Check for "Wrong Button!" popup (means button is trap, not code is wrong)
            wrong_button = await self.browser.page.evaluate("""() => {
                const text = document.body.textContent || '';
                return text.includes('Wrong Button') || text.includes('wrong button');
            }""")
            if wrong_button:
                self.submit_is_trap = True
                print(f"    'Wrong Button!' detected - submit button is trap, trying animated button...", flush=True)
                await self._clear_popups()
                # Try animated/moving button as alternative submit
                if await self._try_animated_button_submit(code, step):
                    return True
                # Also try just pressing Enter on the input
                await self.browser.page.evaluate("""() => {
                    const inp = document.querySelector('input[placeholder*="code" i], input[type="text"]');
                    if (inp) inp.focus();
                }""")
                await self.browser.page.keyboard.press("Enter")
                await asyncio.sleep(0.3)
                url_after = await self.browser.get_url()
                if url_after != url_before:
                    print(f"    Code '{code}' WORKED (Enter key)!", flush=True)
                    return True
                # Return False but caller should know this is "wrong button" not "wrong code"
                print(f"    Code '{code}' - button trap, code may be valid", flush=True)
                return False

            url_after = await self.browser.get_url()

            if url_after != url_before:
                print(f"    Code '{code}' WORKED!", flush=True)
                return True
            else:
                print(f"    Code '{code}' failed", flush=True)
                return False
        except Exception as e:
            print(f"    Fill error: {e}", flush=True)
            return False

    async def _try_animated_button_submit(self, code: str, step: int) -> bool:
        """Find animated/moving elements, freeze them, enter code, click as submit."""
        try:
            result = await self.browser.page.evaluate("""() => {
                // Find elements with CSS animation (moveAround, bounce, etc.)
                const animated = [];
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
                        animated.push(el);
                    }
                });
                if (animated.length === 0) return {found: false};
                // Freeze all animated elements and return first clickable
                const results = [];
                for (const el of animated) {
                    el.style.animation = 'none';
                    el.style.position = 'fixed';
                    el.style.top = '50%';
                    el.style.left = '50%';
                    el.style.zIndex = '99999';
                    el.style.transform = 'translate(-50%, -50%)';
                    const r = el.getBoundingClientRect();
                    results.push({
                        x: Math.round(r.x + r.width/2),
                        y: Math.round(r.y + r.height/2),
                        text: (el.textContent || '').trim().substring(0, 40),
                        cls: (el.getAttribute('class') || '').substring(0, 60)
                    });
                }
                return {found: true, elements: results};
            }""")
            if not result or not result.get('found'):
                return False

            elements = result.get('elements', [])
            print(f"  Found {len(elements)} animated elements, trying as submit...", flush=True)

            # Fill code in input
            await self.browser.page.evaluate(f"""() => {{
                const inp = document.querySelector('input[placeholder*="code" i], input[type="text"]');
                if (inp) {{
                    const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
                    s.call(inp, '{code}');
                    inp.dispatchEvent(new Event('input', {{bubbles: true}}));
                    inp.dispatchEvent(new Event('change', {{bubbles: true}}));
                }}
            }}""")
            await asyncio.sleep(0.1)

            for el in elements:
                await self._clear_popups()
                try:
                    await self.browser.page.mouse.click(el['x'], el['y'])
                except Exception:
                    pass
                await asyncio.sleep(0.3)
                url = await self.browser.get_url()
                if self._check_progress(url, step):
                    print(f"  Animated button '{el['text']}' with code '{code}' WORKED!", flush=True)
                    return True

            return False
        except Exception as e:
            print(f"  Animated button error: {e}", flush=True)
            return False

    async def _brute_force_radio(self, step: int) -> bool:
        """Try all radio/option elements (native + custom). Brute force each + Submit."""
        # Scroll modal containers to reveal radio options (multiple strategies)
        await self.browser.page.evaluate("""() => {
            // Strategy 1: CSS class-based scroll
            document.querySelectorAll('[class*="overflow-y"], [class*="overflow-auto"], [class*="max-h"]').forEach(el => {
                if (el.scrollHeight > el.clientHeight) el.scrollTop = el.scrollHeight;
            });
            // Strategy 2: Scroll ALL scrollable children inside fixed modals
            document.querySelectorAll('.fixed').forEach(modal => {
                const scrollables = modal.querySelectorAll('*');
                scrollables.forEach(el => {
                    if (el.scrollHeight > el.clientHeight + 10) el.scrollTop = el.scrollHeight;
                });
            });
        }""")
        await asyncio.sleep(0.2)
        # Also use Playwright mouse wheel to scroll inside the modal
        modal_center = await self.browser.page.evaluate("""() => {
            const modal = [...document.querySelectorAll('.fixed')].find(el =>
                el.textContent.includes('Please Select') || el.textContent.includes('Submit & Continue'));
            if (!modal) return null;
            const rect = modal.getBoundingClientRect();
            return {x: rect.x + rect.width/2, y: rect.y + rect.height/2};
        }""")
        if modal_center:
            await self.browser.page.mouse.move(modal_center['x'], modal_center['y'])
            for _ in range(5):
                await self.browser.page.mouse.wheel(0, 500)
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.1)

        # Count options: native radios, role-based, OR custom option cards
        count = await self.browser.page.evaluate("""() => {
            // Strategy 1: Native radio inputs
            let opts = document.querySelectorAll('input[type="radio"]');
            if (opts.length > 0) return {count: opts.length, type: 'native'};
            // Strategy 2: Role-based radios
            opts = document.querySelectorAll('[role="radio"]');
            if (opts.length > 0) return {count: opts.length, type: 'role'};
            // Strategy 3: Custom option cards near Submit button
            const submitBtn = [...document.querySelectorAll('button')].find(b =>
                b.textContent.includes('Submit & Continue') || b.textContent.includes('Submit and Continue'));
            if (!submitBtn) return {count: 0, type: 'none'};
            // Walk up to find the modal container
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
                return text.length > 0 && text.length < 80 &&
                    !text.includes('Submit') && !text.includes('Section') &&
                    !text.includes('lorem ipsum') && !text.includes('Introduction');
            });
            return {count: cards.length, type: 'custom'};
        }""")

        radio_count = count.get('count', 0) if isinstance(count, dict) else 0
        radio_type = count.get('type', 'none') if isinstance(count, dict) else 'none'

        # Also detect "Please Select an Option" text as radio modal indicator
        if radio_count == 0:
            has_text = await self.browser.page.evaluate("""() => {
                const text = (document.body.textContent || '').toLowerCase();
                return text.includes('please select an option') && text.includes('submit');
            }""")
            if has_text:
                radio_type = 'text_detected'
                # Try clicking all bordered divs inside the modal
                radio_count = await self.browser.page.evaluate("""() => {
                    const modal = [...document.querySelectorAll('.fixed')].find(el =>
                        el.textContent.includes('Please Select') || el.textContent.includes('Submit & Continue'));
                    if (!modal) return 0;
                    const cards = [...modal.querySelectorAll('div[class*="border"], div[class*="cursor"], label')].filter(el => {
                        const t = el.textContent.trim();
                        return t.length > 0 && t.length < 80 && !t.includes('Submit') && el.offsetParent;
                    });
                    return cards.length;
                }""")

        if radio_count == 0:
            return False

        # Cap at 15 to avoid wasting time on large option sets
        effective_count = min(radio_count, 15)
        print(f"  Brute-forcing {effective_count}/{radio_count} {radio_type} options...", flush=True)

        for i in range(effective_count):
            await self.browser.page.evaluate("""(idx) => {
                // Find all option elements (re-find for React re-renders)
                let options = [...document.querySelectorAll('input[type="radio"]')];
                if (options.length === 0) options = [...document.querySelectorAll('[role="radio"]')];
                if (options.length === 0) {
                    // Custom option cards
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
                const opt = options[idx];
                if (!opt) return;
                opt.click();
                // Also click inner radio if exists
                const innerRadio = opt.querySelector('input[type="radio"]');
                if (innerRadio) innerRadio.click();
                // Click parent card for custom components
                const card = opt.closest('label, [class*="cursor-pointer"]');
                if (card && card !== opt) card.click();
                // Click Submit
                const btns = [...document.querySelectorAll('button')];
                const sub = btns.find(b => b.textContent.includes('Submit'));
                if (sub) sub.click();
            }""", i)
            await asyncio.sleep(0.15)
            url = await self.browser.get_url()
            if self._check_progress(url, step):
                print(f"  Radio option {i+1}/{radio_count} CORRECT!", flush=True)
                return True

        # Hide modal if all wrong
        print(f"  All {effective_count} radio options wrong, hiding modal", flush=True)
        await self.browser.page.evaluate("""() => {
            document.querySelectorAll('.fixed').forEach(el => {
                const text = el.textContent || '';
                if (el.querySelector('input[type="radio"]') || el.querySelector('[role="radio"]') ||
                    text.includes('Please Select') || text.includes('Submit & Continue')) {
                    el.style.display = 'none';
                    el.style.visibility = 'hidden';
                    el.style.pointerEvents = 'none';
                }
            });
        }""")
        return False

    async def _try_trap_buttons(self, step: int, codes: list[str]) -> bool:
        """Try clicking trap-labeled buttons with each code (some are actually real)."""
        TRAP_WORDS_JS = """['proceed', 'continue', 'next step', 'next page', 'next section',
            'move on', 'go forward', 'keep going', 'advance', 'continue journey',
            'click here', 'proceed forward', 'continue reading', 'next', 'go', 'submit code', 'submit']"""

        count = await self.browser.page.evaluate(f"""() => {{
            const TRAPS = {TRAP_WORDS_JS};
            return [...document.querySelectorAll('button, a')].filter(el => {{
                const t = (el.textContent || '').trim().toLowerCase();
                return t.length < 40 && TRAPS.some(w => t === w || t.includes(w));
            }}).length;
        }}""")

        if count == 0:
            return False

        # If many trap buttons, they're likely all decoys - limit attempts
        max_btns = 8 if count > 15 else min(count, 15)
        print(f"  Trying {min(len(codes), 2)} codes with {max_btns}/{count} trap buttons...", flush=True)

        for code in codes[:2]:
            for i in range(max_btns):
                # Clear any blocking popups from previous wrong-button clicks
                await self._clear_popups()

                await self.browser.page.evaluate(f"""(code) => {{
                    const input = document.querySelector('input[placeholder*="code" i], input[type="text"]');
                    if (input) {{
                        const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
                        s.call(input, code);
                        input.dispatchEvent(new Event('input', {{bubbles: true}}));
                        input.dispatchEvent(new Event('change', {{bubbles: true}}));
                    }}
                }}""", code)
                await asyncio.sleep(0.05)

                await self.browser.page.evaluate(f"""(idx) => {{
                    const TRAPS = {TRAP_WORDS_JS};
                    const btns = [...document.querySelectorAll('button, a')].filter(el => {{
                        const t = (el.textContent || '').trim().toLowerCase();
                        return t.length < 40 && TRAPS.some(w => t === w || t.includes(w));
                    }});
                    const btn = btns[idx];
                    if (btn) {{
                        btn.scrollIntoView({{behavior: 'instant', block: 'center'}});
                        btn.click();
                    }}
                }}""", i)
                await asyncio.sleep(0.15)
                url = await self.browser.get_url()
                if self._check_progress(url, step):
                    return True

        return False

    async def _try_math_puzzle(self, failed_codes: list[str] | None = None) -> str | None:
        """Solve math expression, type answer, click Solve. Returns revealed code if found."""
        known_bad = set(failed_codes or [])
        expr = await self.browser.page.evaluate("""() => {
            const text = document.body.textContent || '';
            const m = text.match(/(\\d+)\\s*([+\\-*×÷\\/])\\s*(\\d+)\\s*=\\s*\\?/);
            if (!m) return null;
            const a = parseInt(m[1]), op = m[2], b = parseInt(m[3]);
            let answer;
            switch(op) {
                case '+': answer = a + b; break;
                case '-': answer = a - b; break;
                case '*': case '×': answer = a * b; break;
                case '/': case '÷': answer = Math.floor(a / b); break;
                default: answer = a + b;
            }
            return String(answer);
        }""")
        if not expr:
            return None
        print(f"  Math puzzle answer: {expr}", flush=True)
        # Record codes BEFORE solving to detect new ones
        codes_before = set(await self.browser.page.evaluate("""() => {
            const text = document.body.textContent || '';
            const codes = text.match(/\\b[A-Z0-9]{6}\\b/g) || [];
            return [...new Set(codes)];
        }"""))

        # Try multiple input selectors (some puzzles use type="text" not "number")
        # First: JS setter approach (works with many React components)
        filled_js = await self.browser.page.evaluate(f"""() => {{
            const selectors = [
                'input[type="number"]',
                'input[inputmode="numeric"]',
                'input[placeholder*="answer" i]',
                'input[placeholder*="solution" i]',
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
        }}""")
        # Backup: Playwright native fill (triggers all browser events properly)
        if not filled_js:
            for sel in ['input[type="number"]', 'input[inputmode="numeric"]',
                        'input[placeholder*="answer" i]', 'input[placeholder*="solution" i]']:
                try:
                    loc = self.browser.page.locator(sel).first
                    if await loc.count() > 0:
                        await loc.fill(expr)
                        filled_js = sel
                        break
                except Exception:
                    continue
        await asyncio.sleep(0.2)
        await self.browser.page.keyboard.press("Enter")
        await asyncio.sleep(0.5)

        # Click Solve/Check/Verify button (try multiple times with increasing wait)
        for solve_attempt in range(3):
            clicked = await self.browser.page.evaluate("""() => {
                const btns = [...document.querySelectorAll('button')];
                for (const btn of btns) {
                    const t = (btn.textContent || '').trim().toLowerCase();
                    if ((t === 'solve' || t.includes('check') || t.includes('verify') || t === 'submit') && !btn.disabled) {
                        btn.click(); return true;
                    }
                }
                return false;
            }""")
            await asyncio.sleep(0.8)

            # Check delta after each attempt
            codes_after = set(await self.browser.page.evaluate("""() => {
                const text = document.body.textContent || '';
                const codes = text.match(/\\b[A-Z0-9]{6}\\b/g) || [];
                return [...new Set(codes)];
            }"""))
            new_codes = codes_after - codes_before
            # Filter out false positives AND already-failed codes
            from dom_parser import FALSE_POSITIVES
            new_codes = {c for c in new_codes if c not in FALSE_POSITIVES and c not in known_bad}
            if new_codes:
                # Prefer mixed alpha+digit codes
                sorted_new = self._sort_codes_by_priority(new_codes)
                code = sorted_new[0]
                print(f"  Math puzzle delta code: {code} (new: {sorted_new})", flush=True)
                return code

        # Fallback: pattern-based extraction (filter out already-failed codes)
        puzzle_code = await self.browser.page.evaluate("""() => {
            const text = document.body.textContent || '';
            const patterns = [
                /(?:code(?:\\s+is)?|revealed?)\\s*[:=]\\s*([A-Z0-9]{6})/i,
                /(?:solved|correct|success)[^.]*?\\b([A-Z0-9]{6})\\b/i,
                /\\b([A-Z0-9]{6})\\b(?=[^A-Z0-9]*(?:submit|enter|type|input))/i
            ];
            const results = [];
            for (const p of patterns) {
                const m = text.match(p);
                if (m) results.push(m[1].toUpperCase());
            }
            const successEls = document.querySelectorAll('.text-green-600, .text-green-500, .bg-green-100, .bg-green-50, .text-emerald-600');
            for (const el of successEls) {
                const t = (el.textContent || '').trim();
                const m = t.match(/\\b([A-Z0-9]{6})\\b/);
                if (m) results.push(m[1]);
            }
            return [...new Set(results)];
        }""")
        if puzzle_code:
            from dom_parser import FALSE_POSITIVES
            # Filter out false positives AND already-failed codes
            fresh = [c for c in puzzle_code if c not in known_bad and c not in FALSE_POSITIVES]
            if fresh:
                sorted_fresh = self._sort_codes_by_priority(fresh)
                print(f"  Math puzzle pattern code: {sorted_fresh[0]} (all: {sorted_fresh})", flush=True)
                return sorted_fresh[0]
            # All pattern codes are stale/false-positive - return None to avoid wasting time
            print(f"  Math puzzle pattern: all codes stale/FP: {puzzle_code}", flush=True)
            return None
        return None

    async def _force_reset_puzzle(self) -> str | None:
        """Force-reset a math puzzle that shows cached 'already solved' state.
        Resets the React component state to re-trigger code generation."""
        try:
            # Step 1: Find the math answer from the page text
            expr = await self.browser.page.evaluate("""() => {
                const text = document.body.textContent || '';
                const m = text.match(/(\\d+)\\s*([+\\-*×÷\\/])\\s*(\\d+)\\s*=\\s*\\?/);
                if (!m) return null;
                const a = parseInt(m[1]), op = m[2], b = parseInt(m[3]);
                switch(op) {
                    case '+': return String(a + b);
                    case '-': return String(a - b);
                    case '*': case '×': return String(a * b);
                    case '/': case '÷': return String(Math.floor(a / b));
                    default: return String(a + b);
                }
            }""")
            if not expr:
                return None

            print(f"  Force-reset puzzle: answer={expr}", flush=True)

            # Step 2: Record codes before reset
            codes_before = set(await self.browser.page.evaluate("""() => {
                const text = document.body.textContent || '';
                return [...new Set((text.match(/\\b[A-Z0-9]{6}\\b/g) || []))];
            }"""))

            # Step 3: Find puzzle container and reset React state via fiber
            reset_ok = await self.browser.page.evaluate("""() => {
                // Find the puzzle component (contains "Puzzle" and "Solve" text)
                const puzzleEls = [...document.querySelectorAll('div')].filter(el => {
                    const t = el.textContent || '';
                    return t.includes('Puzzle') && (t.includes('solved') || t.includes('Code revealed'))
                        && el.offsetParent && el.offsetWidth > 100;
                });
                // Sort by specificity (smallest container with puzzle text)
                puzzleEls.sort((a, b) => a.textContent.length - b.textContent.length);

                for (const el of puzzleEls.slice(0, 3)) {
                    // Try to find React fiber
                    const fiberKey = Object.keys(el).find(k =>
                        k.startsWith('__reactFiber$') || k.startsWith('__reactInternalInstance$'));
                    if (!fiberKey) continue;
                    let fiber = el[fiberKey];

                    // Walk up the fiber tree to find state with 'solved' or 'code'
                    let attempts = 0;
                    while (fiber && attempts < 30) {
                        if (fiber.memoizedState) {
                            let state = fiber.memoizedState;
                            let stateIdx = 0;
                            while (state && stateIdx < 20) {
                                const q = state.queue || state;
                                const val = state.memoizedState;
                                // Reset boolean states (likely 'solved' flag)
                                if (val === true && state.queue) {
                                    const dispatch = state.queue.dispatch;
                                    if (dispatch) {
                                        try { dispatch(false); } catch(e) {}
                                    }
                                }
                                // Reset string states that look like codes
                                if (typeof val === 'string' && /^[A-Z0-9]{6}$/.test(val) && state.queue) {
                                    const dispatch = state.queue.dispatch;
                                    if (dispatch) {
                                        try { dispatch(''); } catch(e) {}
                                    }
                                }
                                // Reset number states that might be attempt counters
                                if (typeof val === 'number' && val > 0 && val < 100 && state.queue) {
                                    const dispatch = state.queue.dispatch;
                                    if (dispatch) {
                                        try { dispatch(0); } catch(e) {}
                                    }
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
            }""")

            await asyncio.sleep(0.5)

            # Step 4: Check if input appeared after reset
            has_input = await self.browser.page.evaluate("""() => {
                const inp = document.querySelector('input[type="number"], input[inputmode="numeric"], ' +
                    'input[placeholder*="answer" i], input[placeholder*="solution" i]');
                return !!(inp && inp.offsetParent);
            }""")

            if has_input:
                print(f"  Force-reset: puzzle input appeared, entering {expr}...", flush=True)
                # Fill answer and solve
                await self.browser.page.evaluate(f"""() => {{
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
                }}""")
                await asyncio.sleep(0.2)
                await self.browser.page.keyboard.press("Enter")
                await asyncio.sleep(0.5)

                # Click Solve button
                await self.browser.page.evaluate("""() => {
                    const btns = [...document.querySelectorAll('button')];
                    for (const btn of btns) {
                        const t = (btn.textContent || '').trim().toLowerCase();
                        if ((t === 'solve' || t.includes('check') || t.includes('verify')) && !btn.disabled) {
                            btn.click(); break;
                        }
                    }
                }""")
                await asyncio.sleep(0.8)

            # Step 5: Check for new codes
            codes_after = set(await self.browser.page.evaluate("""() => {
                const text = document.body.textContent || '';
                return [...new Set((text.match(/\\b[A-Z0-9]{6}\\b/g) || []))];
            }"""))
            from dom_parser import FALSE_POSITIVES
            new_codes = codes_after - codes_before
            fresh = [c for c in new_codes if c not in FALSE_POSITIVES]
            if fresh:
                sorted_fresh = self._sort_codes_by_priority(fresh)
                print(f"  Force-reset puzzle: fresh code: {sorted_fresh[0]} (all: {sorted_fresh})", flush=True)
                return sorted_fresh[0]

            print(f"  Force-reset puzzle: no new codes generated", flush=True)
            return None
        except Exception as e:
            print(f"  Force-reset puzzle error: {e}", flush=True)
            return None

    async def _hide_stuck_modals(self) -> int:
        """Hide any modals that are blocking the page after failed radio attempts."""
        return await self.browser.page.evaluate("""() => {
            let hidden = 0;
            // Hide radio/option modals (fixed overlays)
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
            // Also hide any fixed overlay that blocks interaction
            document.querySelectorAll('.fixed.inset-0').forEach(el => {
                if (el.querySelector('input[type="radio"]') || el.querySelector('[role="radio"]')) {
                    el.style.display = 'none';
                    hidden++;
                }
            });
            return hidden;
        }""")

    async def _try_mouse_drag_and_drop(self) -> bool:
        """Drag pieces into slots using Playwright mouse (fallback when JS DragEvent fails)."""
        try:
            for round_num in range(6):
                state = await self.browser.page.evaluate("""() => {
                    const text = document.body.textContent || '';
                    const match = text.match(/(\\d+)\\/(\\d+)\\s*filled/);
                    const filled = match ? parseInt(match[1]) : 0;
                    if (filled >= 6) return {filled, done: true};

                    // Find empty slots
                    const emptySlots = [...document.querySelectorAll('div')].filter(el => {
                        const t = (el.textContent || '').trim();
                        return t.match(/^Slot \\d+$/) &&
                               ((el.getAttribute('class') || '').includes('dashed') || (el.getAttribute('style') || '').includes('dashed'));
                    }).map(el => {
                        el.scrollIntoView({behavior: 'instant', block: 'center'});
                        const rect = el.getBoundingClientRect();
                        return {x: rect.x + rect.width/2, y: rect.y + rect.height/2};
                    });

                    // Find available pieces NOT inside drop zones
                    const dropZones = [...document.querySelectorAll('[class*="border-dashed"]')];
                    const dropZoneSet = new Set(dropZones);
                    const pieces = [...document.querySelectorAll('[draggable="true"]')].filter(el => {
                        // Skip if this piece is inside a drop zone (already placed)
                        let parent = el.parentElement;
                        while (parent) {
                            if (dropZoneSet.has(parent)) return false;
                            parent = parent.parentElement;
                        }
                        return el.offsetParent !== null;
                    }).map(el => {
                        const rect = el.getBoundingClientRect();
                        return {x: rect.x + rect.width/2, y: rect.y + rect.height/2, text: el.textContent.trim()};
                    });
                    return {filled, done: false, emptySlots, pieces: pieces.slice(0, 6)};
                }""")
                if state.get('done') or state.get('filled', 0) >= 6:
                    print(f"  Drag: all slots filled!", flush=True)
                    # Click Complete/Done
                    await self.browser.page.evaluate("""() => {
                        document.querySelectorAll('button').forEach(btn => {
                            const t = (btn.textContent || '').trim().toLowerCase();
                            if ((t.includes('complete') || t.includes('done') || t.includes('verify')) &&
                                !t.includes('clear') && btn.offsetParent && !btn.disabled) btn.click();
                        });
                    }""")
                    return True

                slots = state.get('emptySlots', [])
                pieces = state.get('pieces', [])
                if not slots or not pieces:
                    break

                piece = pieces[0]
                slot = slots[0]
                await self.browser.page.mouse.move(piece['x'], piece['y'])
                await self.browser.page.mouse.down()
                await asyncio.sleep(0.05)
                await self.browser.page.mouse.move(slot['x'], slot['y'], steps=15)
                await asyncio.sleep(0.05)
                await self.browser.page.mouse.up()
                print(f"  Drag: moved '{piece.get('text', '?')}' to slot (round {round_num+1})", flush=True)
                await asyncio.sleep(0.3)
            return False
        except Exception as e:
            print(f"  Drag error: {e}", flush=True)
            return False

    async def _try_scroll_to_find_nav(self, codes_to_try: list[str] | None = None, deep_scroll: bool = True) -> bool:
        """Handle 'Scroll Down to Find Navigation' - scroll to find hidden submit/nav button.

        Phase 1: Scroll through page, click SAFE_WORDS buttons at each position.
        Phase 2: Find outlier buttons (rare labels) and click them.
        Phase 3: Fast full-page scan - scroll to trigger rendering, then click ALL buttons.
        """
        try:
            print(f"  Scroll-to-find: searching...", flush=True)
            SCROLL_TIMEOUT = 25  # Max seconds for entire scroll-to-find attempt
            scroll_global_start = time.time()

            def _scroll_time_left():
                return SCROLL_TIMEOUT - (time.time() - scroll_global_start)

            def _scroll_expired():
                return time.time() - scroll_global_start > SCROLL_TIMEOUT

            # Sort codes by priority: mixed alpha+digit first (most likely real codes)
            sorted_codes = self._sort_codes_by_priority(codes_to_try) if codes_to_try else []

            # Fill BEST code in input (mixed alpha+digit preferred over all-alpha like PROXIM)
            async def _fill_best_code(code_override=None):
                c = code_override or (sorted_codes[0] if sorted_codes else None)
                if c:
                    await self.browser.page.evaluate(f"""() => {{
                        const inp = document.querySelector('input[placeholder*="code" i], input[type="text"]');
                        if (inp) {{
                            const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
                            s.call(inp, '{c}');
                            inp.dispatchEvent(new Event('input', {{bubbles: true}}));
                            inp.dispatchEvent(new Event('change', {{bubbles: true}}));
                        }}
                    }}""")
                return c

            best_code = await _fill_best_code()

            TRAP_WORDS = {'proceed', 'continue', 'next step', 'next page', 'next section',
                          'go to next', 'click here', 'go forward', 'advance'}

            # ===== Phase 0: Mouse.wheel scrolling + DOM diffing =====
            # window.scrollTo() may NOT fire wheel/scroll event listeners.
            # mouse.wheel() fires real wheel events for dynamic DOM injection.
            print(f"  Scroll-to-find: phase 0 - mouse.wheel + DOM diffing...", flush=True)
            await self.browser.page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.2)

            baseline_interactive = await self.browser.page.evaluate(
                "() => document.querySelectorAll('button, a, [role=\"button\"], [tabindex]').length"
            )
            prev_scroll_y = 0
            phase0_start = time.time()
            phase0_codes = set()  # Accumulate codes during scroll

            while time.time() - phase0_start < 10:
                await self.browser.page.mouse.wheel(0, 800)
                await asyncio.sleep(0.10)

                # Check auto-navigation (IntersectionObserver pattern)
                url = await self.browser.get_url()
                if self._check_progress(url, self.current_step):
                    print(f"  Phase 0: auto-navigation detected!", flush=True)
                    return True

                # Accumulate codes from current viewport (virtualized content)
                vp_codes = await self.browser.page.evaluate("""() => {
                    const text = document.body.innerText || '';
                    return (text.match(/\\b[A-Z0-9]{6}\\b/g) || []).filter((v,i,a) => a.indexOf(v) === i);
                }""")
                phase0_codes.update(vp_codes)

                # Detect bottom (scrollY stopped increasing)
                cur_y = await self.browser.page.evaluate("() => window.scrollY")
                if cur_y <= prev_scroll_y and prev_scroll_y > 100:
                    # At bottom - extra wheel events in case threshold is exactly at bottom
                    for _ in range(5):
                        await self.browser.page.mouse.wheel(0, 800)
                        await asyncio.sleep(0.15)
                        url = await self.browser.get_url()
                        if self._check_progress(url, self.current_step):
                            print(f"  Phase 0: auto-nav at bottom!", flush=True)
                            return True
                    break
                prev_scroll_y = cur_y

                # DOM diffing: new interactive elements injected?
                cur_interactive = await self.browser.page.evaluate(
                    "() => document.querySelectorAll('button, a, [role=\"button\"], [tabindex]').length"
                )
                if cur_interactive > baseline_interactive:
                    # New interactive elements! Find them near current viewport
                    new_els = await self.browser.page.evaluate("""() => {
                        const vh = window.innerHeight;
                        const sel = 'button, a, [role="button"], [tabindex], [onclick]';
                        const standard = [...document.querySelectorAll(sel)];
                        const pointers = [...document.querySelectorAll('div, span, p')].filter(el =>
                            window.getComputedStyle(el).cursor === 'pointer' && !el.querySelector('button, a')
                        );
                        return [...standard, ...pointers].filter(el => {
                            if (el.closest('.fixed') || el.disabled) return false;
                            const r = el.getBoundingClientRect();
                            return r.top >= vh * 0.3 && r.top < vh + 50 && r.width > 10 && r.height > 10;
                        }).map(el => {
                            const r = el.getBoundingClientRect();
                            return {x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2),
                                    text: (el.textContent || '').trim().substring(0, 40), tag: el.tagName};
                        });
                    }""")
                    if new_els:
                        print(f"  Phase 0: {cur_interactive - baseline_interactive} new interactive els at scrollY={cur_y}", flush=True)
                        for el in new_els:
                            await self._clear_popups()
                            try:
                                await self.browser.page.mouse.click(el['x'], el['y'])
                                await asyncio.sleep(0.1)
                            except Exception:
                                pass
                            url = await self.browser.get_url()
                            if self._check_progress(url, self.current_step):
                                print(f"  Phase 0: '{el['text']}' ({el['tag']}) WORKED!", flush=True)
                                return True
                    baseline_interactive = cur_interactive

            # Try accumulated codes from Phase 0 scrolling
            LATIN = {'BEATAE','LABORE','DOLORE','VENIAM','NOSTRU','ALIQUA','EXERCI',
                     'TEMPOR','INCIDI','LABORI','MAGNAM','VOLUPT','SAPIEN','FUGIAT',
                     'COMMOD','EXCEPT','OFFICI','MOLLIT','PROIDE','REPUDI','FILLER',
                     'SCROLL','HIDDEN','BUTTON','SUBMIT','OPTION','CHOICE','REVEAL',
                     'PUZZLE','CANVAS','STROKE','SECOND','MEMORY','LOADED','BLOCKS',
                     'CHANGE','DELETE','CREATE','SEARCH','FILTER','NOTICE','STATUS',
                     'RESULT','OUTPUT','INPUTS','BEFORE','LAYOUT','RENDER','EFFECT',
                     'TOGGLE','HANDLE','CUSTOM','STRING','NUMBER','PROMPT','GLOBAL',
                     'MODULE','SHOULD','COOKIE','MOVING','FILLED','PIECES','VERIFY',
                     'DEVICE','SCREEN','MOBILE','TABLET','SELECT','PLEASE','SIMPLE',
                     'NEEDED','EXTEND','RANDOM','ACTIVE','PLAYED','ESCAPE','ALMOST',
                     'INSIDE','SOLVED','CENTER','BOTTOM','SHADOW','CURSOR','ROTATE',
                     'COLORS','IMAGES','CANCEL','RETURN','UPDATE','ALERTS','ERRORS'}
            p0_new = [c for c in phase0_codes if c not in LATIN and not c.isdigit()
                      and c not in (codes_to_try or [])
                      and not re.match(r'^\d+(?:PX|VH|VW|EM|REM|MS|FR)$', c)]
            p0_new.sort(key=lambda c: (c.isalpha(), c))
            if p0_new:
                print(f"  Phase 0 accumulated codes: {p0_new[:10]}", flush=True)
                for code in p0_new[:5]:
                    if await self._fill_and_submit(code, self.current_step):
                        return True

            # ===== Phase 0-deep: React state + CSS + shadow DOM + JS var extraction =====
            if not _scroll_expired():
                deep_codes = await self._deep_code_extraction(codes_to_try)
            else:
                deep_codes = []
            if deep_codes:
                print(f"  Phase 0-deep: {len(deep_codes)} codes from React/CSS/JS: {deep_codes[:8]}", flush=True)
                for code in deep_codes[:10]:
                    if await self._fill_and_submit(code, self.current_step):
                        print(f"  Phase 0-deep: code '{code}' WORKED!", flush=True)
                        return True

            # ===== Phase 0-slow: Slow incremental scroll with pauses =====
            # Some IntersectionObservers need the element visible for a minimum duration.
            if not _scroll_expired():
                print(f"  Scroll-to-find: phase 0-slow - incremental scroll...", flush=True)
                await _fill_best_code()  # Ensure best code is in input
                await self.browser.page.evaluate("window.scrollTo(0, 0)")
                await asyncio.sleep(0.2)
                total_h = await self.browser.page.evaluate("() => document.body.scrollHeight")
                slow_step = 400
                slow_start = time.time()
                prev_y = 0
                for pos in range(0, total_h + slow_step, slow_step):
                    if time.time() - slow_start > 8 or _scroll_expired():
                        break
                    await self.browser.page.mouse.wheel(0, slow_step)
                    await asyncio.sleep(0.25)
                    url = await self.browser.get_url()
                    if self._check_progress(url, self.current_step):
                        print(f"  Phase 0-slow: auto-nav at ~{pos}px!", flush=True)
                        return True
                    cur_y = await self.browser.page.evaluate("() => window.scrollY")
                    if cur_y <= prev_y and prev_y > 100:
                        break
                    prev_y = cur_y

            # After slow scroll, deep extract again (scroll may have populated React state)
            if not _scroll_expired():
                deep2 = await self._deep_code_extraction(codes_to_try)
                new_deep = [c for c in deep2 if c not in (deep_codes or [])]
                if new_deep:
                    print(f"  Phase 0-slow: {len(new_deep)} NEW codes after scroll: {new_deep[:5]}", flush=True)
                    for code in new_deep[:5]:
                        if await self._fill_and_submit(code, self.current_step):
                            return True

            # ===== Phase 0-sections: scrollIntoView each section =====
            if not _scroll_expired():
                await _fill_best_code()  # Re-fill best code
                section_count = await self.browser.page.evaluate("""() => {
                    return [...document.querySelectorAll('div')].filter(el => {
                        const t = (el.textContent || '').trim();
                        return t.match(/^Section \\d+/) && t.length > 30;
                    }).length;
                }""")
                if section_count > 10:
                    print(f"  Phase 0-sections: scrolling {section_count} sections...", flush=True)
                    sec_start = time.time()
                    for i in range(section_count):
                        if time.time() - sec_start > 6 or _scroll_expired():
                            break
                        await self.browser.page.evaluate(f"""(idx) => {{
                            const secs = [...document.querySelectorAll('div')].filter(el => {{
                                const t = (el.textContent || '').trim();
                                return t.match(/^Section \\d+/) && t.length > 30;
                            }});
                            if (idx < secs.length) secs[idx].scrollIntoView({{behavior: 'smooth', block: 'center'}});
                        }}""", i)
                        await asyncio.sleep(0.15)
                        url = await self.browser.get_url()
                        if self._check_progress(url, self.current_step):
                            print(f"  Phase 0-sections: auto-nav at section {i}!", flush=True)
                            return True

            # ===== Phase 0a: Scrollable containers =====
            if _scroll_expired():
                await self.browser.page.evaluate("window.scrollTo(0, 0)")
                return False
            # The scroll listener might be on a nested div, not the window
            containers = await self.browser.page.evaluate("""() => {
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
                        scrollable: el.scrollHeight - el.clientHeight,
                        tag: el.tagName,
                        cls: (el.getAttribute('class') || '').substring(0, 60)
                    });
                }
                return results;
            }""")
            if containers:
                print(f"  Phase 0a: {len(containers)} scrollable containers found", flush=True)
                for cont in containers[:3]:
                    print(f"    Container: {cont['tag']}.{cont['cls'][:30]} scrollable={cont['scrollable']}px", flush=True)
                    # Move mouse to container center, then wheel-scroll it
                    await self.browser.page.mouse.move(cont['x'], cont['y'])
                    await asyncio.sleep(0.05)
                    scroll_remaining = cont['scrollable']
                    while scroll_remaining > 0:
                        await self.browser.page.mouse.wheel(0, 500)
                        scroll_remaining -= 500
                        await asyncio.sleep(0.10)
                        url = await self.browser.get_url()
                        if self._check_progress(url, self.current_step):
                            print(f"  Phase 0a: container scroll WORKED!", flush=True)
                            return True
                    # Check for new elements in container after scrolling
                    cont_els = await self.browser.page.evaluate("""() => {
                        const vh = window.innerHeight;
                        const sel = 'button, a, [role="button"], [tabindex], [onclick]';
                        return [...document.querySelectorAll(sel)].filter(el => {
                            if (el.closest('.fixed') || el.disabled) return false;
                            const r = el.getBoundingClientRect();
                            return r.top >= 0 && r.top < vh && r.width > 10 && r.height > 10;
                        }).map(el => {
                            const r = el.getBoundingClientRect();
                            return {x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2),
                                    text: (el.textContent || '').trim().substring(0, 40)};
                        });
                    }""")
                    for el in cont_els[:10]:
                        if el['text'].lower().strip() in TRAP_WORDS:
                            continue
                        await self._clear_popups()
                        try:
                            await self.browser.page.mouse.click(el['x'], el['y'])
                            await asyncio.sleep(0.08)
                        except Exception:
                            pass
                        url = await self.browser.get_url()
                        if self._check_progress(url, self.current_step):
                            print(f"  Phase 0a: container element '{el['text']}' WORKED!", flush=True)
                            return True

            # After reaching bottom via mouse.wheel, scan for non-standard elements
            bottom_scan = await self.browser.page.evaluate("""() => {
                const vh = window.innerHeight;
                const sel = 'button, a, [role="button"], [tabindex], [onclick]';
                const standard = [...document.querySelectorAll(sel)];
                const pointers = [...document.querySelectorAll('div, span, p, li')].filter(el => {
                    const s = window.getComputedStyle(el);
                    return (s.cursor === 'pointer' || el.hasAttribute('tabindex'))
                        && !el.querySelector('button, a');
                });
                const reactEls = [...document.querySelectorAll('div, span')].filter(el => {
                    const pk = Object.keys(el).find(k => k.startsWith('__reactProps$'));
                    return pk && el[pk] && el[pk].onClick && !el.querySelector('button, a');
                });
                return [...new Set([...standard, ...pointers, ...reactEls])].filter(el => {
                    if (el.closest('.fixed') || el.disabled) return false;
                    const r = el.getBoundingClientRect();
                    return r.top >= 0 && r.top < vh && r.width > 10 && r.height > 10;
                }).map(el => {
                    const r = el.getBoundingClientRect();
                    return {x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2),
                            text: (el.textContent || '').trim().substring(0, 40), tag: el.tagName};
                });
            }""")
            non_trap_first = sorted(bottom_scan, key=lambda e: e['text'].lower().strip() in TRAP_WORDS)
            for el in non_trap_first[:20]:
                await self._clear_popups()
                try:
                    await self.browser.page.mouse.click(el['x'], el['y'])
                    await asyncio.sleep(0.08)
                except Exception:
                    pass
                url = await self.browser.get_url()
                if self._check_progress(url, self.current_step):
                    print(f"  Phase 0 bottom: '{el['text']}' ({el['tag']}) WORKED!", flush=True)
                    return True

            # ===== Phase 0b: Keyboard scrolling =====
            if _scroll_expired():
                await self.browser.page.evaluate("window.scrollTo(0, 0)")
                return False
            # End key fires both keyboard AND scroll events; PageDown fires scroll events
            print(f"  Scroll-to-find: phase 0b - keyboard scrolling...", flush=True)
            await self.browser.page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.1)
            await self.browser.page.keyboard.press("End")
            await asyncio.sleep(0.5)
            url = await self.browser.get_url()
            if self._check_progress(url, self.current_step):
                print(f"  Phase 0b: End key WORKED!", flush=True)
                return True
            # Check for newly injected elements after End key
            end_els = await self.browser.page.evaluate("""() => {
                const vh = window.innerHeight;
                const sel = 'button, a, [role="button"], [tabindex], [onclick]';
                const pointers = [...document.querySelectorAll('div, span')].filter(el =>
                    window.getComputedStyle(el).cursor === 'pointer' && !el.querySelector('button, a')
                );
                return [...document.querySelectorAll(sel), ...pointers].filter(el => {
                    if (el.closest('.fixed') || el.disabled) return false;
                    const r = el.getBoundingClientRect();
                    return r.top >= 0 && r.top < vh && r.width > 10 && r.height > 10;
                }).map(el => {
                    const r = el.getBoundingClientRect();
                    return {x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2),
                            text: (el.textContent || '').trim().substring(0, 40), tag: el.tagName};
                });
            }""")
            for el in sorted(end_els, key=lambda e: e['text'].lower().strip() in TRAP_WORDS)[:15]:
                await self._clear_popups()
                try:
                    await self.browser.page.mouse.click(el['x'], el['y'])
                    await asyncio.sleep(0.08)
                except Exception:
                    pass
                url = await self.browser.get_url()
                if self._check_progress(url, self.current_step):
                    print(f"  Phase 0b: '{el['text']}' after End key WORKED!", flush=True)
                    return True

            # PageDown through entire page
            await self.browser.page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.1)
            for _ in range(80):
                await self.browser.page.keyboard.press("PageDown")
                await asyncio.sleep(0.06)
                url = await self.browser.get_url()
                if self._check_progress(url, self.current_step):
                    print(f"  Phase 0b: PageDown WORKED!", flush=True)
                    return True

            # ===== Phase 0c: Synthetic event dispatch =====
            if _scroll_expired():
                await self.browser.page.evaluate("window.scrollTo(0, 0)")
                return False
            # Dispatch scroll AND wheel events on all targets after programmatic scroll
            print(f"  Scroll-to-find: phase 0c - synthetic events...", flush=True)
            await self.browser.page.evaluate("""() => {
                window.scrollTo(0, document.body.scrollHeight);
                for (const target of [window, document, document.documentElement, document.body]) {
                    target.dispatchEvent(new Event('scroll', {bubbles: true}));
                    target.dispatchEvent(new WheelEvent('wheel', {deltaY: 500, bubbles: true}));
                }
            }""")
            await asyncio.sleep(0.3)
            url = await self.browser.get_url()
            if self._check_progress(url, self.current_step):
                print(f"  Phase 0c: synthetic events WORKED!", flush=True)
                return True
            # Check for elements injected by synthetic events
            post_synth = await self.browser.page.evaluate("""() => {
                const vh = window.innerHeight;
                const sel = 'button, a, [role="button"], [tabindex], [onclick]';
                const pointers = [...document.querySelectorAll('div, span')].filter(el =>
                    window.getComputedStyle(el).cursor === 'pointer' && !el.querySelector('button, a')
                );
                return [...document.querySelectorAll(sel), ...pointers].filter(el => {
                    if (el.closest('.fixed') || el.disabled) return false;
                    const r = el.getBoundingClientRect();
                    return r.top >= 0 && r.top < vh && r.width > 10 && r.height > 10;
                }).map(el => {
                    const r = el.getBoundingClientRect();
                    return {x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2),
                            text: (el.textContent || '').trim().substring(0, 40), tag: el.tagName};
                });
            }""")
            for el in sorted(post_synth, key=lambda e: e['text'].lower().strip() in TRAP_WORDS)[:15]:
                await self._clear_popups()
                try:
                    await self.browser.page.mouse.click(el['x'], el['y'])
                    await asyncio.sleep(0.08)
                except Exception:
                    pass
                url = await self.browser.get_url()
                if self._check_progress(url, self.current_step):
                    print(f"  Phase 0c: '{el['text']}' WORKED!", flush=True)
                    return True

            # ===== Existing fallback phases (scrollTo-based) =====
            if _scroll_expired():
                await self.browser.page.evaluate("window.scrollTo(0, 0)")
                return False
            total_h = await self.browser.page.evaluate("() => document.body.scrollHeight")
            step_px = 800

            # Phase 1: Scroll and click SAFE_WORDS buttons (fast)
            for pos in range(0, total_h + step_px, step_px):
                await self.browser.page.evaluate(f"window.scrollTo(0, {pos})")
                await asyncio.sleep(0.05)

                btn_results = await self.browser.page.evaluate("""() => {
                    const SAFE_WORDS = ['next', 'submit', 'go', '→', 'navigate', 'enter'];
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
                        if (t === '×' || t === 'X' || t === '✕') continue;
                        if (SAFE_WORDS.some(w => tl === w || (tl.includes(w) && tl.length < 15))) {
                            results.push({text: t, idx: btns.indexOf(btn)});
                        }
                    }
                    return results;
                }""")

                for btn in btn_results:
                    await self._clear_popups()
                    await self.browser.page.evaluate(f"(idx) => [...document.querySelectorAll('button, a')][idx]?.click()", btn['idx'])
                    await asyncio.sleep(0.1)
                    url = await self.browser.get_url()
                    if self._check_progress(url, self.current_step):
                        print(f"  Scroll-to-find: button '{btn['text']}' at scroll {pos}px WORKED!", flush=True)
                        return True

            # Phase 2: Find outlier buttons (rare labels among many similar ones)
            outlier_result = await self.browser.page.evaluate("""() => {
                const btns = [...document.querySelectorAll('button')].filter(b => {
                    if (!b.offsetParent || b.disabled || b.closest('.fixed')) return false;
                    const t = b.textContent.trim();
                    return t.length > 0 && t.length < 40 && t !== '×' && t !== 'X' && t !== '✕';
                });
                if (btns.length < 5) return null;
                const freq = {};
                btns.forEach(b => {
                    const label = b.textContent.trim().toLowerCase();
                    freq[label] = (freq[label] || 0) + 1;
                });
                const outliers = btns.filter(b => {
                    const label = b.textContent.trim().toLowerCase();
                    return freq[label] <= 2;
                });
                return outliers.map((b, i) => ({
                    text: b.textContent.trim(),
                    idx: [...document.querySelectorAll('button')].indexOf(b)
                }));
            }""")

            if outlier_result:
                print(f"  Scroll-to-find: found {len(outlier_result)} outlier buttons", flush=True)
                for btn in outlier_result:
                    await self._clear_popups()
                    await self.browser.page.evaluate(f"""(idx) => {{
                        const btn = document.querySelectorAll('button')[idx];
                        if (btn) {{ btn.scrollIntoView({{behavior: 'instant', block: 'center'}}); btn.click(); }}
                    }}""", btn['idx'])
                    await asyncio.sleep(0.12)
                    url = await self.browser.get_url()
                    if self._check_progress(url, self.current_step):
                        print(f"  Scroll-to-find: outlier '{btn['text']}' WORKED!", flush=True)
                        return True

            if not deep_scroll:
                await self.browser.page.evaluate("window.scrollTo(0, 0)")
                return False

            # Phase 3: Fast full-page button scan
            # Step A: Thorough scroll with mouse.wheel to trigger rendering + fire events
            if _scroll_expired():
                await self.browser.page.evaluate("window.scrollTo(0, 0)")
                return False
            await _fill_best_code()  # Ensure correct code in input before clicking buttons
            total_h = await self.browser.page.evaluate("() => document.body.scrollHeight")
            await self.browser.page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.05)
            for pos in range(0, total_h + 1000, 1000):
                await self.browser.page.mouse.wheel(0, 1000)
                await asyncio.sleep(0.05)
            # Re-check height (virtualized content may have expanded)
            new_h = await self.browser.page.evaluate("() => document.body.scrollHeight")
            if new_h > total_h + 500:
                for pos in range(total_h, new_h + 1000, 1000):
                    await self.browser.page.mouse.wheel(0, 1000)
                    await asyncio.sleep(0.05)
            # Dispatch synthetic events at bottom for good measure
            await self.browser.page.evaluate("""() => {
                for (const target of [window, document, document.documentElement, document.body]) {
                    target.dispatchEvent(new Event('scroll', {bubbles: true}));
                    target.dispatchEvent(new WheelEvent('wheel', {deltaY: 500, bubbles: true}));
                }
            }""")
            await self.browser.page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.1)

            # Step B: Get ALL buttons from entire page (they should all be rendered now)
            all_btns = await self.browser.page.evaluate("""() => {
                const btns = [...document.querySelectorAll('button, a')].filter(el => {
                    if (el.disabled || el.closest('.fixed')) return false;
                    const t = (el.textContent || '').trim();
                    return t.length > 0 && t.length < 40 && t !== '×' && t !== 'X' && t !== '✕';
                });
                return btns.map((b, i) => ({text: b.textContent.trim(), idx: i}));
            }""")
            print(f"  Scroll-to-find: phase 3 clicking {len(all_btns)} buttons...", flush=True)

            # Step C: Click each button with popup clearing (in batches of 5 for speed)
            batch_size = 5
            for start in range(0, len(all_btns), batch_size):
                end = min(start + batch_size, len(all_btns))
                await self.browser.page.evaluate(f"""() => {{
                    const start = {start}, end = {end};
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
                        return t.length > 0 && t.length < 40 && t !== '×' && t !== 'X' && t !== '✕';
                    }});
                    for (let i = start; i < Math.min(end, allBtns.length); i++) {{
                        clearP();
                        allBtns[i].scrollIntoView({{behavior: 'instant', block: 'center'}});
                        allBtns[i].click();
                    }}
                }}""")
                await asyncio.sleep(0.1)
                url = await self.browser.get_url()
                if self._check_progress(url, self.current_step):
                    print(f"  Scroll-to-find: phase 3 batch {start}-{end} WORKED!", flush=True)
                    return True

            # Step D: Extract codes from full-page scroll (codes may be hidden in filler sections)
            html_after = await self.browser.get_html()
            scroll_codes = extract_hidden_codes(html_after)
            new_scroll_codes = [c for c in scroll_codes if c not in (codes_to_try or [])]
            if new_scroll_codes:
                print(f"  Scroll-to-find: found new codes during scroll: {new_scroll_codes}", flush=True)
                for code in new_scroll_codes[:3]:
                    if await self._fill_and_submit(code, self.current_step):
                        return True

            # Step E: Try pressing Enter with each code (best codes first)
            if sorted_codes:
                for code in sorted_codes[:3]:
                    await self.browser.page.evaluate("window.scrollTo(0, 0)")
                    await asyncio.sleep(0.05)
                    inp = self.browser.page.locator('input[placeholder*="code" i], input[type="text"]').first
                    try:
                        await inp.click(timeout=1000)
                        await inp.fill(code)
                        await self.browser.page.keyboard.press("Enter")
                        await asyncio.sleep(0.3)
                        url = await self.browser.get_url()
                        if self._check_progress(url, self.current_step):
                            print(f"  Scroll-to-find: Enter with code {code} WORKED!", flush=True)
                            return True
                    except Exception:
                        pass

            # Phase 4: Playwright-native clicks + code accumulation during scroll
            # Extract codes at each scroll position (virtualized content only exists in viewport)
            if _scroll_expired():
                await self.browser.page.evaluate("window.scrollTo(0, 0)")
                return False
            await _fill_best_code()  # Ensure correct code in input
            print(f"  Scroll-to-find: phase 4 - Playwright native clicks...", flush=True)
            total_h = await self.browser.page.evaluate("() => document.body.scrollHeight")
            clicked = 0
            all_labels = []
            accumulated_codes = set()
            phase4_start = time.time()
            phase4_limit = min(12, _scroll_time_left())  # Reduced from 25s
            await self.browser.page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.1)
            scroll_pos = 0
            while scroll_pos < total_h + 800:
                if time.time() - phase4_start > phase4_limit:
                    print(f"  Scroll-to-find: phase 4 time limit ({clicked} clicks)", flush=True)
                    break
                # Use mouse.wheel for real event firing
                await self.browser.page.mouse.wheel(0, 800)
                scroll_pos += 800
                await asyncio.sleep(0.1)
                # Extract codes from current viewport text (catches virtualized content)
                viewport_codes = await self.browser.page.evaluate("""() => {
                    const text = document.body.innerText || '';
                    const codes = text.match(/\\b[A-Z0-9]{6}\\b/g) || [];
                    return [...new Set(codes)];
                }""")
                accumulated_codes.update(viewport_codes)
                # Get interactive elements visible in current viewport
                visible_btns = await self.browser.page.evaluate("""() => {
                    const sel = 'button, a, [role="button"], [class*="cursor-pointer"], [onclick]';
                    const els = [...document.querySelectorAll(sel)].filter(el => {
                        if (el.closest('.fixed')) return false;
                        if (el.disabled) return false;
                        const rect = el.getBoundingClientRect();
                        if (rect.top < -10 || rect.top > window.innerHeight + 10) return false;
                        if (rect.width < 10 || rect.height < 10) return false;
                        const t = (el.textContent || '').trim();
                        if (t.length === 0 || t.length > 60) return false;
                        if (t === '×' || t === 'X' || t === '✕') return false;
                        return true;
                    });
                    return els.map(el => {
                        const rect = el.getBoundingClientRect();
                        return {
                            text: (el.textContent || '').trim().substring(0, 40),
                            x: Math.round(rect.x + rect.width / 2),
                            y: Math.round(rect.y + rect.height / 2)
                        };
                    });
                }""")
                for btn in visible_btns:
                    all_labels.append(btn['text'])
                    await self._clear_popups()
                    try:
                        await self.browser.page.mouse.click(btn['x'], btn['y'])
                        clicked += 1
                        await asyncio.sleep(0.05)
                    except Exception:
                        pass
                url = await self.browser.get_url()
                if self._check_progress(url, self.current_step):
                    print(f"  Scroll-to-find: phase 4 WORKED after {clicked} clicks (scroll {scroll_pos}px)!", flush=True)
                    return True
            # Debug info
            from collections import Counter
            label_counts = Counter(all_labels)
            unique_labels = [l for l, c in label_counts.items() if c <= 2]
            print(f"  Phase 4: {clicked} clicks, {len(label_counts)} unique labels. Rare: {unique_labels[:10]}", flush=True)

            # Try ALL accumulated codes from scrolling (some only exist when their section is in viewport)
            LATIN = {'BEATAE','LABORE','DOLORE','VENIAM','NOSTRU','ALIQUA','EXERCI',
                     'TEMPOR','INCIDI','LABORI','MAGNAM','VOLUPT','SAPIEN','FUGIAT',
                     'COMMOD','EXCEPT','OFFICI','MOLLIT','PROIDE','REPUDI','FILLER',
                     'SCROLL','HIDDEN','BUTTON','SUBMIT','OPTION','CHOICE','REVEAL',
                     'PUZZLE','CANVAS','STROKE','SECOND','MEMORY','LOADED','BLOCKS',
                     'CHANGE','DELETE','CREATE','SEARCH','FILTER','NOTICE','STATUS',
                     'RESULT','OUTPUT','INPUTS','BEFORE','LAYOUT','RENDER','EFFECT',
                     'TOGGLE','HANDLE','CUSTOM','STRING','NUMBER','PROMPT','GLOBAL',
                     'MODULE','SHOULD','COOKIE','MOVING','FILLED','PIECES','VERIFY',
                     'DEVICE','SCREEN','MOBILE','TABLET','SELECT','PLEASE','SIMPLE',
                     'NEEDED','EXTEND','RANDOM','ACTIVE','PLAYED','ESCAPE','ALMOST',
                     'INSIDE','SOLVED','CENTER','BOTTOM','SHADOW','CURSOR','ROTATE',
                     'COLORS','IMAGES','CANCEL','RETURN','UPDATE','ALERTS','ERRORS'}
            new_accumulated = [c for c in accumulated_codes
                             if c not in LATIN and not c.isdigit()
                             and c not in (codes_to_try or [])
                             and not re.match(r'^\d+(?:PX|VH|VW|EM|REM|MS|FR)$', c)]
            # Sort: codes with digits first (more likely real), then all-letter codes
            new_accumulated.sort(key=lambda c: (c.isalpha(), c))
            if new_accumulated:
                print(f"  Phase 4 accumulated codes: {new_accumulated}", flush=True)
                for code in new_accumulated[:5]:
                    if await self._fill_and_submit(code, self.current_step):
                        return True

            # Phase 5: Try hidden navigation - forms, links, direct URL manipulation
            # Some scroll challenges have a hidden link or form at the very bottom
            await self.browser.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(0.3)
            # Look for any navigation links with href patterns
            nav_found = await self.browser.page.evaluate(f"""() => {{
                // Check for hidden links with step-related hrefs
                const links = [...document.querySelectorAll('a[href]')];
                for (const a of links) {{
                    const href = a.getAttribute('href') || '';
                    if (href.includes('step') || href.includes('/{self.current_step + 1}') ||
                        href.includes('challenge') || href.match(/\\/\\d+$/)) {{
                        a.click();
                        return 'link: ' + href;
                    }}
                }}
                // Check for forms
                const forms = document.querySelectorAll('form');
                for (const form of forms) {{
                    if (form.action && form.action !== window.location.href) {{
                        form.submit();
                        return 'form: ' + form.action;
                    }}
                }}
                return null;
            }}""")
            if nav_found:
                print(f"  Scroll-to-find: phase 5 found {nav_found}", flush=True)
                await asyncio.sleep(0.5)
                url = await self.browser.get_url()
                if self._check_progress(url, self.current_step):
                    return True

            # Phase 6: Find non-standard React onClick elements (divs, spans, etc.)
            # Only scan non-button/non-link elements since Phase 4 covered those
            if not _scroll_expired():
                print(f"  Scroll-to-find: phase 6 - React onClick scan...", flush=True)
            await self.browser.page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.05)
            for scroll_pos in range(0, total_h + 1200, 1200):
                if _scroll_expired():
                    break
                await self.browser.page.mouse.wheel(0, 1200)
                await asyncio.sleep(0.08)
                react_btns = await self.browser.page.evaluate("""() => {
                    const results = [];
                    // Only check div/span/p/li - buttons and links already covered
                    const els = document.querySelectorAll('div, span, p, li, td, section');
                    for (const el of els) {
                        if (el.closest('.fixed')) continue;
                        const rect = el.getBoundingClientRect();
                        if (rect.top < -10 || rect.top > window.innerHeight + 10) continue;
                        if (rect.width < 20 || rect.height < 15) continue;
                        const propsKey = Object.keys(el).find(k => k.startsWith('__reactProps$'));
                        if (propsKey && el[propsKey] && el[propsKey].onClick) {
                            const t = (el.textContent || '').trim();
                            if (t === '×' || t === 'X' || t.length > 60) continue;
                            // Skip if this element contains a button/link child (already clicked)
                            if (el.querySelector('button, a')) continue;
                            results.push({
                                x: Math.round(rect.x + rect.width / 2),
                                y: Math.round(rect.y + rect.height / 2)
                            });
                        }
                    }
                    return results;
                }""")
                for btn in react_btns:
                    await self._clear_popups()
                    try:
                        await self.browser.page.mouse.click(btn['x'], btn['y'])
                    except Exception:
                        pass
                url = await self.browser.get_url()
                if self._check_progress(url, self.current_step):
                    print(f"  Scroll-to-find: phase 6 (React onClick) WORKED at scroll {scroll_pos}!", flush=True)
                    return True

            # Phase 7: Use Playwright locators (auto scroll-into-view, handles virtual lists)
            if _scroll_expired():
                await self.browser.page.evaluate("window.scrollTo(0, 0)")
                return False
            await _fill_best_code()  # Ensure correct code in input
            print(f"  Scroll-to-find: phase 7 - Playwright locator clicks...", flush=True)
            try:
                # Try clicking EVERY button on the page using Playwright locators
                all_buttons = self.browser.page.locator('button:not(:has-text("×")):not(:has-text("✕"))')
                count = await all_buttons.count()
                print(f"  Phase 7: {count} buttons found via locator", flush=True)
                phase7_clicked = 0
                for i in range(count):
                    if _scroll_expired():
                        break
                    try:
                        btn = all_buttons.nth(i)
                        # Check if it's inside a .fixed modal (skip those)
                        is_fixed = await btn.evaluate("el => !!el.closest('.fixed')")
                        if is_fixed:
                            continue
                        text = (await btn.text_content() or '').strip()
                        if len(text) > 60 or text in ('×', 'X', '✕'):
                            continue
                        await self._clear_popups()
                        await btn.click(timeout=500, force=True)
                        phase7_clicked += 1
                        url = await self.browser.get_url()
                        if self._check_progress(url, self.current_step):
                            print(f"  Phase 7: button '{text}' WORKED! ({phase7_clicked} clicks)", flush=True)
                            return True
                    except Exception:
                        pass
                print(f"  Phase 7: clicked {phase7_clicked}/{count}, none worked", flush=True)
            except Exception as e:
                print(f"  Phase 7 error: {e}", flush=True)

            # ===== LAST RESORT: Deep extraction + try other codes =====
            if not _scroll_expired():
                last_deep = await self._deep_code_extraction(codes_to_try)
            else:
                last_deep = []
            if last_deep:
                print(f"  LAST RESORT: deep extraction found {len(last_deep)} codes: {last_deep[:10]}", flush=True)
                for code in last_deep[:10]:
                    if await self._fill_and_submit(code, self.current_step):
                        print(f"  LAST RESORT: code '{code}' WORKED!", flush=True)
                        return True

            # Try alternate codes with a quick Phase 0 scroll
            alt_codes = [c for c in sorted_codes[1:] if c != best_code][:2]
            for alt_code in alt_codes:
                if _scroll_expired():
                    break
                print(f"  LAST RESORT: retrying with code '{alt_code}'...", flush=True)
                await _fill_best_code(alt_code)
                await self.browser.page.evaluate("window.scrollTo(0, 0)")
                await asyncio.sleep(0.1)
                prev_y = 0
                retry_start = time.time()
                while time.time() - retry_start < 5:
                    await self.browser.page.mouse.wheel(0, 800)
                    await asyncio.sleep(0.12)
                    url = await self.browser.get_url()
                    if self._check_progress(url, self.current_step):
                        print(f"  LAST RESORT: auto-nav with code '{alt_code}'!", flush=True)
                        return True
                    cur_y = await self.browser.page.evaluate("() => window.scrollY")
                    if cur_y <= prev_y and prev_y > 100:
                        break
                    prev_y = cur_y

            # React debug info
            react_debug = await self.browser.page.evaluate("""() => {
                const info = {hasReact: false, fiberRoots: 0, stateStrings: []};
                let count = 0;
                document.querySelectorAll('*').forEach(el => {
                    for (const key of Object.keys(el)) {
                        if (key.startsWith('__reactFiber$')) { info.hasReact = true; count++; }
                    }
                });
                info.fiberRoots = count;
                return info;
            }""")

            # DEBUG: dump page state when all phases fail
            debug_info = await self.browser.page.evaluate("""() => {
                const vh = window.innerHeight;
                const scrollH = document.body.scrollHeight;
                const allBtns = document.querySelectorAll('button').length;
                const allLinks = document.querySelectorAll('a').length;
                const allInputs = document.querySelectorAll('input').length;
                const allForms = document.querySelectorAll('form').length;
                const iframes = document.querySelectorAll('iframe').length;
                const canvases = document.querySelectorAll('canvas').length;
                // Check for scrollable containers
                const scrollable = [...document.querySelectorAll('div, section')].filter(el => {
                    const s = window.getComputedStyle(el);
                    return (s.overflow + s.overflowY).match(/auto|scroll/) && el.scrollHeight > el.clientHeight + 10;
                }).length;
                // Get visible text at bottom
                window.scrollTo(0, scrollH);
                const bottomText = document.body.innerText.substring(document.body.innerText.length - 500);
                // Check data attributes
                const dataEls = [...document.querySelectorAll('[data-code], [data-value], [data-step], [data-nav]')].map(
                    el => ({tag: el.tagName, attrs: [...el.attributes].map(a => a.name + '=' + a.value.substring(0, 30)).join(', ')})
                );
                return {scrollH, allBtns, allLinks, allInputs, allForms, iframes, canvases,
                        scrollable, bottomText: bottomText.substring(0, 300), dataEls: dataEls.slice(0, 5)};
            }""")
            print(f"  ALL PHASES FAILED. Debug: scrollH={debug_info['scrollH']}, btns={debug_info['allBtns']}, "
                  f"links={debug_info['allLinks']}, inputs={debug_info['allInputs']}, forms={debug_info['allForms']}, "
                  f"iframes={debug_info['iframes']}, canvases={debug_info['canvases']}, "
                  f"scrollableContainers={debug_info['scrollable']}", flush=True)
            print(f"  React: hasReact={react_debug.get('hasReact')}, fibers={react_debug.get('fiberRoots')}", flush=True)
            if debug_info['dataEls']:
                print(f"  Data-attr elements: {debug_info['dataEls']}", flush=True)
            print(f"  Bottom text: {debug_info['bottomText'][:200]}", flush=True)

            await self.browser.page.evaluate("window.scrollTo(0, 0)")
            return False
        except Exception as e:
            print(f"  Scroll-to-find error: {e}", flush=True)
            return False

    async def _try_audio_challenge(self) -> bool:
        """Handle Audio Challenge - force-end speech synthesis in headless Chromium."""
        try:
            # Reset capture state
            await self.browser.page.evaluate("""() => {
                window.__capturedSpeechTexts = window.__capturedSpeechTexts || [];
                window.__capturedSpeechUtterance = window.__capturedSpeechUtterance || null;
                window.__speechDone = false;
            }""")

            # Click Play Audio button (but NOT "Playing...")
            play_result = await self.browser.page.evaluate("""() => {
                const btns = [...document.querySelectorAll('button')];
                for (const btn of btns) {
                    const text = (btn.textContent || '').trim().toLowerCase();
                    if (text.includes('play') && !text.includes('playing') && btn.offsetParent && !btn.disabled) {
                        btn.click(); return 'clicked';
                    }
                }
                for (const btn of btns) {
                    const text = (btn.textContent || '').trim().toLowerCase();
                    if (text.includes('playing')) return 'already_playing';
                }
                return 'not_found';
            }""")
            if play_result == 'not_found':
                return False
            print(f"  Audio: {play_result}", flush=True)

            # Wait for speech synthesis to start, then force-end it
            await asyncio.sleep(3.0)

            # Force-end speech and dispatch 'end' event on captured utterance
            await self.browser.page.evaluate("""() => {
                if (window.speechSynthesis) window.speechSynthesis.cancel();
                const utt = window.__capturedSpeechUtterance;
                if (utt) {
                    try { utt.dispatchEvent(new SpeechSynthesisEvent('end', {utterance: utt})); } catch(e) {
                        try { utt.dispatchEvent(new Event('end')); } catch(e2) {}
                    }
                    if (utt.onend) { try { utt.onend(new Event('end')); } catch(e) {} }
                }
                // Force-end Audio elements
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
            }""")
            print(f"  Audio: force-ended speech synthesis", flush=True)
            await asyncio.sleep(1.0)

            # Click Complete/Done button
            for _ in range(6):
                clicked = await self.browser.page.evaluate("""() => {
                    const btns = [...document.querySelectorAll('button')];
                    for (const btn of btns) {
                        const text = (btn.textContent || '').trim().toLowerCase();
                        if ((text.includes('complete') || text.includes('done') || text.includes('finish')) &&
                            !text.includes('playing') && btn.offsetParent && !btn.disabled) {
                            btn.click(); return true;
                        }
                    }
                    return false;
                }""")
                if clicked:
                    print(f"  Audio: clicked Complete", flush=True)
                    break
                await asyncio.sleep(0.5)
            else:
                # Last resort: click "Playing..." button (might toggle to Complete)
                await self.browser.page.evaluate("""() => {
                    const btns = [...document.querySelectorAll('button')];
                    for (const btn of btns) {
                        const text = (btn.textContent || '').trim().toLowerCase();
                        if (text.includes('playing') && btn.offsetParent) { btn.click(); return; }
                    }
                }""")

            await asyncio.sleep(1.0)
            return True
        except Exception as e:
            print(f"  Audio error: {e}", flush=True)
            return False

    async def _try_canvas_challenge(self) -> bool:
        """Handle Canvas Challenge - draw shapes or strokes on a canvas."""
        try:
            canvas_info = await self.browser.page.evaluate("""() => {
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
            }""")
            if not canvas_info.get('found'):
                return False

            cx, cy, cw, ch = canvas_info['x'], canvas_info['y'], canvas_info['w'], canvas_info['h']
            shape = canvas_info.get('shape', 'strokes')
            print(f"  Canvas: drawing {shape}", flush=True)

            if shape == 'square':
                margin = 0.2
                corners = [(cx+cw*margin, cy+ch*margin), (cx+cw*(1-margin), cy+ch*margin),
                           (cx+cw*(1-margin), cy+ch*(1-margin)), (cx+cw*margin, cy+ch*(1-margin)),
                           (cx+cw*margin, cy+ch*margin)]
                await self.browser.page.mouse.move(corners[0][0], corners[0][1])
                await self.browser.page.mouse.down()
                for corner in corners[1:]:
                    await self.browser.page.mouse.move(corner[0], corner[1], steps=15)
                    await asyncio.sleep(0.05)
                await self.browser.page.mouse.up()
            elif shape == 'circle':
                import math
                center_x, center_y = cx + cw/2, cy + ch/2
                radius = min(cw, ch) * 0.35
                start_x = center_x + radius
                await self.browser.page.mouse.move(start_x, center_y)
                await self.browser.page.mouse.down()
                for i in range(1, 37):
                    angle = (2 * math.pi * i) / 36
                    await self.browser.page.mouse.move(center_x + radius*math.cos(angle),
                                                        center_y + radius*math.sin(angle), steps=3)
                await self.browser.page.mouse.up()
            elif shape == 'triangle':
                margin = 0.2
                corners = [(cx+cw/2, cy+ch*margin), (cx+cw*(1-margin), cy+ch*(1-margin)),
                           (cx+cw*margin, cy+ch*(1-margin)), (cx+cw/2, cy+ch*margin)]
                await self.browser.page.mouse.move(corners[0][0], corners[0][1])
                await self.browser.page.mouse.down()
                for corner in corners[1:]:
                    await self.browser.page.mouse.move(corner[0], corner[1], steps=15)
                    await asyncio.sleep(0.05)
                await self.browser.page.mouse.up()
            else:
                # Default: draw 4 varied strokes
                for i in range(4):
                    sx = cx + cw*0.2 + (i*cw*0.15)
                    sy = cy + ch*0.3 + (i*ch*0.1)
                    ex = cx + cw*0.5 + (i*cw*0.1)
                    ey = cy + ch*0.7 - (i*ch*0.05)
                    await self.browser.page.mouse.move(sx, sy)
                    await self.browser.page.mouse.down()
                    await self.browser.page.mouse.move(ex, ey, steps=10)
                    await self.browser.page.mouse.up()
                    await asyncio.sleep(0.3)

            # Click Complete/Done button
            await asyncio.sleep(0.5)
            await self.browser.page.evaluate("""() => {
                const btns = [...document.querySelectorAll('button')];
                for (const btn of btns) {
                    const t = (btn.textContent || '').trim().toLowerCase();
                    if ((t.includes('complete') || t.includes('done') || t.includes('check') ||
                         t.includes('verify') || t.includes('reveal')) &&
                        !t.includes('clear') && btn.offsetParent && !btn.disabled) {
                        btn.click(); return;
                    }
                }
            }""")
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            print(f"  Canvas error: {e}", flush=True)
            return False

    async def _try_service_worker_challenge(self) -> str | None:
        """Handle Service Worker Challenge: Register → wait for cache → Retrieve → extract code."""
        try:
            # Step 1: Click Register button
            registered = await self.browser.page.evaluate("""() => {
                const btns = [...document.querySelectorAll('button')];
                // Check if already registered
                const text = document.body.textContent || '';
                if (text.includes('Registered') || text.includes('● Registered')) {
                    return 'already';
                }
                for (const btn of btns) {
                    const t = (btn.textContent || '').trim().toLowerCase();
                    if ((t.includes('register') && t.includes('service')) ||
                        (t.includes('register') && !btn.disabled && btn.offsetParent)) {
                        btn.click();
                        return 'clicked';
                    }
                }
                return null;
            }""")
            if not registered:
                return None
            print(f"  Service Worker: register={registered}", flush=True)

            # Step 2: Wait for cache to be populated
            for i in range(15):
                cached = await self.browser.page.evaluate("""() => {
                    const text = document.body.textContent || '';
                    return text.includes('Cached') || text.includes('cached') ||
                           text.includes('● Cached');
                }""")
                if cached:
                    break
                await asyncio.sleep(0.3)

            # Step 3: Click Retrieve from Cache button
            await self.browser.page.evaluate("""() => {
                const btns = [...document.querySelectorAll('button')];
                for (const btn of btns) {
                    const t = (btn.textContent || '').trim().toLowerCase();
                    if ((t.includes('retrieve') || t.includes('from cache')) &&
                        !btn.disabled && btn.offsetParent) {
                        btn.click();
                        return true;
                    }
                }
                return false;
            }""")
            await asyncio.sleep(0.5)

            # Step 4: Extract code from result
            code = await self.browser.page.evaluate("""() => {
                // Look for green box with code
                const greens = document.querySelectorAll('.bg-green-100, .bg-green-50, .text-green-600, .text-green-700, .border-green-500');
                for (const el of greens) {
                    const m = (el.textContent || '').match(/\\b([A-Z0-9]{6})\\b/);
                    if (m) return m[1];
                }
                // Look for "code is:" pattern
                const text = document.body.textContent || '';
                const m = text.match(/(?:code\\s+(?:is|retrieved)[^:]*:\\s*)([A-Z0-9]{6})/i);
                if (m) return m[1].toUpperCase();
                // Look for any new 6-char code near "retrieved" or "cache"
                const m2 = text.match(/(?:retrieved|cache)[^.]*?\\b([A-Z0-9]{6})\\b/i);
                if (m2) return m2[1].toUpperCase();
                return null;
            }""")
            if code:
                from dom_parser import FALSE_POSITIVES
                if code not in FALSE_POSITIVES:
                    print(f"  Service Worker: code={code}", flush=True)
                    return code
            print(f"  Service Worker: no code found", flush=True)
            return None
        except Exception as e:
            print(f"  Service Worker error: {e}", flush=True)
            return None

    async def _try_shadow_dom_challenge(self) -> str | None:
        """Handle Shadow DOM Challenge: click through nested layers to reveal code."""
        try:
            for click_round in range(10):
                result = await self.browser.page.evaluate("""() => {
                    const text = document.body.textContent || '';
                    // Check if code already revealed
                    const codeMatch = text.match(/(?:code|Code)[^:]*:\\s*([A-Z0-9]{6})/);
                    if (codeMatch) return {done: true, code: codeMatch[1]};
                    
                    // Check completion progress  
                    const progressMatch = text.match(/(\\d+)\\/(\\d+)\\s*(?:levels?|layers?)/i);
                    const current = progressMatch ? parseInt(progressMatch[1]) : 0;
                    const total = progressMatch ? parseInt(progressMatch[2]) : 3;
                    
                    // If all levels done, click reveal/complete button
                    if (current >= total) {
                        for (const btn of document.querySelectorAll('button')) {
                            const t = (btn.textContent || '').trim().toLowerCase();
                            if ((t.includes('reveal') || t.includes('complete') || t.includes('extract')) && 
                                !btn.disabled && btn.offsetParent) {
                                btn.click();
                                return {clicked: 'reveal_btn', current, total};
                            }
                        }
                    }
                    
                    // Strategy: find all divs with cursor-pointer that have a DIRECT text
                    // starting with "Shadow Level N" (not inherited from children)
                    const candidates = [...document.querySelectorAll('div')].filter(el => {
                        const cls = el.getAttribute('class') || '';
                        if (!cls.includes('cursor-pointer') && !cls.includes('cursor_pointer')) return false;
                        if (!el.offsetParent || el.offsetWidth < 30) return false;
                        // Get only direct text nodes (not from children)
                        let directText = '';
                        for (const node of el.childNodes) {
                            if (node.nodeType === 3) directText += node.textContent;
                        }
                        directText = directText.trim();
                        if (!directText) {
                            // Maybe first child is a span/strong with the label
                            const first = el.children[0];
                            if (first && first.tagName !== 'DIV') directText = first.textContent.trim();
                        }
                        return /^(?:Shadow\\s+)?Level\\s+\\d/i.test(directText);
                    });
                    
                    // Click first un-completed level
                    for (const el of candidates) {
                        const cls = el.getAttribute('class') || '';
                        const allText = el.textContent || '';
                        if (!cls.includes('green') && !allText.startsWith('✓') && 
                            !allText.includes('✓ Shadow') && !allText.includes('✓Shadow')) {
                            // Get label from direct text only
                            let label = '';
                            for (const node of el.childNodes) {
                                if (node.nodeType === 3) label += node.textContent;
                            }
                            el.click();
                            return {clicked: (label || 'level').trim().substring(0, 30), current, total};
                        }
                    }
                    
                    // Fallback: any slate div that looks like a level
                    for (const el of document.querySelectorAll('div[class*="slate"]')) {
                        const cls = el.getAttribute('class') || '';
                        if (!cls.includes('cursor') && !cls.includes('hover')) continue;
                        if (cls.includes('green')) continue;
                        if (!el.offsetParent || el.offsetWidth < 30) continue;
                        // Check that this div is small enough to be a level box (< 500 chars direct)
                        let directLen = 0;
                        for (const node of el.childNodes) {
                            if (node.nodeType === 3) directLen += node.textContent.trim().length;
                        }
                        if (directLen > 0 && directLen < 50) {
                            el.click();
                            return {clicked: 'slate-fallback', current, total};
                        }
                    }
                    
                    return {clicked: null, current, total};
                }""")
                
                if not result:
                    break
                if result.get('done') and result.get('code'):
                    from dom_parser import FALSE_POSITIVES
                    code = result['code']
                    if code not in FALSE_POSITIVES:
                        print(f"  Shadow DOM: code={code} after {click_round+1} clicks", flush=True)
                        return code
                if result.get('clicked'):
                    print(f"  Shadow DOM: {result['clicked']} ({result.get('current',0)}/{result.get('total',3)})", flush=True)
                    await asyncio.sleep(0.4)
                else:
                    break
            
            # Final extraction attempt
            code = await self.browser.page.evaluate("""() => {
                const text = document.body.textContent || '';
                const m = text.match(/(?:code|Code)[^:]*:\\s*([A-Z0-9]{6})/);
                return m ? m[1] : null;
            }""")
            if code:
                from dom_parser import FALSE_POSITIVES
                if code not in FALSE_POSITIVES:
                    return code
            return None
        except Exception as e:
            print(f"  Shadow DOM error: {e}", flush=True)
            return None

    async def _try_websocket_challenge(self) -> str | None:
        """Handle WebSocket Challenge: click Connect, wait for messages, click Reveal, extract code."""
        try:
            # Step 1: Click Connect button
            connected = await self.browser.page.evaluate("""() => {
                const text = document.body.textContent || '';
                if (text.includes('Connected') || text.includes('● Connected')) return 'already';
                for (const btn of document.querySelectorAll('button')) {
                    const t = (btn.textContent || '').trim().toLowerCase();
                    if ((t.includes('connect') || t === 'connect') && !btn.disabled && btn.offsetParent) {
                        btn.click();
                        return 'clicked';
                    }
                }
                return null;
            }""")
            if not connected:
                return None
            print(f"  WebSocket: connect={connected}", flush=True)
            
            # Step 2: Wait for connection + messages + Reveal Code button
            for i in range(25):
                await asyncio.sleep(0.5)
                status = await self.browser.page.evaluate("""() => {
                    const text = document.body.textContent || '';
                    if (text.match(/\\b[A-Z0-9]{6}\\b/) && (text.includes('code') || text.includes('Code'))) return 'has_code';
                    // Check for Reveal Code button
                    for (const btn of document.querySelectorAll('button')) {
                        const t = (btn.textContent || '').trim().toLowerCase();
                        if (t.includes('reveal') && !btn.disabled && btn.offsetParent) return 'ready';
                    }
                    if (text.includes('Ready to reveal') || text.includes('reveal code')) return 'ready';
                    if (text.includes('Connected') || text.includes('● Connected')) return 'connected';
                    return 'waiting';
                }""")
                if status == 'has_code' or status == 'ready':
                    break
                if status == 'connected' and i > 10:  # Wait up to 5s after connected
                    break
            
            # Step 3: Click Reveal Code / Request / Send buttons (with retries)
            for reveal_attempt in range(5):
                clicked = await self.browser.page.evaluate("""() => {
                    for (const btn of document.querySelectorAll('button')) {
                        const t = (btn.textContent || '').trim().toLowerCase();
                        if ((t.includes('reveal') || t.includes('request') || t.includes('get code') ||
                             t.includes('send') || t.includes('extract')) && 
                            !t.includes('connect') && !btn.disabled && btn.offsetParent) {
                            btn.click();
                            return t.substring(0, 30);
                        }
                    }
                    return null;
                }""")
                if clicked:
                    print(f"  WebSocket: clicked '{clicked}'", flush=True)
                    await asyncio.sleep(0.8)
                else:
                    await asyncio.sleep(0.5)
                    continue
                
                # Try to extract code immediately after clicking
                code = await self.browser.page.evaluate("""() => {
                    const text = document.body.textContent || '';
                    for (const el of document.querySelectorAll('.text-green-600, .text-green-700, .bg-green-100, .bg-green-50')) {
                        const m = (el.textContent || '').match(/\\b([A-Z0-9]{6})\\b/);
                        if (m) return m[1];
                    }
                    const m = text.match(/(?:code|Code|CODE)[^:]*?:\\s*([A-Z0-9]{6})/);
                    if (m) return m[1];
                    for (const el of document.querySelectorAll('[class*="cyan"], [class*="terminal"], pre, code, [class*="mono"]')) {
                        const m2 = (el.textContent || '').match(/\\b([A-Z0-9]{6})\\b/);
                        if (m2) return m2[1];
                    }
                    return null;
                }""")
                if code:
                    from dom_parser import FALSE_POSITIVES
                    if code not in FALSE_POSITIVES:
                        print(f"  WebSocket: code={code}", flush=True)
                        return code
            
            print(f"  WebSocket: no code found", flush=True)
            return None
        except Exception as e:
            print(f"  WebSocket error: {e}", flush=True)
            return None

    async def _try_delayed_reveal(self) -> str | None:
        """Handle Delayed Reveal Challenge: wait for timer to complete, extract code."""
        try:
            # Check timer remaining
            for i in range(25):
                result = await self.browser.page.evaluate("""() => {
                    const text = document.body.textContent || '';
                    // Check if code already revealed
                    const codeMatch = text.match(/(?:code|Code|challenge code)[^:]*:\\s*([A-Z0-9]{6})/i);
                    if (codeMatch) return {code: codeMatch[1]};
                    // Check for any 6-char code in revealed/success area
                    const greens = document.querySelectorAll('.text-green-600, .bg-green-100, .bg-blue-100, .bg-purple-100');
                    for (const el of greens) {
                        const m = (el.textContent || '').match(/\\b([A-Z0-9]{6})\\b/);
                        if (m) return {code: m[1]};
                    }
                    // Check timer
                    const timerMatch = text.match(/(\\d+\\.?\\d*)\\s*s(?:econds?)?\\s*remaining/i);
                    const remaining = timerMatch ? parseFloat(timerMatch[1]) : null;
                    const done = text.includes('revealed') || text.includes('Complete') || text.includes('100%');
                    return {remaining, done};
                }""")
                if result and result.get('code'):
                    from dom_parser import FALSE_POSITIVES
                    code = result['code']
                    if code not in FALSE_POSITIVES:
                        print(f"  Delayed Reveal: code={code} after {i*0.4:.1f}s", flush=True)
                        return code
                if result and result.get('done'):
                    await asyncio.sleep(0.3)
                    continue  # One more loop to extract code
                if result and result.get('remaining') is not None and result['remaining'] < 0.5:
                    await asyncio.sleep(0.6)
                    continue
                await asyncio.sleep(0.4)
            
            # Final extraction
            code = await self.browser.page.evaluate("""() => {
                const text = document.body.textContent || '';
                // Broad code search near "code" or "revealed"
                const matches = text.match(/\\b([A-Z0-9]{6})\\b/g) || [];
                const known = new Set(['FILLER', 'SUBMIT', 'BUTTON', 'SCROLL', 'REVEAL']);
                return matches.find(m => !known.has(m) && /[0-9]/.test(m)) || null;
            }""")
            if code:
                from dom_parser import FALSE_POSITIVES
                if code not in FALSE_POSITIVES:
                    print(f"  Delayed Reveal: code={code} (final)", flush=True)
                    return code
            print(f"  Delayed Reveal: no code found", flush=True)
            return None
        except Exception as e:
            print(f"  Delayed Reveal error: {e}", flush=True)
            return None

    async def _try_mutation_challenge(self) -> str | None:
        """Handle DOM Mutation Challenge: click Trigger Mutation button N times, then Complete."""
        try:
            # Scroll mutation challenge into view first
            await self.browser.page.evaluate("""() => {
                for (const el of document.querySelectorAll('div, h2, h3, p')) {
                    const t = (el.textContent || '').substring(0, 200);
                    if (t.includes('Mutation Challenge') || t.includes('Trigger Mutation')) {
                        el.scrollIntoView({behavior: 'instant', block: 'center'});
                        break;
                    }
                }
            }""")
            await asyncio.sleep(0.2)
            
            for click_round in range(12):
                # Check progress and code with space-aware regex
                result = await self.browser.page.evaluate("""() => {
                    const text = document.body.textContent || '';
                    const codeMatch = text.match(/(?:code|Code)[^:]*:\\s*([A-Z0-9]{6})/);
                    if (codeMatch) return {done: true, code: codeMatch[1]};
                    
                    // Space-aware progress: "4 / 5" or "4/5"
                    let current = 0, total = 5;
                    const m = text.match(/(\\d+)\\s*\\/\\s*(\\d+)\\s*(?:mutations?|triggered|complete)/i) ||
                              text.match(/Mutations?[^:]*?:\\s*(\\d+)\\s*\\/\\s*(\\d+)/i) ||
                              text.match(/triggered[^:]*?:\\s*(\\d+)\\s*\\/\\s*(\\d+)/i);
                    if (m) { current = parseInt(m[1]); total = parseInt(m[2]); }
                    
                    return {current, total};
                }""")
                
                if not result:
                    break
                if result.get('done') and result.get('code'):
                    from dom_parser import FALSE_POSITIVES
                    code = result['code']
                    if code not in FALSE_POSITIVES:
                        print(f"  Mutation: code={code} after {click_round+1} clicks", flush=True)
                        return code
                
                current = result.get('current', 0)
                total = result.get('total', 5)
                
                # If complete, click Complete button via Playwright locator
                if current >= total:
                    try:
                        complete_btn = self.browser.page.locator('button:has-text("Complete")').first
                        if await complete_btn.is_visible(timeout=500):
                            await complete_btn.click(timeout=2000)
                            print(f"  Mutation: clicking complete ({current}/{total})", flush=True)
                            await asyncio.sleep(0.5)
                            continue
                    except Exception:
                        pass
                
                # Click Trigger Mutation via Playwright locator
                try:
                    trigger_btn = self.browser.page.locator('button:has-text("Trigger")').first
                    if await trigger_btn.is_visible(timeout=500):
                        await trigger_btn.click(timeout=2000)
                        print(f"  Mutation: trigger ({current}/{total})", flush=True)
                        await asyncio.sleep(0.3)
                        continue
                except Exception:
                    pass
                break
            
            # Final extraction
            code = await self.browser.page.evaluate("""() => {
                const text = document.body.textContent || '';
                const m = text.match(/(?:code|Code)[^:]*:\\s*([A-Z0-9]{6})/);
                if (m) return m[1];
                for (const el of document.querySelectorAll('.text-green-600, .bg-green-100, .bg-rose-100, [class*="success"]')) {
                    const cm = (el.textContent || '').match(/\\b([A-Z0-9]{6})\\b/);
                    if (cm) return cm[1];
                }
                return null;
            }""")
            if code:
                from dom_parser import FALSE_POSITIVES
                if code not in FALSE_POSITIVES:
                    return code
            return None
        except Exception as e:
            print(f"  Mutation error: {e}", flush=True)
            return None

    async def _try_iframe_challenge(self) -> str | None:
        """Handle Recursive Iframe Challenge: navigate through N nested levels, extract code."""
        try:
            extract_attempts = 0
            for click_round in range(15):
                # Scroll iframe challenge into view
                await self.browser.page.evaluate("""() => {
                    for (const el of document.querySelectorAll('div')) {
                        const t = (el.textContent || '').substring(0, 200);
                        if (t.includes('Iframe Challenge') || t.includes('Recursive Iframe')) {
                            el.scrollIntoView({behavior: 'instant', block: 'center'});
                            break;
                        }
                    }
                }""")
                await asyncio.sleep(0.2)
                
                # Check state
                result = await self.browser.page.evaluate("""() => {
                    const text = document.body.textContent || '';
                    
                    // Check for revealed code
                    const codeEls = document.querySelectorAll('.text-green-600, .text-green-700, .bg-green-100, .bg-green-50, .bg-emerald-100, .text-emerald-600, [class*="success"]');
                    for (const el of codeEls) {
                        const m = (el.textContent || '').match(/\\b([A-Z0-9]{6})\\b/);
                        if (m && !['IFRAME','BWRONG','1WRONG','CWRONG'].includes(m[1])) return {done: true, code: m[1], source: 'green'};
                    }
                    const codeMatch = text.match(/(?:code|Code)[^:]*?:\\s*([A-Z0-9]{6})/);
                    if (codeMatch && !['IFRAME','BWRONG','1WRONG','CWRONG'].includes(codeMatch[1])) return {done: true, code: codeMatch[1], source: 'text'};
                    
                    // Space-aware depth: "4 / 5" or "4/5"
                    const depthMatch = text.match(/(?:depth|level)[^:]*?:\\s*(\\d+)\\s*\\/\\s*(\\d+)/i) ||
                                       text.match(/(\\d+)\\s*\\/\\s*(\\d+)\\s*(?:depth|levels?)/i);
                    const current = depthMatch ? parseInt(depthMatch[1]) : 0;
                    const total = depthMatch ? parseInt(depthMatch[2]) : 4;
                    
                    // Find incomplete level divs (green, not emerald = not yet entered)
                    const levelDivs = [];
                    for (const div of document.querySelectorAll('div[class*="border-2"][class*="rounded"]')) {
                        const cls = div.getAttribute('class') || '';
                        const firstText = (div.childNodes[0]?.textContent || '').trim();
                        const levelMatch = firstText.match(/(?:Iframe\\s+)?Level\\s+(\\d+)/i);
                        if (levelMatch && div.offsetParent && div.offsetWidth > 50) {
                            const isComplete = firstText.includes('✓') || firstText.includes('✔') || cls.includes('emerald');
                            levelDivs.push({level: parseInt(levelMatch[1]), complete: isComplete, text: firstText});
                        }
                    }
                    
                    // Find buttons
                    const enterBtns = [];
                    let extractBtn = false;
                    for (const btn of document.querySelectorAll('button')) {
                        if (!btn.offsetParent || btn.disabled) continue;
                        const r = btn.getBoundingClientRect();
                        if (r.width === 0) continue;
                        const t = (btn.textContent || '').trim().toLowerCase();
                        const enterMatch = t.match(/enter\\s+level\\s+(\\d+)/i) || t.match(/level\\s+(\\d+)/i);
                        if (enterMatch && !t.includes('submit') && !t.includes('extract')) {
                            enterBtns.push({level: parseInt(enterMatch[1]), text: t});
                        }
                        if (t.includes('go deeper') || t.includes('next level') || t.includes('descend')) {
                            enterBtns.push({level: current + 1, text: t});
                        }
                        if (t.includes('extract') || t.includes('get code')) {
                            extractBtn = true;
                        }
                    }
                    
                    const atDeepest = text.includes('deepest level') || text.includes('reached the deepest');
                    return {current, total, levelDivs, enterBtns, extractBtn, atDeepest};
                }""")
                
                if not result:
                    break
                if result.get('done') and result.get('code'):
                    from dom_parser import FALSE_POSITIVES
                    code = result['code']
                    if code not in FALSE_POSITIVES:
                        print(f"  Iframe: code={code} ({result.get('source','?')}) after {click_round+1} clicks", flush=True)
                        return code
                
                current = result.get('current', 0)
                total = result.get('total', 4)
                
                # PRIORITY 1: Click incomplete level div (one that's not emerald/checked)
                level_divs = result.get('levelDivs', [])
                incomplete = [d for d in level_divs if not d['complete']]
                if incomplete:
                    # Click the LOWEST incomplete level first
                    incomplete.sort(key=lambda d: d['level'])
                    target = incomplete[0]
                    try:
                        # Use evaluate_handle to get exact element, then Playwright click
                        handle = await self.browser.page.evaluate_handle(f"""() => {{
                            for (const div of document.querySelectorAll('div[class*="border-2"][class*="rounded"]')) {{
                                const t = (div.childNodes[0]?.textContent || '').trim();
                                if (t.includes('Level {target["level"]}') && !t.includes('✓') && !t.includes('✔')) {{
                                    div.scrollIntoView({{behavior: 'instant', block: 'center'}});
                                    return div;
                                }}
                            }}
                            return null;
                        }}""")
                        el = handle.as_element()
                        if el:
                            await el.click(timeout=2000)
                            print(f"  Iframe: clicked level div {target['level']} ({current}/{total})", flush=True)
                            extract_attempts = 0
                            await asyncio.sleep(0.4)
                            continue
                    except Exception as e:
                        print(f"  Iframe: level div click failed: {e}", flush=True)
                
                # PRIORITY 2: Click Enter Level button for the CORRECT next level
                enter_btns = result.get('enterBtns', [])
                if current < total and enter_btns:
                    # Sort by level, pick the one closest to current+1
                    target_level = current + 1
                    enter_btns.sort(key=lambda b: abs(b['level'] - target_level))
                    target = enter_btns[0]
                    try:
                        handle = await self.browser.page.evaluate_handle(f"""() => {{
                            for (const btn of document.querySelectorAll('button')) {{
                                if (!btn.offsetParent || btn.disabled) continue;
                                const t = btn.textContent.trim().toLowerCase();
                                if (t.includes('{target["text"][:25]}')) {{
                                    btn.scrollIntoView({{behavior: 'instant', block: 'center'}});
                                    return btn;
                                }}
                            }}
                            return null;
                        }}""")
                        el = handle.as_element()
                        if el:
                            await el.click(timeout=2000)
                            print(f"  Iframe: clicked '{target['text']}' ({current}/{total})", flush=True)
                            extract_attempts = 0
                            await asyncio.sleep(0.3)
                            continue
                    except Exception as e:
                        print(f"  Iframe: enter btn click failed: {e}", flush=True)
                
                # PRIORITY 3: Extract code (at deepest or current >= total)
                if result.get('extractBtn') and (current >= total or result.get('atDeepest')):
                    extract_attempts += 1
                    if extract_attempts > 3:
                        print(f"  Iframe: Extract clicked {extract_attempts}x with no code, bailing", flush=True)
                        break
                    try:
                        handle = await self.browser.page.evaluate_handle("""() => {
                            for (const btn of document.querySelectorAll('button')) {
                                if (!btn.offsetParent || btn.disabled) continue;
                                const t = btn.textContent.trim().toLowerCase();
                                if (t.includes('extract') || t.includes('get code')) {
                                    btn.scrollIntoView({behavior: 'instant', block: 'center'});
                                    return btn;
                                }
                            }
                            return null;
                        }""")
                        el = handle.as_element()
                        if el:
                            await el.click(timeout=2000)
                            print(f"  Iframe: extract ({current}/{total})", flush=True)
                            await asyncio.sleep(0.8)
                            continue
                    except Exception as e:
                        print(f"  Iframe: extract click failed: {e}", flush=True)
                
                print(f"  Iframe: nothing to click ({current}/{total}, deepest={result.get('atDeepest')})", flush=True)
                break
            
            # Final broad extraction - search entire page for 6-char codes near iframe area
            code = await self.browser.page.evaluate("""() => {
                const text = document.body.textContent || '';
                // Look for code pattern near "extracted" or "code" text
                const m = text.match(/(?:code|Code|extracted)[^:]*?:\\s*([A-Z0-9]{6})/);
                if (m) return m[1];
                // Look in any element with dashed border or green styling
                for (const el of document.querySelectorAll('[class*="dashed"], [class*="green"], [class*="emerald"]')) {
                    const t = (el.textContent || '').trim();
                    const cm = t.match(/\\b([A-Z0-9]{6})\\b/);
                    if (cm && t.length < 200) return cm[1];
                }
                return null;
            }""")
            if code:
                from dom_parser import FALSE_POSITIVES
                if code not in FALSE_POSITIVES:
                    print(f"  Iframe: code={code} (final extraction)", flush=True)
                    return code
            return None
        except Exception as e:
            print(f"  Iframe error: {e}", flush=True)
            return None

    async def _try_split_parts(self) -> bool:
        """Handle Split Parts Challenge - click scattered Part N elements."""
        try:
            for click_round in range(10):
                result = await self.browser.page.evaluate("""() => {
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
                        el.click();
                        clicked++;
                    });
                    return {found, total, clicked, done: false};
                }""")
                if result.get('done'):
                    print(f"  Split parts: all collected!", flush=True)
                    break
                if result.get('clicked', 0) == 0:
                    await self.browser.page.evaluate("() => window.scrollBy(0, 400)")
                await asyncio.sleep(0.5)
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            print(f"  Split parts error: {e}", flush=True)
            return False

    async def _try_rotating_code(self) -> bool:
        """Handle Rotating Code Challenge - click Capture N times."""
        try:
            for _ in range(15):
                state = await self.browser.page.evaluate("""() => {
                    const btns = [...document.querySelectorAll('button')];
                    let done = 0, required = 3;
                    for (const btn of btns) {
                        const t = (btn.textContent || '').trim();
                        const m = t.match(/[Cc]apture.*?(\\d+)\\/(\\d+)/);
                        if (m) { done = parseInt(m[1]); required = parseInt(m[2]); break; }
                    }
                    return {done, required, complete: done >= required};
                }""")
                if state.get('complete'):
                    return True
                clicked = await self.browser.page.evaluate("""() => {
                    for (const btn of document.querySelectorAll('button')) {
                        const t = (btn.textContent || '').trim().toLowerCase();
                        if (t.includes('capture') && btn.offsetParent && !btn.disabled) {
                            btn.click(); return true;
                        }
                    }
                    return false;
                }""")
                if not clicked:
                    break
                await asyncio.sleep(1.0)
            return True
        except Exception as e:
            print(f"  Rotating code error: {e}", flush=True)
            return False

    async def _try_multi_tab(self) -> bool:
        """Handle Multi-Tab Challenge - click through all tabs to collect code parts."""
        try:
            for _ in range(3):
                result = await self.browser.page.evaluate("""() => {
                    const btns = [...document.querySelectorAll('button')];
                    const tabBtns = btns.filter(b => {
                        const t = (b.textContent || '').trim().toLowerCase();
                        return (t.includes('tab') || t.match(/^\\d+$/)) && b.offsetParent;
                    });
                    for (const btn of tabBtns) btn.click();
                    return tabBtns.length;
                }""")
                if result > 0:
                    print(f"  Multi-tab: clicked {result} tabs", flush=True)
                await asyncio.sleep(0.5)
            return True
        except Exception as e:
            print(f"  Multi-tab error: {e}", flush=True)
            return False

    async def _try_sequence_challenge(self) -> bool:
        """Handle Sequence Challenge - perform 4 actions: click, hover, type, scroll."""
        try:
            # Action 1: Click "Click Me" button
            await self.browser.page.evaluate("""() => {
                for (const btn of document.querySelectorAll('button')) {
                    const t = (btn.textContent || '').trim().toLowerCase();
                    if (t.includes('click me') && btn.offsetParent && !btn.disabled) { btn.click(); return; }
                }
            }""")
            await asyncio.sleep(0.3)

            # Action 2: Hover over the hover area
            hover_info = await self.browser.page.evaluate("""() => {
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
            }""")
            if hover_info:
                await self.browser.page.mouse.move(hover_info['x'], hover_info['y'])
                await asyncio.sleep(0.5)
                await self.browser.page.evaluate(f"""() => {{
                    const el = document.elementFromPoint({hover_info['x']}, {hover_info['y']});
                    if (el) {{
                        el.dispatchEvent(new MouseEvent('mouseenter', {{bubbles: true}}));
                        el.dispatchEvent(new MouseEvent('mouseover', {{bubbles: true}}));
                    }}
                }}""")
                await asyncio.sleep(0.8)

            # Action 3: Type text in non-code input
            await self.browser.page.evaluate("""() => {
                const inputs = [...document.querySelectorAll('input[type="text"], input:not([type]), textarea')];
                const inp = inputs.find(i => {
                    const ph = (i.placeholder || '').toLowerCase();
                    return !ph.includes('code') && i.offsetParent && i.type !== 'number' && i.type !== 'hidden';
                });
                if (inp) {
                    inp.scrollIntoView({behavior: 'instant', block: 'center'});
                    inp.focus();
                    const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
                    s.call(inp, 'hello world');
                    inp.dispatchEvent(new Event('input', {bubbles: true}));
                    inp.dispatchEvent(new Event('change', {bubbles: true}));
                }
            }""")
            await asyncio.sleep(0.3)

            # Action 4: Scroll inside scroll box
            scroll_info = await self.browser.page.evaluate("""() => {
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
            }""")
            if scroll_info:
                await self.browser.page.mouse.move(scroll_info['x'], scroll_info['y'])
                await self.browser.page.mouse.wheel(0, 300)
            await asyncio.sleep(0.3)

            # Click Complete button
            await self.browser.page.evaluate("""() => {
                for (const btn of document.querySelectorAll('button')) {
                    const t = (btn.textContent || '').trim().toLowerCase();
                    if (t.includes('complete') && btn.offsetParent && !btn.disabled) { btn.click(); return; }
                }
            }""")
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            print(f"  Sequence error: {e}", flush=True)
            return False

    async def _try_video_challenge(self) -> bool:
        """Handle Video Frames Challenge - navigate to target frame."""
        try:
            state = await self.browser.page.evaluate("""() => {
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
            }""")
            target = state.get('targetFrame')
            if target is None:
                return False
            print(f"  Video: navigating to frame {target} (at {state.get('currentFrame')})", flush=True)

            # Perform required seek operations
            for _ in range(5):
                await self.browser.page.evaluate("""() => {
                    for (const btn of document.querySelectorAll('button')) {
                        if (btn.textContent.trim() === '+1' && btn.offsetParent) { btn.click(); return; }
                    }
                }""")
                await asyncio.sleep(0.3)

            # Navigate to target frame using +10/-10 and +1/-1
            for _ in range(20):
                current = await self.browser.page.evaluate("""() => {
                    const m = (document.body.textContent || '').match(/Frame\\s+(\\d+)\\//);
                    return m ? parseInt(m[1]) : 0;
                }""")
                if current == target:
                    break
                diff = target - current
                btn_text = '+10' if diff >= 10 else '-10' if diff <= -10 else '+1' if diff > 0 else '-1'
                await self.browser.page.evaluate(f"""() => {{
                    for (const btn of document.querySelectorAll('button')) {{
                        if (btn.textContent.trim() === '{btn_text}' && btn.offsetParent) {{ btn.click(); return; }}
                    }}
                }}""")
                await asyncio.sleep(0.2)

            # Click Complete/Reveal button
            await asyncio.sleep(0.5)
            await self.browser.page.evaluate("""() => {
                for (const btn of document.querySelectorAll('button')) {
                    const t = (btn.textContent || '').trim().toLowerCase();
                    if ((t.includes('complete') || t.includes('done') || t.includes('reveal')) &&
                        btn.offsetParent && !btn.disabled) { btn.click(); return; }
                }
            }""")
            await asyncio.sleep(0.5)
            return True
        except Exception as e:
            print(f"  Video error: {e}", flush=True)
            return False

    # ── Action execution ────────────────────────────────────────────────

    async def _execute_action(self, action) -> str:
        """Execute the agent's suggested action. Returns description string."""
        atype = action.action_type
        target = action.target_selector
        value = action.value

        try:
            if atype == ActionType.CLICK or atype == ActionType.CLICK_REVEAL:
                if target:
                    try:
                        await self.browser.page.click(target, timeout=2000)
                        return f"clicked {target}"
                    except Exception:
                        # Try JS click
                        await self.browser.page.evaluate(f"""() => {{
                            const el = document.querySelector('{target}');
                            if (el) el.click();
                        }}""")
                        return f"js-clicked {target}"
                return "click (no target)"

            elif atype == ActionType.TYPE:
                if target and value:
                    try:
                        await self.browser.page.fill(target, value)
                    except Exception:
                        await self.browser.page.evaluate(f"""(val) => {{
                            const el = document.querySelector('{target}');
                            if (el) {{
                                const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
                                s.call(el, val);
                                el.dispatchEvent(new Event('input', {{bubbles: true}}));
                                el.dispatchEvent(new Event('change', {{bubbles: true}}));
                            }}
                        }}""", value)
                    return f"typed '{value}' in {target}"
                return "type (no target/value)"

            elif atype == ActionType.SCROLL:
                await self.browser.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(0.3)
                return "scrolled to bottom"

            elif atype == ActionType.SCROLL_UP:
                await self.browser.page.evaluate("window.scrollTo(0, 0)")
                await asyncio.sleep(0.3)
                return "scrolled to top"

            elif atype == ActionType.HOVER:
                if target:
                    try:
                        loc = self.browser.page.locator(target)
                        if await loc.count() > 0:
                            await loc.first.hover(timeout=2000)
                            await asyncio.sleep(1.5)
                            return f"hovered {target} for 1.5s"
                    except Exception:
                        pass
                    # Fallback: JS dispatch hover events
                    await self.browser.page.evaluate(f"""() => {{
                        const el = document.querySelector('{target}');
                        if (el) {{
                            el.scrollIntoView({{behavior: 'instant', block: 'center'}});
                            const rect = el.getBoundingClientRect();
                            const opts = {{bubbles: true, clientX: rect.x + rect.width/2, clientY: rect.y + rect.height/2}};
                            el.dispatchEvent(new MouseEvent('mouseenter', opts));
                            el.dispatchEvent(new MouseEvent('mouseover', opts));
                            el.dispatchEvent(new MouseEvent('mousemove', opts));
                        }}
                    }}""")
                    await asyncio.sleep(1.5)
                    return f"js-hovered {target} for 1.5s"
                return "hover (no target)"

            elif atype == ActionType.KEYBOARD:
                if value:
                    # Value like "Control+A" or "Shift+K"
                    keys = [k.strip() for k in value.split(",")]
                    await self.browser.page.evaluate("() => document.body.focus()")
                    for key in keys:
                        await self.browser.page.keyboard.press(key.strip())
                        await asyncio.sleep(0.3)
                    return f"pressed keys: {value}"
                return "keyboard (no value)"

            elif atype == ActionType.WAIT:
                await asyncio.sleep(1.0)
                return "waited 1s"

            elif atype == ActionType.EXTRACT_CODE:
                return "extract_code (handled in main loop)"

            elif atype == ActionType.CANVAS_DRAW:
                await self._try_canvas_challenge()
                return "canvas_draw executed"

        except Exception as e:
            return f"action error: {e}"

        return f"unknown action: {atype}"