import asyncio
import re
from playwright.async_api import async_playwright, Page, Browser
from typing import Any


class BrowserController:
    def __init__(self):
        self.browser: Browser | None = None
        self.page: Page | None = None
        self.playwright = None
        self.intercepted_codes: set[str] = set()
        self._code_re = re.compile(r'\b[A-Z0-9]{6}\b')

    async def start(self, url: str, headless: bool = False) -> None:
        """Launch browser and navigate to URL."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=headless)
        self.page = await self.browser.new_page()
        await self.page.set_viewport_size({"width": 1280, "height": 800})

        # Inject audio interception BEFORE page scripts run.
        # This catches SpeechSynthesis, Audio(), and blob audio from auto-play.
        await self.page.add_init_script("""
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
        """)

        await self.page.goto(url)

        # Intercept network responses to capture codes from API calls
        self.page.on("response", self._on_response)

    async def _on_response(self, response):
        """Capture 6-char codes from network responses."""
        try:
            ct = response.headers.get("content-type", "")
            if "json" in ct or "text" in ct or "javascript" in ct:
                body = await response.text()
                if body and len(body) < 50000:
                    matches = self._code_re.findall(body.upper())
                    self.intercepted_codes.update(matches)
        except Exception:
            pass

    async def stop(self) -> None:
        """Close browser."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def screenshot(self) -> bytes:
        """Take screenshot of current page."""
        return await self.page.screenshot(type="png")

    async def get_html(self) -> str:
        """Get page HTML."""
        return await self.page.content()

    async def get_url(self) -> str:
        """Get current URL."""
        return self.page.url

    async def click(self, selector: str) -> bool:
        """Click element by selector. Returns success."""
        try:
            await self.page.click(selector, timeout=2000)
            return True
        except Exception:
            return False

    async def click_by_text(self, text: str) -> bool:
        """Click element containing text."""
        try:
            await self.page.click(f"text={text}", timeout=2000)
            return True
        except Exception:
            return False

    async def type_text(self, selector: str, text: str) -> bool:
        """Type text into input field."""
        try:
            await self.page.fill(selector, text)
            return True
        except Exception:
            return False

    async def scroll_to_bottom(self) -> None:
        """Scroll to page bottom."""
        await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

    async def close_popup_by_x(self) -> bool:
        """Try to close popup by clicking X button."""
        # Try various X button selectors
        selectors = [
            "button:has(img[alt*='close'])",
            "[aria-label*='close']",
            "[aria-label*='Close']",
            ".close-button",
            ".close",
            "button:has-text('×')",
            "button:has-text('X')",
        ]
        for sel in selectors:
            try:
                await self.page.click(sel, timeout=500)
                return True
            except Exception:
                continue
        return False

    async def wait_for_navigation(self, timeout: int = 5000) -> bool:
        """Wait for navigation to complete."""
        try:
            await self.page.wait_for_load_state("networkidle", timeout=timeout)
            return True
        except Exception:
            return False

    async def execute_js(self, script: str) -> Any:
        """Execute JavaScript on page."""
        return await self.page.evaluate(script)

    async def wait_for_selector(self, selector: str, timeout: int = 5000) -> bool:
        """Wait for element to appear."""
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception:
            return False