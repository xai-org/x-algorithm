import asyncio
import logging
import time
from typing import Optional, List
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from playwright.async_api import async_playwright, Browser, BrowserContext

logger = logging.getLogger(__name__)


@dataclass
class BrowserPool:
    browsers: List[Browser]
    contexts: List[BrowserContext]
    available_contexts: asyncio.Queue = field(init=False)

    def __post_init__(self):
        self.available_contexts = asyncio.Queue()


class TweetRenderForGrox:
    def __init__(self, pool_size: int = 2, max_contexts_per_browser: int = 5):
        self.pool_size = pool_size
        self.max_contexts_per_browser = max_contexts_per_browser
        self.browser_pool: Optional[BrowserPool] = None
        self._playwright = None
        self._initialization_lock = asyncio.Lock()
        self._initialized = False

    async def _initialize_pool(self):
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:
                return
            self._playwright = await async_playwright().start()
            browsers = []
            contexts = []

            for _ in range(self.pool_size):
                browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-gpu",
                        "--disable-dev-shm-usage",
                        "--disable-features=TranslateUI",
                        "--disable-ipc-flooding-protection",
                    ],
                )
                browsers.append(browser)
                for _ in range(self.max_contexts_per_browser):
                    context = await browser.new_context(
                        viewport={"width": 1200, "height": 800}
                    )
                    contexts.append(context)

            self.browser_pool = BrowserPool(browsers=browsers, contexts=contexts)
            for context in contexts:
                await self.browser_pool.available_contexts.put(context)

            self._initialized = True

    @asynccontextmanager
    async def get_browser_context(self):
        if not self._initialized:
            await self._initialize_pool()

        context = await asyncio.wait_for(
            self.browser_pool.available_contexts.get(), timeout=30.0
        )
        try:
            yield context
        finally:
            try:
                for page in list(context.pages):
                    try:
                        if not page.is_closed():
                            await page.close()
                    except Exception as e:
                        logger.debug(f"Error closing page: {e}")
            except Exception as e:
                logger.warning(f"Error cleaning up context: {e}")
            finally:
                self.browser_pool.available_contexts.put_nowait(context)

    def get_embed_html(self, tweet_id: str, theme: str = "light") -> str:
        bg_color = "#15202b" if theme == "dark" else "#ffffff"
        return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta name="color-scheme" content="{theme}">
    </head>
    <body style="margin:0; padding:20px; background:{bg_color};">
        <div style="max-width:600px; margin:0 auto;">
            <blockquote class="twitter-tweet"
                        data-dnt="true"
                        data-theme="{theme}"
                        data-conversation="none">  <!-- Hides thread/replies if unwanted -->
                <a href="https://twitter.com/dummy/status/{tweet_id}"></a>
            </blockquote>
            <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
        </div>
    </body>
    </html>
    """

    async def take_screenshot(
        self, tweet_id: str, theme: str = "light"
    ) -> Optional[bytes]:
        logger.info(f"Taking screenshot for tweet_id: {tweet_id}")
        start = time.perf_counter()

        try:
            async with self.get_browser_context() as context:
                page = await context.new_page()
                try:
                    await page.set_content(self.get_embed_html(tweet_id, theme=theme))
                    await page.wait_for_load_state("networkidle", timeout=10000)

                    iframe_locator = page.frame_locator(
                        'iframe.twitter-widget, iframe[src*="platform."]'
                    )
                    tweet_locator = iframe_locator.locator(
                        f'article[role="article"]:has(a[href*="/status/{tweet_id}"])'
                    )

                    await tweet_locator.wait_for(state="visible", timeout=10000)

                    screenshot_bytes = await tweet_locator.screenshot(
                        type="jpeg", quality=85
                    )

                    logger.info(
                        f"Screenshot taken in {time.perf_counter() - start:.2f}s"
                    )

                    return screenshot_bytes
                finally:
                    if not page.is_closed():
                        await page.close()
        except Exception as e:
            logger.warning(f"Error screenshotting {tweet_id}: {str(e)}")
            return None

    async def cleanup(self):
        if not self.browser_pool:
            return

        for context in self.browser_pool.contexts:
            try:
                await context.close()
            except Exception as e:
                logger.debug(f"Error closing context: {e}")

        for browser in self.browser_pool.browsers:
            try:
                await browser.close()
            except Exception as e:
                logger.debug(f"Error closing browser: {e}")

        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as e:
                logger.debug(f"Error stopping playwright: {e}")

        self._initialized = False
        self.browser_pool = None

    async def __aenter__(self):
        await self._initialize_pool()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
