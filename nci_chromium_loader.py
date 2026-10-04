"""Wait for NCI cancer-terms expand pages to finish rendering."""
from urllib.parse import urlparse

from langchain_community.document_loaders import AsyncChromiumLoader
from playwright.async_api import async_playwright

DICTIONARY_READY = r"""() => {
    const root = document.querySelector('#NCI-glossary-app-root');
    const match = root?.querySelector('.dictionary-list-container h4')
        ?.textContent.match(/^([\d,]+) results? found for:/);
    if (!match) return false;
    const expected = Number(match[1].replaceAll(',', ''));
    const terms = root.querySelectorAll('dl.dictionary-list dt dfn');
    const definitions = root.querySelectorAll('dl.dictionary-list dd.definition');
    return expected > 0 && terms.length === expected &&
        definitions.length === expected &&
        [...definitions].every(d => d.textContent.trim().length > 0);
}"""


class NciReadyChromiumLoader(AsyncChromiumLoader):
    """Wait for complete NCI term lists; delegate other URLs to the base loader."""

    async def ascrape_playwright(self, url):
        parsed = urlparse(url)
        if not (parsed.hostname == "www.cancer.gov" and
                parsed.path.startswith("/publications/dictionaries/cancer-terms/expand/")):
            return await super().ascrape_playwright(url)
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(url)
                await page.wait_for_function(DICTIONARY_READY, timeout=45000)
                return await page.content()
            finally:
                await browser.close()
