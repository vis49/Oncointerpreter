"""Exercise real Flask routes and templates with a local, deterministic LLM fixture."""
import hashlib
import json
from pathlib import Path
import threading
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_community.llms.fake import FakeListLLM
from langchain.schema.runnable import RunnableLambda
from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server
import page

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'ui_diagnostics'


def index_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'faiss_index').glob('*') if p.is_file()}


def main():
    OUT.mkdir(exist_ok=True)
    before = index_hashes()
    baseline = OUT / 'baseline/mistral.py'
    baseline_matches = (ROOT / 'rag_pipeline.py').read_bytes() == baseline.read_bytes() if baseline.exists() else None
    if baseline_matches is not None:
        assert baseline_matches
    document = Document(page_content='A local UI test fixture.', metadata={
        'source': 'https://www.cancer.gov/publications/dictionaries/cancer-terms/expand/B',
        'provenance': {'collection': 'NCI Dictionary of Cancer Terms', 'source_type': 'nci_dictionary',
            'entries': [{'entry_title': 'BRAF gene', 'entry_id': '561325',
                         'entry_url': 'https://www.cancer.gov/publications/dictionaries/cancer-terms/def/braf-gene'}]}})
    db = Mock()
    db.as_retriever.return_value = RunnableLambda(lambda query: [document])
    llm = FakeListLLM(responses=['## Research context\n\nThis is a **UI verification response** using a local fixture.\n\nThe source below demonstrates the existing reference display.'])
    checks = []
    server = make_server('127.0.0.1', 0, page.app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with patch.object(page, 'load_tokenizer_and_llm', return_value=llm) as load_llm, patch.object(page, 'load_data', return_value=db) as load_data:
            client = page.app.test_client()
            for path in ('/', '/upload_pdf', '/upload_image', '/static/app.css', '/static/app.js', '/static/upload.js'):
                assert client.get(path).status_code == 200, path
            checks.append('Real application imports and all page/static GET routes return 200')
            for mode, expected in (('existing', False), ('rebuild', True)):
                response = client.post('/', data={'query': 'UI regression question', 'index_mode': mode})
                assert response.status_code == 200
                load_data.assert_called_with(force_rebuild=expected)
                assert b'NCI Dictionary of Cancer Terms' in response.data
                assert b'/def/braf-gene' in response.data
            checks.append('Both index options reach the unchanged pipeline; generated answer and sources render')
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                tab = browser.new_page(viewport={'width': 1280, 'height': 920}, device_scale_factor=1)
                errors = []
                tab.on('pageerror', lambda error: errors.append(str(error)))
                for width, height, label in ((1280, 920, 'desktop'), (390, 844, 'mobile')):
                    tab.set_viewport_size({'width': width, 'height': height})
                    tab.goto(base)
                    assert tab.locator('input[name="model"]').count() == 0
                    assert tab.locator('input[name="knowledgeBase"]').count() == 0
                    assert tab.locator('#index-existing').is_checked()
                    assert tab.evaluate('document.documentElement.scrollWidth <= innerWidth')
                    tab.screenshot(path=str(OUT / f'{label}_input.png'), full_page=True)
                    tab.locator('#query').fill('What is BRAF?')
                    tab.get_by_role('button', name='Analyze').click()
                    tab.get_by_role('heading', name='Your results', exact=True).wait_for()
                    assert tab.locator('.result-text').inner_text().find('UI verification response') >= 0
                    assert tab.locator('.source-list a').get_attribute('href').endswith('/def/braf-gene')
                    assert tab.evaluate('document.documentElement.scrollWidth <= innerWidth')
                    tab.screenshot(path=str(OUT / f'{label}_results.png'), full_page=True)
                    tab.get_by_role('link', name='New question', exact=True).click()
                    tab.locator('#query').wait_for()
                checks.append('Desktop/mobile browser input → Analyze → results → new question; no horizontal overflow')
                load_llm.side_effect = page.TogetherKeyConfigurationError('Together AI key is missing. Save your key, then restart the app.')
                tab.locator('#query').fill('Preserve my question')
                tab.locator('summary').click()
                tab.locator('#index-rebuild').check()
                tab.get_by_role('button', name='Analyze').click()
                tab.locator('[role="alert"]').wait_for()
                assert tab.locator('#query').input_value() == 'Preserve my question'
                assert tab.locator('#index-rebuild').is_checked()
                assert tab.get_by_role('button', name='Analyze').is_enabled()
                tab.screenshot(path=str(OUT / 'mobile_error.png'), full_page=True)
                assert not errors, errors
                checks.append('Missing-key message preserves question and index choice; zero browser JavaScript errors')
                browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        after = index_hashes()
        assert before == after
        (OUT / 'verification.json').write_text(json.dumps({
            'checks': checks, 'saved_index_unchanged': before == after,
            'rag_pipeline_byte_identical_to_before_rename': baseline_matches,
            'api_calls': 0, 'index_hashes': after,
        }, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
