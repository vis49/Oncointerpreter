"""Temporary single-URL ingestion diagnostic. Never imports rag_pipeline or saves FAISS.

Default: exact installed loader, then an instrumented browser comparison.
--build-faiss optionally checks an IN-MEMORY index with the original BGE model.
Artifacts are written to a new timestamped debug_scrape_runs subdirectory.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import pickle
from pathlib import Path
import time
from datetime import datetime
from urllib.parse import urlparse

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from playwright.async_api import async_playwright
from nci_chromium_loader import DICTIONARY_READY, NciReadyChromiumLoader

URL = "https://www.cancer.gov/publications/dictionaries/cancer-terms/expand/B"
NEEDLES = ("BRAF", "BRAF gene", "BRAF V600")
ROOT = Path(__file__).resolve().parent

class DocstoreUnpickler(pickle.Unpickler):
    """Read the local docstore without permitting arbitrary pickle globals."""

    def find_class(self, module, name):
        from langchain_community.docstore.in_memory import InMemoryDocstore
        allowed = {
            ("langchain_community.docstore.in_memory", "InMemoryDocstore"): InMemoryDocstore,
            ("langchain_core.documents.base", "Document"): Document,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unexpected pickle global: {module}.{name}")
        return allowed[(module, name)]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def index_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "faiss_index").glob("*") if p.is_file()}


def summarize(text):
    return {"characters": len(text),
            "exact_counts": {s: text.count(s) for s in NEEDLES},
            "case_insensitive_counts": {s: text.lower().count(s.lower()) for s in NEEDLES},
            "beginning": text[:500],
            "middle": text[max(0, len(text)//2-250):len(text)//2+250],
            "end": text[-500:]}


def stage(out, name, docs):
    folder = out / name
    folder.mkdir()
    rows = []
    for i, doc in enumerate(docs):
        (folder / f"{i:04d}.txt").write_text(doc.page_content, encoding="utf-8")
        rows.append({"position": i, "metadata": doc.metadata, **summarize(doc.page_content)})
    summary = {"documents": len(docs), "characters": sum(len(d.page_content) for d in docs),
               "exact_counts": {s: sum(d.page_content.count(s) for d in docs) for s in NEEDLES},
               "sources": sorted({str(d.metadata.get('source')) for d in docs})}
    write_json(folder / "summary.json", {**summary, "items": rows})
    print(name, json.dumps(summary), flush=True)
    if rows:
        print("  Representative beginning/middle/end:",
              json.dumps({k: rows[0][k] for k in ("beginning", "middle", "end")}), flush=True)
    return summary


def pipeline(out, name, docs):
    raw = stage(out, name + "_raw", docs)
    transformed = Html2TextTransformer().transform_documents(docs)
    text = stage(out, name + "_text", transformed)
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20).split_documents(transformed)
    chunk_summary = stage(out, name + "_chunks_faiss_input", chunks)
    return chunks, {"raw": raw, "text": text, "chunks_faiss_input": chunk_summary}


async def observe(out):
    """Match the loader's capture point; then observe a content-based condition.

    The initial BRAF condition is diagnostic, NOT a generic proposed fix.
    """
    start = time.perf_counter()
    events = []
    tasks = []
    def event(kind, **values):
        events.append({"seconds": round(time.perf_counter()-start, 4), "kind": kind, **values})

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        async def response_seen(response):
            req = response.request
            event("response", url=response.url, status=response.status, resource_type=req.resource_type)
            if req.resource_type in ("xhr", "fetch") and urlparse(response.url).hostname == "webapis.cancer.gov":
                try:
                    body = await response.text()
                    n = len(list(out.glob("response_*.txt")))
                    file = f"response_{n:03d}.txt"
                    (out / file).write_text(body, encoding="utf-8")
                    event("response_body", url=response.url, file=file, **summarize(body))
                except Exception as exc:
                    event("response_body_error", url=response.url, error=str(exc))
        page.on("request", lambda req: event("request", url=req.url, resource_type=req.resource_type))
        page.on("response", lambda response: tasks.append(asyncio.create_task(response_seen(response))))
        page.on("requestfailed", lambda req: event("requestfailed", url=req.url, error=req.failure))
        page.on("pageerror", lambda exc: event("pageerror", error=str(exc)))
        page.on("domcontentloaded", lambda: event("domcontentloaded"))
        page.on("load", lambda: event("load"))
        try:
            response = await page.goto(URL)
            event("goto_returned", status=response.status, url=page.url)
            immediate = await page.content()
            event("immediate_capture", **summarize(immediate))
            (out / "navigation_response.html").write_text(await response.text(), encoding="utf-8")
            (out / "browser_at_load.html").write_text(immediate, encoding="utf-8")
            try:
                await page.wait_for_function("() => document.body.innerText.includes('BRAF gene')", timeout=45000)
                event("braf_visible")
                await page.wait_for_function(DICTIONARY_READY, timeout=45000)
                event("all_definitions_ready")
            except Exception as exc:
                event("readiness_error", error=str(exc))
            ready = await page.content()
            (out / "browser_after_condition.html").write_text(ready, encoding="utf-8")
            write_json(out / "dictionary_dom.json", await page.evaluate("""() => ({
                title: document.title,
                terms: document.querySelectorAll('#NCI-glossary-app-root dl.dictionary-list dt dfn').length,
                definitions: document.querySelectorAll('#NCI-glossary-app-root dl.dictionary-list dd.definition').length,
                matches: [...document.querySelectorAll('a')].filter(a => a.textContent.includes('BRAF')).map(a => ({text:a.textContent, html:a.parentElement.outerHTML})),
                scripts: [...document.scripts].map(s => ({src:s.src, text:s.src ? '' : s.textContent})),
                headings: [...document.querySelectorAll('h1,h2,h3,h4')].map(x => ({text:x.textContent, html:x.outerHTML}))
            })"""))
            if tasks:
                await asyncio.gather(*tasks)
            return [Document(page_content=immediate, metadata={"source": URL})], [Document(page_content=ready, metadata={"source": URL})]
        finally:
            write_json(out / "browser_events.json", events)
            await browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-faiss", action="store_true", help="Use unchanged BGE in memory; never save an index")
    args = parser.parse_args()
    out = ROOT / "debug_scrape_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    print("Artifacts:", out, flush=True)
    before = index_hashes()
    write_json(out / "index_hashes_before.json", before)
    write_json(out / "versions.json", {name: importlib.metadata.version(name) for name in
               ("langchain", "langchain-community", "playwright", "html2text", "faiss-cpu", "sentence-transformers")})
    (out / "installed_loader.py.txt").write_text(inspect.getsource(AsyncChromiumLoader), encoding="utf-8")
    report = {}
    try:
        cached = ROOT / "faiss_index" / "index.pkl"
        if cached.exists():
            with cached.open("rb") as handle:
                docstore, mapping = DocstoreUnpickler(handle).load()
            stored = [docstore.search(mapping[i]) for i in sorted(mapping)]
            cached_b = [d for d in stored if d.metadata.get("source") == URL]
            report["cached_B"] = stage(out, "cached_B_docstore", cached_b)
            report["cached_all"] = {"documents": len(stored), "braf_hits": [
                {"metadata": d.metadata, "content": d.page_content}
                for d in stored if "BRAF" in d.page_content]}
        docs = AsyncChromiumLoader([URL]).load()
        baseline_chunks, report["installed_loader"] = pipeline(out, "installed_loader", docs)
        at_load, after_condition = asyncio.run(observe(out))
        _, report["browser_at_load"] = pipeline(out, "browser_at_load", at_load)
        ready_chunks, report["browser_after_condition"] = pipeline(out, "browser_after_condition", after_condition)
        fixed_docs = NciReadyChromiumLoader([URL]).load()
        ready_chunks, report["candidate_fix"] = pipeline(out, "candidate_fix", fixed_docs)
        assert all(s in fixed_docs[0].page_content for s in NEEDLES), "Candidate did not capture expected B entries"
        if args.build_faiss:
            import torch
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
            from langchain_community.vectorstores import FAISS
            embeddings = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-large-en-v1.5",
                model_kwargs={"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")},
                encode_kwargs={"normalize_embeddings": True})
            for name, chunks in (("baseline", baseline_chunks), ("ready", ready_chunks)):
                print("Building in-memory FAISS:", name, flush=True)
                db = FAISS.from_documents(chunks, embeddings)
                stored = [db.docstore.search(db.index_to_docstore_id[i]) for i in range(db.index.ntotal)]
                assert [(d.page_content, d.metadata) for d in stored] == [(d.page_content, d.metadata) for d in chunks]
                report[name + "_faiss"] = stage(out, name + "_faiss_docstore", stored)
                report[name + "_faiss"]["ntotal"] = db.index.ntotal
        else:
            report["faiss_note"] = "Captured exact pre-embedding input; actual FAISS insertion not executed. Use --build-faiss."
    finally:
        after = index_hashes()
        write_json(out / "index_hashes_after.json", after)
        report["saved_index_unchanged"] = before == after
        write_json(out / "report.json", report)
        assert before == after, "Existing index files changed during diagnostic"
        print("Saved index unchanged:", before == after, flush=True)


if __name__ == "__main__":
    main()
