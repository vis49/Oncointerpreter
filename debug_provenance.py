"""Read-only corpus diagnostic: real BGE/default FAISS queries, metadata-only preview.

Fetches only dictionary pages actually retrieved by the three queries. Does not
invoke Together, embed documents again, save FAISS, or modify faiss_index/.
"""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle

import faiss
import torch
from flask import Flask, render_template
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from debug_scrape import DocstoreUnpickler
from nci_chromium_loader import NciReadyChromiumLoader
from source_provenance import prepare_provenance, annotate_chunks, display_sources

ROOT = Path(__file__).resolve().parent
QUERIES = ("BRAF", "vemurafenib", "melanoma")


def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "faiss_index").iterdir() if p.is_file()}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    out = ROOT / "provenance_diagnostics" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    before = hashes()
    print("Artifacts:", out, flush=True)
    report = {"queries": {}, "pages": {}}
    try:
        with (ROOT / "faiss_index/index.pkl").open("rb") as handle:
            docstore, mapping = DocstoreUnpickler(handle).load()
        vectors = faiss.read_index(str(ROOT / "faiss_index/index.faiss"))
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")},
            encode_kwargs={"normalize_embeddings": True})
        baseline = FAISS(embeddings, vectors, docstore, mapping)
        retrieved_before = {q: baseline.as_retriever().invoke(q) for q in QUERIES}
        urls = sorted({d.metadata["source"] for docs in retrieved_before.values() for d in docs
                       if "/publications/dictionaries/" in d.metadata.get("source", "")})
        preview_docs = {key: deepcopy(value) for key, value in docstore._dict.items()}
        for page_number, url in enumerate(urls):
            print("Annotating retrieved source:", url, flush=True)
            raw = NciReadyChromiumLoader([url]).load()
            (out / f"page_{page_number:02d}.html").write_text(raw[0].page_content, encoding="utf-8")
            text = Html2TextTransformer().transform_documents(raw)
            old_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20).split_documents(text)
            pages = prepare_provenance(raw, text)
            new_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20, add_start_index=True).split_documents(text)
            annotate_chunks(new_chunks, pages)
            assert [d.page_content for d in old_chunks] == [d.page_content for d in new_chunks]
            current = [(key, docstore._dict[key]) for key in mapping.values() if docstore._dict[key].metadata.get("source") == url]
            exact_page = [d.page_content for _, d in current] == [d.page_content for d in new_chunks]
            candidates = defaultdict(list)
            for chunk in new_chunks:
                candidates[chunk.page_content].append(chunk)
            transferred = 0
            for i, (key, old) in enumerate(current):
                choices = [new_chunks[i]] if exact_page else candidates[old.page_content]
                # Never guess an entry for an ambiguous repeated chunk.
                if len(choices) == 1:
                    preview_docs[key].metadata = deepcopy(choices[0].metadata)
                    transferred += 1
            report["pages"][url] = {
                "matched_entries": len(pages.get(url, {}).get("spans", [])),
                "unmatched_entries": pages.get(url, {}).get("unmatched_entries", 0),
                "chunks": len(new_chunks), "chunk_text_unchanged": True,
                "exact_match_to_cached_page": exact_page, "cached_chunks_annotated": transferred,
            }
            print(json.dumps(report["pages"][url]), flush=True)

        # Verify metadata serialization without writing or loading any vector index.
        with (out / "preview_docstore.pkl").open("wb") as handle:
            pickle.dump((InMemoryDocstore(preview_docs), mapping), handle)
        with (out / "preview_docstore.pkl").open("rb") as handle:
            preview_docstore, preview_mapping = DocstoreUnpickler(handle).load()
        preview = FAISS(embeddings, vectors, preview_docstore, preview_mapping)
        assert all(docstore._dict[key].page_content == preview_docstore._dict[key].page_content for key in docstore._dict)
        app = Flask(__name__, template_folder=str(ROOT / "templates"))
        app.add_url_rule('/', endpoint='index', view_func=lambda: '')
        for query in QUERIES:
            retrieved = preview.as_retriever().invoke(query)
            old = retrieved_before[query]
            assert [(d.page_content, d.metadata["source"]) for d in old] == [(d.page_content, d.metadata["source"]) for d in retrieved]
            citations = display_sources(retrieved)
            report["queries"][query] = {
                "retrieved_text_and_order_unchanged": True,
                "before_sources": [d.metadata["source"] for d in old],
                "after_sources": citations,
                "chunks": [{"metadata": d.metadata, "characters": len(d.page_content),
                            "text": d.page_content} for d in retrieved],
            }
            with app.test_request_context():
                html = render_template("result.html", sources=citations, query=query,
                    answer="Diagnostic preview of retrieved sources; no LLM answer generated.",
                    filename=None, time_taken="Diagnostic run")
            (out / f"{query}_sources.html").write_text(html, encoding="utf-8")
            print(query, json.dumps(citations), flush=True)
        report["vector_index_reused_without_modification"] = True
    finally:
        report["saved_index_hashes_before"] = before
        report["saved_index_hashes_after"] = hashes()
        report["saved_index_unchanged"] = before == report["saved_index_hashes_after"]
        write_json(out / "report.json", report)
        assert report["saved_index_unchanged"]


if __name__ == "__main__":
    main()
