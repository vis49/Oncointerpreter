"""Focused provenance tests: preserve retrieval inputs and the LLM's context."""
from copy import deepcopy
from pathlib import Path
import unittest
import time
from unittest.mock import Mock

from flask import Flask, render_template
from langchain_core.documents import Document
from langchain_core.callbacks import BaseCallbackHandler
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.llms.fake import FakeListLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from source_provenance import prepare_provenance, annotate_chunks, display_sources, invoke_with_baseline_context
from test_index_options import isolated_function

URL = "https://www.cancer.gov/publications/dictionaries/cancer-terms/expand/B"
ROOT = Path(__file__).resolve().parent


def page(definition="First definition."):
    return Document(page_content=f'''<h1>NCI Dictionary of Cancer Terms</h1>
      <dl class="dictionary-list">
        <div><dt><dfn data-cdr-id="1"><a href="/publications/dictionaries/cancer-terms/def/braf-gene">BRAF gene</a></dfn></dt>
          <dd class="definition">{definition}</dd></div>
        <div><dt><dfn data-cdr-id="2"><a href="/publications/dictionaries/cancer-terms/def/braf-v600-mutation">BRAF V600 mutation</a></dfn></dt>
          <dd class="definition">Second definition.</dd></div>
      </dl><footer>Contact us</footer>''', metadata={"source": URL, "original_field": "kept"})


def ingest(raw):
    transformed = Html2TextTransformer().transform_documents([raw])
    baseline = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20).split_documents(transformed)
    pages = prepare_provenance([raw], transformed)
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20, add_start_index=True).split_documents(transformed)
    annotate_chunks(chunks, pages)
    return baseline, chunks, pages, transformed


class PromptRecorder(BaseCallbackHandler):
    def __init__(self):
        self.prompts = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.prompts.extend(prompts)


class SourceProvenanceTests(unittest.TestCase):
    def test_same_chunk_text_and_original_metadata(self):
        baseline, chunks, pages, transformed = ingest(page("A long definition without an entry name. " * 40))
        self.assertEqual([d.page_content for d in baseline], [d.page_content for d in chunks])
        self.assertEqual(len(pages[URL]["spans"]), 2)
        for old, new in zip(baseline, chunks):
            self.assertEqual(old.metadata, {k: v for k, v in new.metadata.items() if k != "provenance"})
            self.assertGreaterEqual(new.metadata["provenance"]["start_index"], 0)
        continuation = next(d for d in chunks if "BRAF" not in d.page_content and d.metadata["provenance"]["entries"])
        self.assertEqual(continuation.metadata["provenance"]["entries"][0]["entry_title"], "BRAF gene")
        self.assertEqual(continuation.metadata["provenance"]["source_type"], "nci_dictionary")
        self.assertNotIn("entries", transformed[0].metadata["provenance"])
        self.assertIsNot(chunks[0].metadata["provenance"], chunks[1].metadata["provenance"])

    def test_cross_entry_chunk_preserves_both_and_deduplicates_repeats(self):
        _, chunks, _, _ = ingest(page())
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0].metadata["provenance"]["entries"]), 2)
        documents = chunks + deepcopy(chunks)
        before = deepcopy(documents)
        citations = display_sources(documents)
        self.assertEqual([c["entry_title"] for c in citations], ["BRAF gene", "BRAF V600 mutation"])
        self.assertTrue(all(c["source_url"] == URL for c in citations))
        self.assertTrue(citations[0]["url"].endswith("/def/braf-gene"))
        self.assertEqual(documents, before)

        renamed_copy = deepcopy(chunks[0])
        renamed_copy.metadata["provenance"]["entries"][0]["entry_title"] = "Alternate display label"
        self.assertEqual(len(display_sources(chunks + [renamed_copy])), 2)

    def test_unmatched_structure_does_not_invent_title(self):
        raw = page()
        transformed = [Document(page_content="BRAF has semantic relevance but this is not the rendered entry.", metadata={"source": URL})]
        pages = prepare_provenance([raw], transformed)
        self.assertEqual(pages[URL]["unmatched_entries"], 2)
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20, add_start_index=True).split_documents(transformed)
        annotate_chunks(chunks, pages)
        self.assertEqual(chunks[0].metadata["provenance"]["entries"], [])
        self.assertEqual(display_sources(chunks)[0]["url"], URL)

    def test_non_dictionary_text_and_metadata_unchanged(self):
        raw = Document(page_content="<p>General cancer information.</p>", metadata={"source": "https://www.cancer.org/cancer.html"})
        baseline, chunks, pages, _ = ingest(raw)
        self.assertEqual(pages, {})
        self.assertEqual(baseline, chunks)

    def test_legacy_index_displays_deduplicated_page_urls(self):
        docs = [Document(page_content="a", metadata={"source": URL}), Document(page_content="b", metadata={"source": URL})]
        self.assertEqual(len(display_sources(docs)), 1)
        self.assertEqual(display_sources(docs)[0]["label"], URL)

    def test_actual_llm_chain_prompt_is_byte_identical(self):
        baseline, rich, _, _ = ingest(page())
        recorder = PromptRecorder()
        llm = FakeListLLM(responses=["test answer"], callbacks=[recorder])
        chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template("Given context: {context}\nQuestion: {question}"))
        expected = chain.invoke({"context": baseline, "question": "BRAF"})
        actual = invoke_with_baseline_context(chain, {"context": rich, "question": "BRAF"})
        self.assertEqual(recorder.prompts[0], recorder.prompts[1])
        self.assertEqual(expected["text"], actual["text"])
        self.assertIs(actual["context"], rich)
        self.assertIn("provenance", actual["context"][0].metadata)

    def test_rendered_labels_links_and_legacy_compatibility(self):
        _, chunks, _, _ = ingest(page())
        app = Flask(__name__, template_folder=str(ROOT / "templates"))
        app.add_url_rule('/', endpoint='index', view_func=lambda: '')
        with app.test_request_context():
            html = render_template("result.html", sources=display_sources(chunks + chunks))
            legacy = render_template("result.html", sources=[URL])
        self.assertEqual(html.count("NCI Dictionary of Cancer Terms — BRAF gene"), 1)
        self.assertIn("NCI Dictionary of Cancer Terms — BRAF V600 mutation", html)
        self.assertIn('href="https://www.cancer.gov/publications/dictionaries/cancer-terms/def/braf-gene"', html)
        self.assertIn(f'href="{URL}"', legacy)

    def test_process_query_keeps_rich_context_until_display(self):
        _, rich, _, _ = ingest(page())
        db = Mock()
        db.as_retriever.return_value = RunnableLambda(lambda query: rich)
        namespace = {
            "time": time, "log": Mock(), "LLMChain": LLMChain,
            "template": PromptTemplate.from_template("{context}\n{question}"),
            "RunnableLambda": RunnableLambda, "RunnablePassthrough": RunnablePassthrough,
            "invoke_with_baseline_context": invoke_with_baseline_context,
            "process_llm_response": lambda answer: display_sources(answer["context"]),
        }
        process = isolated_function("rag_pipeline.py", "process_query", namespace)
        citations = process("BRAF", FakeListLLM(responses=["test answer"]), db)
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0]["entry_title"], "BRAF gene")
        db.as_retriever.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
