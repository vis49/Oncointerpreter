"""DOM-derived source metadata without changing page text or chunk boundaries."""
from bisect import bisect_right
import hashlib
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_transformers import Html2TextTransformer


def safe_url(value):
    return value if isinstance(value, str) and urlparse(value).scheme in ("https", "http") else ""


def prepare_provenance(raw_documents, transformed_documents):
    """Read entry labels/links before HTML is lost; locate spans in original text.

    Whitespace normalization is ONLY for matching; page_content is never edited.
    An unlocatable entry is omitted rather than guessed from a chunk's meaning.
    """
    pages = {}
    for raw, transformed in zip(raw_documents, transformed_documents):
        source = raw.metadata.get("source", "")
        parsed = urlparse(source)
        if parsed.hostname != "www.cancer.gov" or not parsed.path.startswith("/publications/dictionaries/"):
            continue
        soup = BeautifulSoup(raw.page_content, "html.parser")
        heading = soup.find("h1")
        collection = heading.get_text(" ", strip=True) if heading else "NCI Dictionary"
        provenance = {"version": 1, "source_type": "nci_dictionary", "collection": collection}
        transformed.metadata["provenance"] = provenance
        tokens = list(re.finditer(r"\S+", transformed.page_content))
        normalized = " ".join(token.group() for token in tokens)
        normalized_starts = []
        offset = 0
        for token in tokens:
            normalized_starts.append(offset)
            offset += len(token.group()) + 1

        def original_offset(position):
            index = bisect_right(normalized_starts, position) - 1
            return tokens[index].start() + position - normalized_starts[index]

        spans, cursor, unmatched = [], 0, 0
        transformer = Html2TextTransformer()
        for term in soup.select("dl.dictionary-list dt dfn"):
            link = term.find("a", href=True)
            block = term.find_parent("dt").parent
            if not link or not block.select_one("dd.definition"):
                continue
            title = term.get_text(" ", strip=True)
            entry_url = safe_url(urljoin(source, link["href"]))
            if not title or not entry_url:
                continue
            rendered = transformer.transform_documents([Document(page_content=str(block))])[0].page_content
            needle = " ".join(rendered.split())
            position = normalized.find(needle, cursor) if needle else -1
            if position < 0:
                unmatched += 1
                continue
            end = position + len(needle)
            spans.append({
                "start": original_offset(position), "end": original_offset(end - 1) + 1,
                "entry_title": title, "entry_url": entry_url,
                "entry_id": term.get("data-cdr-id") or entry_url,
            })
            cursor = end
        pages[source] = {"spans": spans, "unmatched_entries": unmatched}
    return pages


def annotate_chunks(chunks, pages):
    """Assign every overlapping entry, including both sides of a boundary chunk."""
    for chunk in chunks:
        # This offset is supplied by LangChain; keep it with provenance only.
        start = chunk.metadata.pop("start_index", -1)
        source = chunk.metadata.get("source", "")
        if source not in pages:
            continue
        provenance = chunk.metadata["provenance"]
        end = start + len(chunk.page_content)
        provenance["entries"] = [
            {key: span[key] for key in ("entry_title", "entry_url", "entry_id")}
            for span in pages[source]["spans"]
            if start >= 0 and start < span["end"] and end > span["start"]
        ]
        provenance["start_index"] = start
        provenance["chunk_id"] = hashlib.sha256(
            f"{source}\0{start}\0{chunk.page_content}".encode("utf-8")
        ).hexdigest()[:20]
        provenance["mapping_status"] = "entry_overlap" if provenance["entries"] else "page_only"
    return chunks


def display_sources(documents):
    """Deduplicate citations in retrieval order, without filtering Documents."""
    sources, seen = [], set()
    for document in documents:
        metadata = document.metadata
        source = safe_url(metadata.get("source"))
        provenance = metadata.get("provenance", {})
        collection = provenance.get("collection", "")
        entries = provenance.get("entries", [])
        for entry in entries or [{}]:
            title = entry.get("entry_title", "")
            url = safe_url(entry.get("entry_url")) or source
            identity = entry.get("entry_id") or safe_url(entry.get("entry_url")) or (source, title)
            key = ("entry", collection, identity) if title else ("page", source)
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "label": f"{collection} — {title}" if collection and title else title or collection or source or "Source unavailable",
                "url": url, "source_url": source, "entry_title": title,
                "collection": collection, "source_type": provenance.get("source_type", "webpage"),
            })
    return sources


def invoke_with_baseline_context(llm_chain, inputs):
    """Prevent new metadata from changing the existing Document-list prompt.

    LLMChain formats the original list of Documents via str(); extra metadata
    would change that string. Preserve original metadata for the LLM, then return
    the retrieved rich Documents for source display.
    """
    original_context = inputs["context"]
    baseline_context = [Document(page_content=d.page_content, metadata={
        key: value for key, value in d.metadata.items() if key != "provenance"
    }) for d in original_context]
    answer = llm_chain.invoke({**inputs, "context": baseline_context})
    answer["context"] = original_context
    return answer
