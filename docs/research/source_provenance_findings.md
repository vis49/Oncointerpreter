# Retrieved-source provenance: implementation and verification

This change preserves the current research pipeline's embedding model, 500/20
splitter settings, page text, chunk boundaries, FAISS search defaults, prompt and
LLM. It adds ingestion metadata and display-only citation deduplication. No
claim-level citations, entry-based rechunking, filtering or reranking were added.

## A. Current metadata flow

Previously:

1. AsyncChromiumLoader yields a full rendered HTML Document with `source = URL`.
2. Html2TextTransformer creates a text Document, copying metadata. Its default
   link removal discards the entry hrefs from text, while raw HTML remains available.
3. RecursiveCharacterTextSplitter copies the page metadata to every chunk.
4. FAISS embeds only page_content and stores text plus metadata in its docstore.
5. The default retriever returns those Documents; LLMChain returns its context
   together with generated text.
6. process_llm_response extracts only metadata['source']; result.html repeats each URL.

Now, after the same full-page transformation, `prepare_provenance` reads the
retained raw DOM, extracts the h1 collection name, dfn title, data-cdr-id, and
actual entry anchor href, and locates each entry in the transformed page text.
Whitespace normalization is used for matching only; no page text is rewritten.

Collection/type metadata is attached before splitting. The installed LangChain
splitter's `create_documents()` uses `copy.deepcopy` for metadata; no custom
metadata-copying splitter is needed. Its `add_start_index=True` option supplies
offsets without changing text. `annotate_chunks` uses those offsets to attach
every overlapping entry, then moves the offset into the provenance block.

FAISS persists the richer metadata with unchanged chunk text. Retrieved rich
Documents reach `process_llm_response`, which constructs distinct citation
records. The template renders a label and the entry's actual link.

## B. Root cause of coarse and duplicate sources

The loader records only the letter page URL. Entry titles, IDs and links in the
rendered DOM were never retained as metadata. Every chunk from that page thus
inherits the same source. The response formatter emits one URL per retrieved
chunk, with no presentation deduplication.

This is independent of retrieval ranking. The BRAF verification still retrieves
four chunks from B, but now displays three distinct entry citations.

## C. Minimal implementation and behavior classification

**Ingestion metadata:** one helper module reads the actual dictionary structure
and records collection, source type, entry title/link/ID, source offset and a
stable chunk identifier. A chunk spanning entries gets an entries list. Unmapped
or boilerplate chunks retain page-level provenance; titles are never inferred
from a query or semantic similarity. Existing `source` remains the original page.

**Display only:** `display_sources` retains first-seen retrieval order and
deduplicates by collection plus entry ID (or entry URL, then page/title fallback).
Multiple chunks from one entry yield one citation; distinct entries sharing a
letter page remain distinct. Legacy source-only Documents fall back to deduplicated
page links. The result template also accepts legacy URL strings from the LLaMA path.

**Retrieval:** no changes to text, ordering, vectors, search type, k, scoring,
filtering, normalization or model parameters. Splitting each entry into its own
Document was deliberately avoided because it would change chunk boundaries.

**Prompt preservation:** the existing LLMChain formats the list of Documents,
including metadata, into the context. Simply adding metadata would therefore
change the prompt. A small invocation adapter removes ONLY the newly introduced
`provenance` key from temporary copies passed to the same LLMChain, then restores
the rich retrieved context in its response for source rendering. A recording
fake LLM test verifies byte-identical formatted prompts; another exercises the
actual process_query chain. The real LLM is not called by diagnostics.

Rebuilding from a changing website can change the corpus and therefore retrieval,
regardless of metadata. That is distinct from this metadata-only implementation.

## D. Files and functions changed

- `source_provenance.py`: added `prepare_provenance` (DOM extraction and exact
  normalized span matching), `annotate_chunks` (overlap metadata), `display_sources`
  (presentation deduplication), and `invoke_with_baseline_context` (prompt preservation).
- `mistral.py`: `load_data` calls the metadata helpers and enables offset metadata;
  `process_llm_response` returns structured citations; `process_query` invokes the
  existing LLMChain through the context-preserving adapter. Imports updated.
- `templates/result.html`: labels and URLs for structured citations, legacy string
  support, and the heading "Retrieved sources".
- `debug_provenance.py`: temporary live diagnostic for BRAF, vemurafenib and melanoma.
- `test_source_provenance.py`: targeted provenance, text invariance, prompt and rendering tests.
- `test_index_options.py`: supplies the two new metadata helper mocks to existing
  isolated load_data tests; their original index/backup behavior remains tested.
- This findings document and `provenance_diagnostics/` contain evidence and baseline snapshots.

No change in this task to page.py, nci_chromium_loader.py, dependencies, the prompt,
the LLM or existing FAISS files. Earlier uncommitted work in this workspace remains.

## E. Example metadata before and after

Before:

```json
{"source": "https://www.cancer.gov/publications/dictionaries/cancer-terms/expand/B"}
```

Actual enriched metadata from a retrieved BRAF chunk:

```json
{
  "source": "https://www.cancer.gov/publications/dictionaries/cancer-terms/expand/B",
  "provenance": {
    "version": 1,
    "source_type": "nci_dictionary",
    "collection": "NCI Dictionary of Cancer Terms",
    "entries": [{
      "entry_title": "BRAF V600 mutation",
      "entry_url": "https://www.cancer.gov/publications/dictionaries/cancer-terms/def/braf-v600-mutation",
      "entry_id": "721263"
    }],
    "start_index": 135292,
    "chunk_id": "99cb3387346d85115698",
    "mapping_status": "entry_overlap"
  }
}
```

Nested provenance allows a multi-entry chunk to be described truthfully and keeps
the original metadata available unchanged for LLM context formatting.

## F. Example source display

Before, BRAF returned four identical `/expand/B` links.

After, the same four retrieved chunks display:

- NCI Dictionary of Cancer Terms — BRAF kinase inhibitor
  https://www.cancer.gov/publications/dictionaries/cancer-terms/def/braf-kinase-inhibitor
- NCI Dictionary of Cancer Terms — BRAF V600 mutation
  https://www.cancer.gov/publications/dictionaries/cancer-terms/def/braf-v600-mutation
- NCI Dictionary of Cancer Terms — BRAF (V600E) kinase inhibitor RO5185426
  https://www.cancer.gov/publications/dictionaries/cancer-terms/def/braf-v600e-kinase-inhibitor-ro5185426

No BRAF gene citation is fabricated just because the query says BRAF. In this
run, that title appears on a boundary chunk retrieved for vemurafenib instead.

## G. Verification and artifacts

```powershell
.\onco-env\Scripts\python.exe -m unittest test_source_provenance test_index_options -v
.\onco-env\Scripts\python.exe debug_provenance.py
```

18 regression tests pass. Live run: `provenance_diagnostics/20261004_144556/`.
The diagnostic uses the original BGE model and default FAISS retriever over the
current full saved index. It fetches only the sources returned by the three test
queries and builds a metadata-only in-memory preview over the exact same vectors.
It does not embed documents again, save an index, or call Together AI.

| Query | Retrieved chunks | Distinct displayed entries | Text/order unchanged |
|---|---:|---:|---|
| BRAF | 4 | 3 | Yes |
| vemurafenib | 4 | 8 | Yes |
| melanoma | 4 | 5 | Yes |

Six actual source pages were needed: B, M, P, R, S and V. All 3,411 DOM entries
matched transformed text, with zero unmatched entries. All 4,245 regenerated
chunk texts matched the currently saved chunks in the same order. Metadata
survived a docstore serialization round trip before repeat retrieval.

`report.json` contains each retrieved chunk's complete text and metadata, before
and after citation records, unchanged retrieval checks and saved-file hashes.
`BRAF_sources.html`, `vemurafenib_sources.html` and `melanoma_sources.html` are
rendered with the actual result template. `page_00.html` through `page_05.html`
preserve the rendered input. `preview_docstore.pkl` is a partial-provenance
diagnostic docstore, not an application index; do not replace the production
index.pkl with it. Full metadata enrichment is installed on future ingestion.

The current saved index was preserved byte-for-byte:

- index.faiss: `4f9fa7233e5bd6c071fc8ceed72dc5c872da24beebbf237da2ac9e4a8aa849a0`
- index.pkl: `8c4578cb4e90593ac706a884c0a9ad770a88830f28d8840420219513187e0c3e`

Before edits, the working source was preserved in `provenance_diagnostics/` as
`mistral.before.py`, `result.before.html`, `nci_chromium_loader.before.py`, and
`baseline_tracked.diff` (which includes the user's earlier work).

## H. Remaining limitations and activation

- Existing saved indexes contain only their old metadata. They immediately benefit
  from page-link deduplication, but entry labels require ingestion with this code.
  Restart the app and use the existing Rebuild option to populate them; that option
  preserves the previous index in a timestamped backup. No rebuild was run here.
- Entry extraction is verified against the cancer-terms dictionary's current DOM.
  Other dictionary collections/markup, ACS pages, and unrecognized entry structures
  retain page-level citations. No drug-dictionary readiness changes were added.
- A chunk may contain an entry's trailing text and the next entry's heading. Both
  are represented. Vemurafenib's eight citations include adjacent entries such as
  vena cava; suppressing them based on presumed relevance would require a separate
  policy. This display records retrieved text provenance, not claim-level support.
- DOM or formatting changes can prevent exact matching. The fallback is the page
  source, never a guessed entry. Diagnostic unmatched counts expose this condition.
- The full index's text/vector content was not rebuilt or re-evaluated against the
  live website. The verified metadata preview covers the six retrieved source pages.
