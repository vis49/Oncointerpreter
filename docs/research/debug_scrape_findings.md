# Single-page NCI ingestion diagnosis

Implementation update: the targeted loader fix is now applied at the user's request.
`mistral.py:18` imports the local `NciReadyChromiumLoader`; the readiness check is
in `nci_chromium_loader.py:7-38`. The diagnostic now imports that actual helper
instead of keeping a duplicate candidate implementation. The original diagnosis
and timings below remain the reproduction baseline.

Post-implementation verification: `debug_scrape_runs/20261004_141206_504434/`
captured all three BRAF strings, produced 177,912 transformed characters and 547
chunks, and confirmed the saved index hashes were unchanged. Raw HTML contained
435,383 characters; incidental page scripts can change its size between captures.
No index rebuild was performed. Embeddings, splitting, retrieval, prompt and LLM
remain unchanged. `debug_scrape_implemented.log` records this verification.

Diagnosed October 4, 2026. Only `/publications/dictionaries/cancer-terms/expand/B`
was navigated; its normal subresources/API requests were observed. No production
code, dependencies, embedding settings, retrieval settings, prompt, or LLM changed.
The existing FAISS files were read, never saved or overwritten.

## A. Root cause

**Premature HTML capture by AsyncChromiumLoader, before the glossary API response
and React rendering.** The dictionary entries are already absent in the raw loaded
Document. Html2TextTransformer, splitting, and FAISS do not remove them.

The installed loader awaits `page.goto(url)` (the default load event), immediately
calls `page.content()`, and closes Chromium. The page's load event does not mean
that this asynchronously fetched dictionary has rendered.

There is a second persistence mechanism: `load_data()` returns the saved index
before reaching the scraper whenever `faiss_index/` exists. Correcting the scraper
alone cannot repair that saved corpus.

## B. Evidence

Authoritative local run: `debug_scrape_runs/20261004_140316_362078/`.
The earlier run independently reproduced the same missing-content behavior.

| Stage | Original loader | Content-ready candidate | Exact BRAF count, original / ready |
|---|---:|---:|---:|
| Raw HTML characters | 32,776 | 434,265 | 0 / 21 |
| Transformed text characters | 1,506 | 177,912 | 0 / 21 |
| Chunks passed to FAISS | 4 | 547 | 0 / 23 |
| Sum of chunk characters | 1,515 | 179,120 | 0 / 23 |
| Actual in-memory FAISS docstore documents | 4 | 547 | 0 / 23 |

The navigation response itself contains 29,545 characters and no BRAF. The loaded
HTML is a partially hydrated application, with search/navigation but no entries.
After readiness, both raw HTML and transformed text contain exact `BRAF gene`
and `BRAF V600` once each. Chunks/docstore contain them twice and once,
respectively. Overlap explains increased chunk occurrence counts; these counts
are not numbers of distinct dictionary entries. The literal `BRAF gene` test also
does not match Markdown `_BRAF_ gene`, so broader BRAF counts are reported too.

Observed request:

`https://webapis.cancer.gov/glossary/v1/Terms/expand/Cancer.gov/Patient/en/B?size=10000`

Its JSON has `meta.totalResults = 422` and 422 `results`. The rendered page has
422 `dt dfn` terms, 422 `dd.definition` definitions, and an `h4` reporting 422
results. BRAF gene and BRAF V600 mutation appear among its term links.

Recorded seconds from the instrumented run's start:

| Event | Seconds |
|---|---:|
| Glossary XHR requested | 0.6023 |
| Browser load event | 0.6377 |
| `goto()` returned (HTTP 200) | 0.6381 |
| Immediate HTML captured; BRAF absent | 0.6433 |
| Glossary XHR response received (HTTP 200) | 1.1629 |
| BRAF gene rendered | 1.3850 |
| Full term/definition count predicate satisfied | 1.4206 |

These times are measurements, not proposed sleep durations. API response receipt
alone is also insufficient: the DOM must finish rendering.

The saved docstore has 481 documents overall, **four for this exact cancer-terms
B URL**, not eight. Those four match the reproduced baseline chunks verbatim.
One saved document contains BRAF, from the Zelboraf entry on cancer-terms/expand/Z
(three literal BRAF occurrences within that one document). The earlier reported
eight chunks may have used a broader source filter; that was not assumed.

The in-memory check used the original `BAAI/bge-large-en-v1.5`, original device
selection and normalized embeddings, original 500/20 splitter, and real FAISS.
Assertions confirm every input chunk's text AND metadata equals its indexed
docstore counterpart in order. No replacement/fake embeddings were used.

Evidence files include `report.json`, `browser_events.json`, `dictionary_dom.json`,
`response_000.txt`, `navigation_response.html`, both browser HTML snapshots,
`versions.json`, installed loader source, and per-stage full documents/chunks
with character counts, exact/case-insensitive searches, source metadata, and
beginning/middle/end excerpts in each `summary.json`.

Dependency/code trace:

- `mistral.py:142`: original BGE configuration; `:158-178`: existing-index return.
- `mistral.py:199-203`: loader, load, then HTML transformation.
- `mistral.py:205-212`: original recursive splitter, 500/20.
- `mistral.py:216-219`: chunks enter `FAISS.from_documents`; `:227`: persistence.
- Installed `langchain_community/document_loaders/chromium.py:58-59`: goto then content;
  `:78-80`: attach source metadata and yield Document.
- Installed `langchain_community/document_transformers/html2text.py:39-46`:
  ignore links/images and call `h.handle`, preserving metadata.
- Installed `langchain_core/vectorstores.py:548-550`: extract texts/metadata.
- Installed `langchain_community/vectorstores/faiss.py:930`: embed those texts;
  `:185-203`: construct Documents, add vectors and unchanged Documents to docstore.

Versions: langchain 0.1.16, langchain-community 0.0.32, Playwright 1.63.0,
html2text 2025.4.15, sentence-transformers 3.0.1, faiss-cpu 1.15.1.

## C. Minimal fix (now applied; originally proposed after diagnosis)

`proposed_nci_wait.patch` preserves the originally reviewed small local loader
subclass and one-line import replacement in `mistral.py`. It passed
`git apply --check` before implementation. The implementation is now in
`nci_chromium_loader.py:7-38` and `mistral.py:18`.

The behavioral change is a single readiness operation between `goto()` and
`content()` for NCI cancer-terms expand URLs:

```python
await page.goto(url)
await page.wait_for_function(DICTIONARY_READY, timeout=45000)
return await page.content()
```

The predicate requires the positive displayed result count to equal BOTH rendered
term and definition counts, with every definition nonempty. This is stronger than
waiting for a search box, a single entry, or BRAF specifically. It follows observed
DOM structure, preserves the full-page HTML and downstream pipeline, and delegates
all other URLs to the installed loader. No package upgrade or site-packages edit
is needed; this installed loader has no constructor argument for custom waits.

45 seconds is a bounded failure timeout, not a delay: capture proceeds immediately
when the predicate succeeds. A timeout/navigation error propagates for the targeted
pages instead of indexing partial HTML/error text. The browser closes in `finally`.

**Classification:** diagnostic script/artifacts plus an applied ingestion timing
fix. No dependency compatibility fix was needed. No retrieval/embedding experiment
or corpus redesign was made. The initial BRAF sentinel was diagnostic only; the
candidate fix uses completeness, not a query-specific sentinel.

## D. Verification procedure

From the repository, run the existing interpreter:

```powershell
.\onco-env\Scripts\python.exe debug_scrape.py
.\onco-env\Scripts\python.exe debug_scrape.py --build-faiss
```

The first command diagnoses stages and the candidate wait without embeddings.
The optional second command additionally builds original-loader and candidate
indexes **in memory**, verifies exact docstore preservation, and never saves them.
Each invocation creates a new timestamped artifact directory. The script does
not import `mistral.py`, invoke the LLM, or call `db.save_local()`.

Check API total against rendered term/definition counts; all three BRAF strings
in raw, transformed and chunk stages; matching chunk and FAISS document counts;
and `saved_index_unchanged: true`. Expect site content/counts to evolve. The stock
loader may occasionally win the race; intermittent success does not disprove it.

**The saved index must be rebuilt after the fix to use the missing content.**
Preserve the current index as the reproduction baseline, validate the other source
families, then build a replacement separately and switch only after validation.
Do not merely rerun `load_data()` with the current directory still present: its
early return will keep loading the defective index. No rebuild was performed here.

SHA-256 before and after this diagnostic matched:

- `index.faiss`: `0d45109231f19c4337bb558b7d33d96b4e494b72122fb65b87d9954f40994490`
- `index.pkl`: `c6db3c4367de237181565d2e563eeab21020cdc96bff3560771c167e26ec3b04`

## E. Remaining uncertainty

Only cancer-terms B was live-tested. Other letters, cancer-drug, ACS and other NCI
pages need separate readiness checks before a full rebuild. The candidate deliberately
does not extend an unverified selector to cancer-drug. Valid zero-result pages,
changed markup or pagination would require an explicitly verified policy; the
current positive-count predicate fails closed in those cases. Future network/API
failures remain possible. The original cache has no historical raw scrape, but its
B chunks exactly match the reproduced premature-capture output. Retrieval quality
in the complete repaired corpus was not evaluated; this test establishes ingestion
integrity, not ranking quality or clinical validity.

## F. Exact files/lines changed

- Changed `mistral.py:18`: import the targeted local loader under the existing
  `AsyncChromiumLoader` name. This is the only change to the existing application.
- Added `nci_chromium_loader.py:1-38`: result-count readiness predicate and narrowly
  scoped browser loader override. Timeouts propagate before indexing partial content.
- Added `debug_scrape.py:1-212`: standalone diagnostic; imports the production helper
  at `:24`, restricted read-only cached-docstore inspection `:30-41` and `:167-176`,
  stage diagnostics `:44-86`, browser/network observations `:89-150`, and
  optional unchanged BGE/FAISS verification `:185-199`.
- Added `proposed_nci_wait.patch:1-51`: historical review artifact for the applied fix.
- Added this findings document, `debug_scrape_latest.log`,
  `debug_scrape_implemented.log`, and three timestamped `debug_scrape_runs/`
  artifact directories. These are temporary diagnostic outputs.
