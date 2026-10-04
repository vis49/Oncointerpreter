# Baseline V1

Recorded on 2026-10-04 at the researcher's request. This is the version to benchmark and test before making further improvements.

- Git tag: `baseline-v1`
- Application commit: `b65ba6a9b6efac21c1e800caa8bbc026783c13b1`
- Repository: https://github.com/vis49/Oncointerpreter

The tag identifies the existing application commit. The documentation recording this decision is a subsequent documentation-only commit.

## Configuration at designation

- Entry point: `page.py`; active pipeline: `rag_pipeline.py`.
- Provider/model: Together AI, `openai/gpt-oss-120b`.
- Generation settings: temperature `0.1`, maximum output tokens `2048`.
- Embeddings: `BAAI/bge-large-en-v1.5`, normalized embeddings.
- Splitter: `RecursiveCharacterTextSplitter`, chunk size `500`, overlap `20`.
- Retrieval: the existing FAISS `as_retriever()` defaults, with no search overrides.
- Prompt: the exact prompt stored in the tagged pipeline.
- Includes the current NCI ingestion readiness fix, source provenance/display, index controls, and UI.

V1 names this application and RAG configuration; it does not version the externally hosted model weights.

## Benchmark policy

Benchmark this version before implementing further model, embedding, retrieval, chunking, prompt, ingestion, or interface improvements. Keep experimental changes in separate commits/branches and compare them against `baseline-v1`. Do not move or replace this tag.

Record the benchmark questions, expected answers, evaluation method, runtime/dependency versions, model identifier, generation settings, and corpus/index identity alongside results. Separate end-to-end model benchmarking from the existing fixture-based regression and browser tests. No model benchmark has been run as part of this designation.

## Existing local index snapshot

The saved index is excluded from Git. These fingerprints record the local files at designation; they do not certify corpus quality or freshness. Preserve a private copy of these files and the working Python environment before rebuilding the index or updating dependencies. Live source pages and hosted model behavior may change over time.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `faiss_index/index.faiss` | 49733677 | `2f0978d58b5ca098ba1e364ff27c3ada379fa9dfc51bd84c6007dc59b02fc112` |
| `faiss_index/index.pkl` | 7948716 | `b25ad17774d9bbcf57641e2365f92a39238696ef217ba088e0d85b40f557cc57` |

For benchmarks against this snapshot, select **Use existing index**. A rebuild creates a different corpus snapshot and must be identified separately in the benchmark results.

## Validation already completed

At designation, 18 local regression tests and the desktop/mobile browser checks passed in an isolated test environment. Those checks use local fixtures and do not establish answer accuracy or clinical performance.
