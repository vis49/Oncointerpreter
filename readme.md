# Oncointerpreter.ai

A research interface for asking oncology questions and reviewing generated interpretations alongside retrieved source references.

The current application is designated **Baseline V1**, tagged `baseline-v1` at commit `b65ba6a`. Benchmark and test this version before further improvements. See [the baseline record](docs/research/baseline_v1.md) for configuration, index fingerprints, and comparison policy.

## Run locally

From the repository root:

```powershell
.\onco-env\Scripts\python.exe page.py
```

Open http://127.0.0.1:5000. The Flask entry point remains `page.py`; `flask --app page run` and the Dockerfile's `FLASK_APP=page.py` still point to it.

Save your Together AI key once in `%LOCALAPPDATA%\Oncointerpreter\.env`:

```dotenv
TOGETHER_API_KEY=your_actual_key
```

Restart the server after changing it. This plain-text file is outside the OneDrive project; keep it private. A Git-ignored project `.env` is also supported. Existing environment variables take priority, followed by the local AppData file, then the project file. `.env.example` contains only a blank template.

## Current workflow

Type a question or paste relevant case/report text into the question field, then select **Analyze**. **Source index options** offers existing-index reuse or a rebuild with a backup. Results show the generated interpretation and the existing retrieved-source information.

The current pipeline uses Together AI with `openai/gpt-oss-120b` (temperature 0.1, max_tokens 2048), as configured in `rag_pipeline.py`. The interface identifies the provider without duplicating a model selector. Embeddings use BAAI/bge-large-en-v1.5 and select CUDA when available, otherwise CPU.

The original PDF/image routes perform OCR into `report.txt`, but the active pipeline does not read that file. They remain accessible as historical utilities at `/upload_pdf` and `/upload_image` and are not advertised as analysis inputs. Paste relevant text into the question field instead. OCR still requires Tesseract and PDF conversion requires Poppler.

## Project files

- `page.py`: Flask application, query route and retained extraction routes.
- `rag_pipeline.py`: active retrieval and Together AI generation; renamed from `mistral.py` with its contents preserved.
- `templates/`, `static/`: Jinja templates, shared responsive CSS and small interaction scripts.
- `nci_chromium_loader.py`: existing NCI readiness behavior.
- `source_provenance.py`: existing entry metadata and citation display preparation.
- `legacy/llama2.py`, `legacy/gpt.py`: historical implementations retained unchanged; not imported by the app.
- `debug_scrape.py`, `debug_provenance.py`: reproduction diagnostics.
- `docs/research/ui_cleanup_findings.md`: UI audit, changes and verification notes.

Install the existing dependencies with `pip install -r requirements.txt`. No frontend framework or new dependency was added by the UI cleanup. The Dockerfile is retained as historical deployment configuration; its older base image/system setup was not modernized or validated here.

## Verification

```powershell
.\onco-env\Scripts\python.exe -m unittest test_source_provenance test_index_options -v
.\onco-env\Scripts\python.exe verify_ui.py
```

The UI verification boots the real Flask application with local LLM/retriever fixtures and captures desktop/mobile screenshots. It makes no Together API calls and does not rebuild the saved index. It also runs without a saved index or local baseline snapshots; historical byte comparisons are performed only when a snapshot exists.

GitHub Actions runs these Python tests and Chromium UI checks on pushes and pull requests to `main`/`master`, or manually through Actions. The workflow uses Python 3.11, a CPU-only test environment from `requirements-ci.txt`, and no API secrets or saved index. Test logs and screenshots are uploaded as `oncointerpreter-test-results`. The historical Node/Playwright configuration is no longer used by CI. This test environment does not replace the research runtime in `requirements.txt`.

## Research provenance

Historical diagnostic reports and patch snapshots are in `docs/research/` and keep their original filenames and line references. Generated run artifacts, uploads, local environments, logs, and FAISS indexes/backups are Git-ignored and remain local. References there to `mistral.py` refer to the module now called `rag_pipeline.py`. Original Mistral/LLaMA experiments are historical, not selectable models in the current UI. Backend prompt conventions and commented research code remain unchanged.

![Original architecture](figure.png "Original project architecture")

## Non-Commercial Research Only License

©2023-2024 Rutgers, The State University of New Jersey, All rights reserved. Do not copy or reproduce without permission.
