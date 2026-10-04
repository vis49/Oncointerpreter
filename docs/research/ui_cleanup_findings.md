# Oncointerpreter UI cleanup

## Repository audit

- Entry point: `page.py`, a Flask app. It is not Streamlit, despite historical
  Streamlit imports in the pipeline modules. The UI is Jinja HTML in `templates/`.
- Active RAG/generation implementation: formerly `mistral.py`, now `rag_pipeline.py`.
  Its ChatTogether configuration is `openai/gpt-oss-120b`, max_tokens 2048,
  temperature 0.1. Those settings were inspected in code, not inferred from filenames.
- The old model selector actually selected two code paths. Its "Mistral" label
  invoked the current GPT-OSS pipeline. "LLaMA" invoked the separate historical
  pipeline configured for `togethercomputer/llama-2-7b-chat`. The latter was removed
  from the active interface as requested, not silently represented as the same model.
- The query textarea and index reuse/rebuild radio buttons were functional.
  Knowledge-base selection and token count were outside the submitted form and
  unused by the route. They were misleading controls, not working settings.
- PDF/image uploads perform real OCR and write `report.txt`. Only the historical
  alternative pipeline reads that file. The active pipeline does not incorporate
  uploaded documents. Merely displaying the uploaded filename on results also
  implied use that did not occur.
- `page.py` was the only application importer of the two pipeline modules.
  Diagnostic tests referenced `mistral.py` by filename for AST-based function checks.
  Prior findings and patch snapshots contained historical references. `gpt.py` was
  a standalone, unreferenced GPT-Neo experiment.
- Launching uses `python page.py` / the project virtualenv interpreter. Docker
  declares `FLASK_APP=page.py`; the entry point was kept, so these commands remain
  valid. The Docker image itself was not built or modernized.

## Changes

1. Added a shared responsive Jinja shell, restrained teal/neutral CSS, system fonts,
   visible keyboard focus, a skip link, clear question/results sections, consistent
   actions and accessible status messages. No frontend framework, font CDN or new
   dependency was introduced.
2. The main form now offers typed/pasted question or case text, one Analyze action,
   and an expandable Source index options section. Source labels/links returned
   by the existing formatter are rendered in a subordinate evidence card.
3. Removed the legacy model selector, unused knowledge-base and token controls,
   upload buttons from the active workflow, repeated route decorators, and the
   misleading uploaded-filename result label. The old LLaMA branch/import was
   removed from the Flask query route.
4. Preserved failed-submit question text and index choice for missing-key errors.
   Added a busy state to discourage duplicate submissions; it resets on browser
   back-navigation. There is no invented percentage progress indicator.
5. Historical upload routes remain with the same extraction/save behavior, shared
   styling and clear scope. Their browser handlers now show extraction failure or
   completion instead of leaving a spinner or silently implying a document was
   analyzed. OCR output remains the same local `report.txt` file.

## Renames and retained code

- `mistral.py` → `rag_pipeline.py`: same bytes, more accurate name. The active
  import in `page.py`, AST test references and diagnostic description were updated.
- `llama2.py` → `legacy/llama2.py`: retained unchanged, no active app importer.
- `gpt.py` → `legacy/gpt.py`: retained unchanged, no active app importer.
- `page.py` remains the entry point; no command or Docker entry-point change needed.
- README now documents the actual pipeline, supported input, launch command and
  retained historical code. `legacy/README.md` explains the moved files.
- Prior diagnostic reports, baseline snapshots and the proposed scrape patch keep
  their original filenames/line references. The README explicitly identifies them
  as historical records. The old model comments and prompt conventions inside the
  renamed pipeline are preserved with the backend; none is displayed in the UI.
- Other unrelated helpers/imports in `page.py` and historical research files were
  left in place rather than widening this task into dependency/dead-code cleanup.

These are filesystem moves; Git can track their similarity when the changes are
staged/committed. No commits or history rewrites were performed.

## Backend boundary

The active pipeline module is byte-identical to its pre-rename snapshot. HTML
scraping, source extraction, provenance, embeddings, chunking, FAISS, prompt,
LLM configuration, generation and response formatting were not edited.

The route now always calls that active pipeline instead of accepting a legacy
model branch. The index flag and call order are retained. Error rendering passes
the existing question and index choice back to the form. Both saved index files
were hashed before/after verification and remained unchanged.

## Verification

- Ran all 18 source-provenance and index/key regression tests successfully.
- Imported the real Flask app and its renamed pipeline, started it on an ephemeral
  localhost port, and checked home, both retained utility pages and static assets.
- Submitted existing/rebuild form choices through the real route. Only the costly
  LLM initialization and index loading were replaced with local fixtures. The
  actual RAG chain, response formatter, citation rendering and Flask/Jinja route
  executed. No Together API calls, OCR processing or saved-index rebuild was run.
- Playwright verified desktop (1280 px) and mobile (390 px) question → Analyze →
  results → new question; source title/link rendering; no horizontal overflow;
  and missing-key errors retaining input and selection. Zero JavaScript errors.
- Inspected screenshots for desktop/mobile layout. Evidence is in `ui_diagnostics/`:
  `desktop_input.png`, `desktop_results.png`, `mobile_input.png`,
  `mobile_results.png`, `mobile_error.png`, `verification.json` and test logs.
- Confirmed there are no active imports of the old modules, no obsolete model
  names in `templates/`, `static/` or `page.py`, and no whitespace errors in Git diff.

`verify_ui.py` reproduces browser verification using local fixtures. Copies of the
pre-change entry point, active pipeline and four templates are in
`ui_diagnostics/baseline/`. The full current model/API response was not retested
against Together because the generation implementation did not change.

## Intentionally deferred

Document-to-analysis integration, background rebuild jobs/progress streaming,
chat history, configurable model selection, claim-level citations, general error
handling, dependency modernization and Docker fixes would require changes beyond
this UI pass. The interface does not advertise these as available.

Restart the server using the unchanged command:

```powershell
.\onco-env\Scripts\python.exe page.py
```
