"""Index-choice regression checks, without scraping, embedding, or touching the corpus."""
import ast
from datetime import datetime
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from flask import Flask, request, session, render_template

ROOT = Path(__file__).resolve().parent


def isolated_function(filename, name, namespace):
    # Compile the actual function without executing model/other import side effects.
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


class IndexOptionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous_cwd = os.getcwd()
        os.chdir(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(os.chdir, self.previous_cwd)
        Path("faiss_index").mkdir()
        Path("faiss_index/index.faiss").write_bytes(b"old vectors")
        Path("faiss_index/index.pkl").write_bytes(b"old docstore")
        self.db = Mock()
        self.db.save_local.side_effect = self.save_index
        self.faiss = Mock()
        self.faiss.from_documents.return_value = self.db
        self.loader = Mock()
        self.namespace = {
            "HuggingFaceBgeEmbeddings": Mock(),
            "torch": SimpleNamespace(device=lambda value: value, cuda=SimpleNamespace(is_available=lambda: False)),
            "os": os, "tempfile": tempfile, "time": time, "datetime": datetime,
            "FAISS": self.faiss, "log": Mock(), "AsyncChromiumLoader": self.loader,
            "Html2TextTransformer": Mock(), "RecursiveCharacterTextSplitter": Mock(),
            "prepare_provenance": Mock(), "annotate_chunks": Mock(),
        }
        self.load_data = isolated_function("rag_pipeline.py", "load_data", self.namespace)

    @staticmethod
    def save_index(path):
        Path(path, "index.faiss").write_bytes(b"new vectors")
        Path(path, "index.pkl").write_bytes(b"new docstore")

    def assert_old_index(self):
        self.assertEqual(Path("faiss_index/index.faiss").read_bytes(), b"old vectors")
        self.assertEqual(Path("faiss_index/index.pkl").read_bytes(), b"old docstore")

    def test_default_loads_existing_without_scraping(self):
        self.assertIs(self.load_data(), self.faiss.load_local.return_value)
        self.loader.assert_not_called()
        self.faiss.from_documents.assert_not_called()
        self.assert_old_index()

    def test_rebuild_bypasses_cache_and_preserves_old_index(self):
        self.assertIs(self.load_data(force_rebuild=True), self.db)
        self.faiss.load_local.assert_not_called()
        self.loader.assert_called_once()
        self.assertEqual(Path("faiss_index/index.faiss").read_bytes(), b"new vectors")
        backups = list(Path(".").glob("faiss_index_backup_*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "index.faiss").read_bytes(), b"old vectors")
        self.assertEqual((backups[0] / "index.pkl").read_bytes(), b"old docstore")

    def test_save_failure_leaves_existing_index(self):
        self.db.save_local.side_effect = RuntimeError("save failed")
        with self.assertRaisesRegex(RuntimeError, "save failed"):
            self.load_data(force_rebuild=True)
        self.assert_old_index()
        self.assertFalse(list(Path(".").glob("faiss_index_backup_*")))

    def test_scrape_failure_leaves_existing_index(self):
        self.loader.return_value.load.side_effect = RuntimeError("scrape failed")
        with self.assertRaisesRegex(RuntimeError, "scrape failed"):
            self.load_data(force_rebuild=True)
        self.assert_old_index()

    def test_replacement_failure_restores_existing_index(self):
        original_rename = os.rename
        def rename(source, destination):
            if str(source).startswith("./faiss_index_build_") or Path(source).name.startswith("faiss_index_build_"):
                raise OSError("replacement failed")
            return original_rename(source, destination)
        with patch.object(os, "rename", side_effect=rename):
            with self.assertRaisesRegex(OSError, "replacement failed"):
                self.load_data(force_rebuild=True)
        self.assert_old_index()

    def test_missing_index_builds_by_default(self):
        os.rename("faiss_index", "baseline")
        self.assertIs(self.load_data(), self.db)
        self.loader.assert_called_once()
        self.assertEqual(Path("faiss_index/index.faiss").read_bytes(), b"new vectors")

    def test_form_routes_index_choice_to_active_pipeline(self):
        app = Flask(__name__)
        app.secret_key = "test-only"
        loader = Mock()
        response = {"answer": "test", "sources": []}
        namespace = {
            "request": request, "session": session, "datetime": datetime,
            "load_tokenizer_and_llm": Mock(), "load_data": loader,
            "process_query": Mock(return_value=response),
            "TogetherKeyConfigurationError": ValueError,
            "render_template": lambda template, **values: {"template": template},
        }
        handler = isolated_function("page.py", "index", namespace)
        app.add_url_rule("/", view_func=handler, methods=["GET", "POST"])
        client = app.test_client()
        for mode, expected in (("existing", False), ("rebuild", True), ("invalid", False), (None, False)):
            form = {"query": "test"}
            if mode is not None:
                form["index_mode"] = mode
            self.assertEqual(client.post("/", data=form).status_code, 200)
            loader.assert_called_with(force_rebuild=expected)
        loader.reset_mock()
        namespace["load_tokenizer_and_llm"].side_effect = ValueError("Together AI key is missing")
        self.assertEqual(client.post("/", data={"query": "test"}).status_code, 400)
        loader.assert_not_called()

    def test_missing_key_fails_before_client_initialization(self):
        constructor = Mock()
        namespace = {"os": os, "ChatTogether": constructor, "TogetherKeyConfigurationError": ValueError}
        load_llm = isolated_function("rag_pipeline.py", "load_tokenizer_and_llm", namespace)
        with patch.dict(os.environ, {"TOGETHER_API_KEY": "   "}):
            with self.assertRaisesRegex(ValueError, "Together AI key is missing"):
                load_llm()
        constructor.assert_not_called()

    def test_configured_key_passed_to_client(self):
        constructor = Mock()
        namespace = {"os": os, "ChatTogether": constructor, "TogetherKeyConfigurationError": ValueError}
        load_llm = isolated_function("rag_pipeline.py", "load_tokenizer_and_llm", namespace)
        with patch.dict(os.environ, {"TOGETHER_API_KEY": " test-only-value "}):
            self.assertIs(load_llm(), constructor.return_value)
        self.assertEqual(constructor.call_args.kwargs["together_api_key"], "test-only-value")

    def test_template_contains_index_choices_without_legacy_controls(self):
        app = Flask(__name__, template_folder=str(ROOT / "templates"))
        app.add_url_rule('/', endpoint='index', view_func=lambda: '')
        with app.test_request_context():
            html = render_template("index.html")
        self.assertIn('name="index_mode" value="existing" checked', html)
        self.assertIn('name="index_mode" value="rebuild"', html)
        self.assertIn('id="index-options"', html)
        self.assertNotIn('name="model"', html)
        self.assertNotIn('name="knowledgeBase"', html)


if __name__ == "__main__":
    unittest.main()
