"""Recovery tests use disposable repositories, never the user's working files."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("project_setup", Path(__file__).resolve().parents[1] / "setup.py")
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


class RepositoryRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Setup Test")
        self.write("app.py", "print('original')\n")
        self.write("analysis.ipynb", json.dumps({"nbformat": 4, "cells": []}))
        self.write("model.joblib", "original binary model fixture")
        self.write("model.csv", "price_cold,area_sqm,rooms\n2000,80,3\n")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)

    def write(self, name, text):
        (self.root / name).write_text(text, encoding="utf-8")

    def test_valid_local_edits_are_preserved(self):
        self.write("app.py", "print('my work')\n")
        setup.inspect_repository(self.root)
        self.assertIn("my work", (self.root / "app.py").read_text())

    def test_corrupt_notebook_is_backed_up_before_repair(self):
        self.write("analysis.ipynb", "{unfinished")
        with self.assertRaises(setup.SetupError):
            setup.inspect_repository(self.root)
        setup.inspect_repository(self.root, repair=True)
        self.assertEqual(json.loads((self.root / "analysis.ipynb").read_text())["cells"], [])
        backups = list((self.root / ".setup-backups").glob("files-*/analysis.ipynb"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), "{unfinished")

    def test_missing_file_is_restored(self):
        (self.root / "app.py").unlink()
        setup.inspect_repository(self.root, repair=True)
        self.assertEqual((self.root / "app.py").read_text(), "print('original')\n")

    def test_changed_binary_is_detected_without_deserialization(self):
        self.write("model.joblib", "different nonempty bytes")
        with self.assertRaisesRegex(setup.SetupError, "changed/damaged"):
            setup.inspect_repository(self.root)
        setup.inspect_repository(self.root, allow_local_data=True)
        setup.inspect_repository(self.root, repair=True)
        self.assertEqual((self.root / "model.joblib").read_text(), "original binary model fixture")

    def test_staged_changes_are_never_repaired(self):
        self.write("analysis.ipynb", "{my staged work")
        self.git("add", "analysis.ipynb")
        with self.assertRaisesRegex(setup.SetupError, "staged change"):
            setup.inspect_repository(self.root, repair=True)
        self.assertEqual((self.root / "analysis.ipynb").read_text(), "{my staged work")

    def test_lfs_pointer_and_invalid_notebook_shape(self):
        self.write("analysis.ipynb", "version https://git-lfs.github.com/spec/v1\n")
        self.assertIn("Git LFS pointer", setup.inspect_file(self.root / "analysis.ipynb"))
        self.write("analysis.ipynb", "[]")
        self.assertIn("invalid", setup.inspect_file(self.root / "analysis.ipynb"))

    def test_unmerged_index_stops_setup(self):
        original_branch = self.git("branch", "--show-current").stdout.decode().strip()
        self.git("checkout", "-qb", "other")
        self.write("app.py", "print('other')\n")
        self.git("commit", "-qam", "other")
        self.git("checkout", "-q", original_branch)
        self.write("app.py", "print('main')\n")
        self.git("commit", "-qam", "main")
        result = subprocess.run(["git", "merge", "other"], cwd=self.root, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        with self.assertRaisesRegex(setup.SetupError, "merge conflicts"):
            setup.inspect_repository(self.root, repair=True)


class EnvironmentSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_existing_compatible_environment_is_reused(self):
        env = self.root / ".venv"
        env.mkdir()
        (env / "sentinel").write_text("keep")
        with patch.object(setup, "environment_valid", return_value=True), patch.object(setup.venv, "EnvBuilder") as builder:
            self.assertEqual(setup.prepare_environment(self.root), setup.python_in(env))
            builder.assert_not_called()
        self.assertEqual((env / "sentinel").read_text(), "keep")

    def test_incompatible_environment_is_renamed_not_deleted(self):
        env = self.root / ".venv"
        env.mkdir()
        (env / "sentinel").write_text("keep")
        with patch.object(setup, "environment_valid", return_value=False), patch.object(setup.venv, "EnvBuilder") as builder:
            builder.return_value.create.side_effect = lambda path: path.mkdir()
            setup.prepare_environment(self.root)
        backups = list(self.root.glob(".venv.backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "sentinel").read_text(), "keep")
        self.assertTrue(env.is_dir())

    def test_path_escape_refused(self):
        with self.assertRaises(setup.SetupError):
            setup.safe_path(self.root, "../outside")

    def test_bad_bootstrap_checksum_never_executes_download(self):
        with patch.object(setup, "run") as run, patch.object(setup.urllib.request, "urlopen") as download:
            run.return_value.returncode = 1
            download.return_value.__enter__.return_value.read.return_value = b"corrupt wheel"
            with self.assertRaisesRegex(setup.SetupError, "checksum"):
                setup.pip_command(Path("python"), self.root)
            self.assertEqual(run.call_count, 1)
        self.assertFalse(list(self.root.rglob("*.whl")))


class CheckoutPortabilityTests(unittest.TestCase):
    def test_tracked_paths_work_on_windows(self):
        names = setup.git("ls-files", "-z").stdout.decode().split("\0")
        reserved = {"con", "prn", "aux", "nul"} | {f"{prefix}{i}" for prefix in ("com", "lpt") for i in range(1, 10)}
        for name in filter(None, names):
            for part in Path(name).parts:
                self.assertEqual(part, part.rstrip(" ."), name)
                self.assertFalse(set(part) & set('<>:"|?*'), name)
                self.assertNotIn(part.split(".")[0].lower(), reserved, name)


if __name__ == "__main__":
    unittest.main()
