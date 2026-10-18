"""Tests for repository preparation, not numerical method validation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bootstrap", ROOT / "scripts/verify_bootstrap.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class BootstrapTests(unittest.TestCase):
    def test_repository_contract(self):
        self.assertEqual(module.check_repository(ROOT)["status"], "PASS")

    def test_safe_names(self):
        self.assertTrue(module.safe_member("root/src/module.py"))
        for name in ("../x", "/tmp/x", "x/../y", "x\\y", "C:/x", "", "x\x00y"):
            with self.subTest(name=name):
                self.assertFalse(module.safe_member(name))

    def make_archive(self, directory, contents=b"reference", expected=None, extra=False):
        path = Path(directory) / "source.zip"
        expected = contents if expected is None else expected
        digest = hashlib.sha256(expected).hexdigest()
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("baseline/code.py", contents)
            archive.writestr("baseline/SHA256SUMS.txt", digest + "  code.py\n")
            if extra:
                archive.writestr("baseline/unlisted.txt", "not admitted")
        record = {"inputs": [{"filename": "source.zip", "sha256": module.sha256(path)}], "archive_root": "baseline", "manifest_entries": 1}
        return path, record

    def test_manifest_pass_without_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            path, record = self.make_archive(directory)
            result = module.verify_archive(path, record)
            self.assertEqual(result["manifest_entries_verified"], 1)
            self.assertFalse(result["extracted"])
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["source.zip"])

    def test_wrong_outer_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path, record = self.make_archive(directory)
            record["inputs"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                module.verify_archive(path, record)

    def test_wrong_member_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path, record = self.make_archive(directory, contents=b"altered", expected=b"original")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                module.verify_archive(path, record)

    def test_unlisted_member(self):
        with tempfile.TemporaryDirectory() as directory:
            path, record = self.make_archive(directory, extra=True)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                module.verify_archive(path, record)

    def test_numerical_work_not_authorized(self):
        record = json.loads((ROOT / "provenance/baseline.json").read_text())
        self.assertFalse(record["new_scientific_runs_authorized"])
        self.assertFalse(record["numerical_import_authorized"])
        self.assertFalse(record["manuscript_changes_authorized"])

    def test_references_not_installed_environment(self):
        record = json.loads((ROOT / "environment/reference.json").read_text())
        self.assertFalse(record["automatic_dependency_installation"])
        self.assertEqual(record["numpy"], "2.3.5")

if __name__ == "__main__":
    unittest.main()
