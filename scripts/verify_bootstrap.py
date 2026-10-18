"""Infrastructure checks only; never extract or execute a source archive."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def safe_member(name: str) -> bool:
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and ".." not in p.parts and "\\" not in name and ":" not in name and "\x00" not in name

def verify_archive(path: Path, record: dict) -> dict:
    expected = next(x for x in record["inputs"] if x["filename"].endswith(".zip"))
    if sha256(path) != expected["sha256"]:
        raise ValueError("Source archive SHA256 mismatch")
    root = record["archive_root"] + "/"
    manifest = root + "SHA256SUMS.txt"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP names")
        if sum(i.file_size for i in archive.infolist()) > 1024**3:
            raise ValueError("Archive exceeds bootstrap inspection budget")
        for item in archive.infolist():
            if not safe_member(item.filename) or not item.filename.startswith(root):
                raise ValueError("Unsafe ZIP member")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("ZIP symbolic links are not admitted")
        rows = archive.read(manifest).decode("utf-8").splitlines()
        checked = set()
        for line in rows:
            digest, relative = line.split("  ", 1)
            if not re.fullmatch(r"[0-9a-f]{64}", digest) or not safe_member(relative):
                raise ValueError("Invalid manifest entry")
            name = root + relative
            if name in checked:
                raise ValueError("Duplicate manifest entry")
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Archive member checksum mismatch: " + relative)
            checked.add(name)
        files = {i.filename for i in archive.infolist() if not i.is_dir()}
        if checked | {manifest} != files or len(checked) != record["manifest_entries"]:
            raise ValueError("Incomplete source manifest")
    return {"archive_sha256": expected["sha256"], "manifest_entries_verified": len(checked), "extracted": False, "executed": False}

def check_repository(root: Path) -> dict:
    required = ["README.md", "AGENTS.md", ".gitignore", ".gitattributes", "docs/IMPORT_PLAN.md", "docs/RELEASE_CHECKLIST.md", "environment/reference.json", "provenance/baseline.json", "scripts/verify_bootstrap.py", "scripts/run_checks_no_exit.sh", "tests/test_bootstrap.py", "logbook/README.md"]
    for name in required:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or linked required file: " + name)
        path.read_text(encoding="utf-8")
    record = json.loads((root / "provenance/baseline.json").read_text())
    if record["repository"] != "Atalebe/bdtp" or record["visibility_at_bootstrap"] != "private":
        raise ValueError("Incorrect repository contract")
    for key in ("manuscript_changes_authorized", "numerical_import_authorized", "new_scientific_runs_authorized"):
        if record[key] is not False:
            raise ValueError("Bootstrap scope widened")
    if record["licence_status"] != "UNDECIDED" or record["archival_doi"] is not None:
        raise ValueError("Unapproved release metadata")
    for item in record["inputs"]:
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Invalid source identity")
    for forbidden in ("reviewer", "reviews", "correspondence", "manuscript"):
        if (root / forbidden).exists():
            raise ValueError("Content outside bootstrap scope: " + forbidden)
    env = json.loads((root / "environment/reference.json").read_text())
    if env["numpy"] != "2.3.5" or env["automatic_dependency_installation"] is not False:
        raise ValueError("Historical environment contract changed")
    return {"status": "PASS", "scope": "repository infrastructure only", "required_files": len(required), "new_scientific_runs": 0}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    report = check_repository(ROOT)
    if args.archive:
        report["source"] = verify_archive(args.archive, json.loads((ROOT / "provenance/baseline.json").read_text()))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
