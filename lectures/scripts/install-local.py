"""Copy the finished package into a new directory and verify every inventoried file."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

source = Path(__file__).resolve().parents[1]
destination = Path(sys.argv[1]).expanduser()
if not destination.is_absolute():
    raise SystemExit("The destination must be absolute.")
if destination.exists():
    raise SystemExit("Refusing to overwrite an existing destination.")
manifest = json.loads((source / "asset-manifest.json").read_text())
for number in range(1, 6):
    for report in ["browser-checks.json", "pdf-checks.json"]:
        record = json.loads((source / f"lecture{number}/qa" / report).read_text())
        if not record["passed"]:
            raise SystemExit(f"Lecture {number}: {report} did not pass.")
shutil.copytree(source, destination,
                ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints", ".DS_Store"))
for entry in manifest["files"]:
    target = destination / entry["path"]
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != entry["sha256"]:
        raise SystemExit(f"Copy differs: {entry['path']}")
print(json.dumps({"destination": str(destination), "verifiedFiles": len(manifest["files"]),
                  "bytes": sum(e["bytes"] for e in manifest["files"]), "status": "verified"}))
