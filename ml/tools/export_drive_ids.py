"""Export Google Drive cloud file IDs for the Colab folder from the local DriveFS metadata DB.

    python tools/export_drive_ids.py [--pack-zip] [--embed-notebook colab/train_bimodal_g4.ipynb]

The DB (plus -wal/-shm) is copied to a temporary directory and opened read-only, so DriveFS is never locked.
--pack-zip rewrites math-recogniser.zip in place, which keeps its Drive file ID (and public link) unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile
import zipfile

ML = Path(__file__).resolve().parents[1]
FORMAT = "drive-file-ids-v1"
DEFAULT_FOLDER = "math-recogniser-colab"
DEFAULT_DRIVE_DIR = Path(r"G:\My Drive") / DEFAULT_FOLDER
IDS_NAME = "drive_file_ids.json"
ZIP_NAME = "math-recogniser.zip"
ZIP_ROOT = "math-recogniser/ml"
EXCLUDED_DIRS = {".venv", "__pycache__", ".pytest_cache", "checkpoints", ".smoke-checkpoints", "datasets", ".git",
                 "node_modules", ".ipynb_checkpoints", "cache", "src"}
EXCLUDED_SUFFIXES = {".pt", ".pth", ".onnx", ".data", ".tar", ".gz", ".zip", ".npz", ".png", ".pyc", ".jsonl", ".bin",
                     ".safetensors", ".ckpt"}
BEGIN_MARK = "# BEGIN EMBEDDED_DRIVE_IDS"
END_MARK = "# END EMBEDDED_DRIVE_IDS"


def find_db(override: Path | None = None) -> Path:
    if override is not None:
        if not Path(override).is_file():
            raise SystemExit(f"--db {override} does not exist")
        return Path(override)
    root = Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "DriveFS"
    found = sorted(root.glob("*/metadata_sqlite_db"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not found:
        raise SystemExit(f"No DriveFS metadata_sqlite_db under {root}; pass --db")
    if len(found) > 1:
        print(f"note: {len(found)} DriveFS accounts found, using the most recently updated {found[0].parent.name}")
    return found[0]


def snapshot(db: Path, directory: Path) -> Path:
    copy = directory / db.name
    for suffix in ("", "-wal", "-shm"):
        source = db.with_name(db.name + suffix)
        if source.is_file():
            shutil.copyfile(source, directory / source.name)
    return copy


def connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def live(row: sqlite3.Row) -> bool:
    return not row["trashed"] and not row["is_tombstone"]


def has_cloud_id(value: str | None) -> bool:
    return bool(value) and not value.startswith("local-")


def find_root(connection: sqlite3.Connection, folder: str, root_id: str | None = None) -> sqlite3.Row:
    rows = connection.execute(
        "SELECT stable_id, id FROM items WHERE local_title = ? AND is_folder = 1 AND trashed = 0 "
        "AND is_tombstone = 0", (folder,)).fetchall()
    if root_id:
        rows = [row for row in rows if row["id"] == root_id]
    if not rows:
        raise SystemExit(f"No live Drive folder named {folder!r}{f' with id {root_id}' if root_id else ''}")
    if len(rows) > 1:
        raise SystemExit(f"{len(rows)} folders named {folder!r} ({', '.join(str(row['id']) for row in rows)}); "
                         "pass --root-id")
    return rows[0]


def walk(connection: sqlite3.Connection, root_stable_id: int) -> tuple[dict[str, dict], list[str]]:
    files: dict[str, dict] = {}
    chosen: dict[str, int] = {}
    pending: list[str] = []
    stack: list[tuple[int, str]] = [(root_stable_id, "")]
    while stack:
        parent, prefix = stack.pop()
        children = connection.execute(
            "SELECT i.stable_id, i.id, i.trashed, i.is_folder, i.file_size, i.is_tombstone, i.local_title "
            "FROM items i JOIN stable_parents p ON p.item_stable_id = i.stable_id WHERE p.parent_stable_id = ?",
            (parent,)).fetchall()
        for row in children:
            if not live(row) or not row["local_title"]:
                continue
            path = prefix + row["local_title"]
            if row["is_folder"]:
                stack.append((row["stable_id"], path + "/"))
            elif not has_cloud_id(row["id"]):
                pending.append(path)
            elif path not in chosen or row["stable_id"] > chosen[path]:
                if path in chosen:
                    print(f"warning: duplicate Drive entries for {path}; keeping the newest")
                chosen[path] = row["stable_id"]
                files[path] = {"id": row["id"], "size": int(row["file_size"] or 0)}
    return dict(sorted(files.items())), sorted(pending)


def directory_of(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else "."


def build_export(files: dict[str, dict], folder: str, root_id: str) -> dict:
    return {"format": FORMAT, "root_folder": folder, "root_folder_id": root_id,
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "counts": {"total": len(files), "by_dir": dict(sorted(Counter(map(directory_of, files)).items()))},
            "files": files}


def export_ids(db: Path, folder: str = DEFAULT_FOLDER, root_id: str | None = None) -> tuple[dict, list[str]]:
    with tempfile.TemporaryDirectory(prefix="drivefs-") as directory:
        connection = connect_readonly(snapshot(db, Path(directory)))
        try:
            connection.row_factory = sqlite3.Row
            root = find_root(connection, folder, root_id)
            files, pending = walk(connection, root["stable_id"])
        finally:
            connection.close()
    return build_export(files, folder, root["id"]), pending


def untracked_local_files(drive_dir: Path, known: set[str]) -> list[str]:
    if not drive_dir.is_dir():
        return []
    found = []
    for path in drive_dir.rglob("*"):
        if path.is_file() and not path.name.startswith(("~$", ".")) and path.name.lower() != "desktop.ini":
            relative = path.relative_to(drive_dir).as_posix()
            if relative not in known:
                found.append(relative)
    return sorted(found)


# --------------------------------------------------------------------------- zip


def zip_members(ml_root: Path) -> list[Path]:
    members = []
    for directory, folders, names in os.walk(ml_root):
        folders[:] = sorted(folder for folder in folders if folder not in EXCLUDED_DIRS)
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix.lower() not in EXCLUDED_SUFFIXES and not name.endswith((".tmp", ".part")):
                members.append(path)
    return members


def pack_zip(target: Path, ml_root: Path = ML) -> tuple[int, int]:
    """Writes the archive in %TEMP%, then overwrites ``target`` in place so Drive keeps the file ID."""
    members = zip_members(ml_root)
    with tempfile.TemporaryDirectory(prefix="math-recogniser-zip-") as directory:
        staged = Path(directory) / ZIP_NAME
        with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in members:
                archive.write(path, f"{ZIP_ROOT}/{path.relative_to(ml_root).as_posix()}")
        with zipfile.ZipFile(staged) as archive:
            if (broken := archive.testzip()) is not None:
                raise SystemExit(f"zip self-check failed at {broken}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged, target)
    size = target.stat().st_size
    with open(target, "rb") as handle:
        if handle.read(4) != b"PK\x03\x04":
            raise SystemExit(f"{target} is not a zip after copy")
    return len(members), size


# --------------------------------------------------------------------------- notebook


def embedded_entries(files: dict[str, dict]) -> dict[str, dict]:
    return {path: entry for path, entry in files.items() if path != IDS_NAME and not path.endswith(".ipynb")}


def render_embedded(files: dict[str, dict]) -> str:
    lines = [f"{BEGIN_MARK} (tools/export_drive_ids.py --embed-notebook)", "EMBEDDED_DRIVE_IDS = {"]
    lines += [f"    {json.dumps(path)}: {json.dumps(entry)}," for path, entry in embedded_entries(files).items()]
    lines += ["}", END_MARK]
    return "\n".join(lines)


def embed_notebook(notebook: Path, export: dict) -> tuple[int, str]:
    data = json.loads(notebook.read_text(encoding="utf-8"))
    files = export["files"]
    ids_file_id = (files.get(IDS_NAME) or {}).get("id", "")
    block = re.compile(re.escape(BEGIN_MARK) + r".*?" + re.escape(END_MARK), re.S)
    ids_line = re.compile(r'^DRIVE_IDS_FILE_ID = "[^"\n]*"', re.M)
    patched = False
    for cell in data["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not block.search(source):
            continue
        source = block.sub(lambda _: render_embedded(files), source, count=1)
        source = ids_line.sub(lambda _: f"DRIVE_IDS_FILE_ID = {json.dumps(ids_file_id)}", source, count=1)
        cell["source"] = source.splitlines(keepends=True)
        patched = True
        break
    if not patched:
        raise SystemExit(f"{notebook} has no {BEGIN_MARK} ... {END_MARK} block")
    notebook.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(embedded_entries(files)), ids_file_id


# --------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, help="DriveFS metadata_sqlite_db (default: auto-discover)")
    parser.add_argument("--folder", default=DEFAULT_FOLDER)
    parser.add_argument("--root-id", help="cloud ID of the folder when several share its name")
    parser.add_argument("--drive-dir", type=Path, default=DEFAULT_DRIVE_DIR, help="local DriveFS path of --folder")
    parser.add_argument("--output", type=Path, help=f"default: <drive-dir>/{IDS_NAME}")
    parser.add_argument("--embed-notebook", type=Path)
    parser.add_argument("--pack-zip", action="store_true", help=f"refresh <drive-dir>/{ZIP_NAME} in place from ml/")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output or args.drive_dir / IDS_NAME
    export, pending = export_ids(find_db(args.db), args.folder, args.root_id)
    files = export["files"]
    if args.pack_zip:
        count, size = pack_zip(args.drive_dir / ZIP_NAME)
        print(f"packed {count} files -> {args.drive_dir / ZIP_NAME} ({size / 2**20:.2f} MiB, overwritten in place)")
        if ZIP_NAME in files:
            files[ZIP_NAME]["size"] = size
        else:
            print(f"note: {ZIP_NAME} has no cloud ID yet; re-run after Drive uploads it")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Rewritten in place (no rename) so Drive keeps this file's ID too.
    output.write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")

    print(f"exported {export['counts']['total']} files from {export['root_folder']} "
          f"({export['root_folder_id']}) -> {output}")
    for directory, count in export["counts"]["by_dir"].items():
        print(f"  {directory:<28} {count}")
    for name in (ZIP_NAME, IDS_NAME):
        print(f"  {name}: {(files.get(name) or {}).get('id', '<no cloud ID yet>')}")
    known = set(files) | {output.relative_to(args.drive_dir).as_posix()
                          if output.is_relative_to(args.drive_dir) else ""}
    waiting = sorted(set(pending) | set(untracked_local_files(args.drive_dir, known)))
    if waiting:
        print(f"{len(waiting)} local file(s) without a cloud ID yet (still uploading; re-run when synced):")
        groups: dict[str, list[str]] = {}
        for path in waiting:
            groups.setdefault(directory_of(path), []).append(path.rsplit("/", 1)[-1])
        for directory, names in groups.items():
            shown = ", ".join(names[:3]) + (f", ... (+{len(names) - 3})" if len(names) > 3 else "")
            print(f"  {directory:<28} {len(names):>4}  {shown}")
    if args.embed_notebook:
        embedded, ids_file_id = embed_notebook(args.embed_notebook, export)
        print(f"embedded {embedded} IDs into {args.embed_notebook}; DRIVE_IDS_FILE_ID = {ids_file_id or '<pending>'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
