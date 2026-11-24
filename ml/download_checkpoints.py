"""Download the verified InkTeX-75M checkpoint from the GitHub Release (zero extra deps)."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request

REPO = "Wosca/inktex"
RELEASE_TAG = "v0.1.0-models"
FILENAME = "inktex-75m-stage26-best.pt"
EXPECTED_SHA256 = "d55df173d3d40205fea272fb4f874cbb7bd7cadf4587af91f09dd7d03b871528"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(out_dir: Path) -> Path:
    dest = out_dir / FILENAME
    out_dir.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        actual = sha256_file(dest)
        if actual == EXPECTED_SHA256:
            print(f"[ok] {dest} already exists and matches SHA-256")
            return dest
        print(f"[re-downloading] {dest} checksum mismatch ({actual[:12]} != {EXPECTED_SHA256[:12]})")

    url = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{FILENAME}"
    tmp = dest.with_suffix(".pt.part")
    print(f"Downloading {url} -> {dest}")

    def progress(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100.0, blocks * block_size * 100.0 / total)
            mb = blocks * block_size / (1024 * 1024)
            tot_mb = total / (1024 * 1024)
            print(f"\r  {mb:6.1f} / {tot_mb:6.1f} MB ({pct:5.1f}%)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=progress)
        print()
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"Download failed ({exc}). If the repo is still private, use:\n"
            f"  gh release download {RELEASE_TAG} -R {REPO} -p {FILENAME} -D {out_dir}"
        ) from exc

    actual = sha256_file(tmp)
    if actual != EXPECTED_SHA256:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"SHA-256 mismatch for {FILENAME}: got {actual}, expected {EXPECTED_SHA256}")
    tmp.replace(dest)
    print(f"[verified] {dest} (sha256={actual[:16]}...)")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "checkpoints",
        help="Destination folder (default: <repo>/checkpoints)",
    )
    args = parser.parse_args()
    download(args.output_dir)


if __name__ == "__main__":
    main()
