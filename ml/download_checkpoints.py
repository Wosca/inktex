"""Download verified InkTeX model checkpoints and iPad ONNX bundles from GitHub Releases (zero extra deps)."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request
import zipfile

REPO = "Wosca/inktex"

ARTIFACTS: dict[str, dict[str, str]] = {
    "v2": {
        "tag": "v0.2.0-models",
        "filename": "inktex_v2_release.pt",
        "sha256": "58445f26faeb02187df72786afafca6fee7e344e71b050b8d7834ea21ee5e20a",
        "description": "InkTeX v2 Stage C Soup + Tier 1 WriterStyleEncoder PyTorch checkpoint (375.7 MB)",
    },
    "ipad-v2": {
        "tag": "v0.2.0-models",
        "filename": "inktex-v2-ipad-bundle-fp16.zip",
        "sha256": "062f72cf8fcfa69c4890bf177495873178840bab2a0f25017f7063159c1720f3",
        "description": "InkTeX v2 5-graph FP16 ONNX bundle for iPadOS (173.0 MB zip / 199 MB unzipped)",
    },
    "inkforge-v2": {
        "tag": "v0.3.0-inkforge",
        "filename": "inkforge_v2_release.pt",
        "sha256": "2cdda54c73ac535d10c6f2ab33a0eedef0916ee6c2473fa1e6001c206fc36b57",
        "description": "InkForge v2 (62.7M Stage 2.5 v2 STaR+DPO) handwritten math stroke generator checkpoint (239.3 MB)",
    },
    "inkforge-v2-styles": {
        "tag": "v0.3.0-inkforge",
        "filename": "inkforge_v2_writer_styles.npz",
        "sha256": "d1615c7e7719085dcbe06074f05238827105dc89c83c9cc7e4620c25c9a1c331",
        "description": "InkForge v2 658-writer real-human style latent bank (52 KB)",
    },
    "v1": {
        "tag": "v0.1.0-models",
        "filename": "inktex-75m-stage26-best.pt",
        "sha256": "d55df173d3d40205fea272fb4f874cbb7bd7cadf4587af91f09dd7d03b871528",
        "description": "Legacy InkTeX v1 Stage 2.6 PyTorch checkpoint (319.9 MB)",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(out_dir: Path, *, artifact: str = "v2", extract_zip: bool = True) -> Path:
    spec = ARTIFACTS[artifact]
    tag = spec["tag"]
    filename = spec["filename"]
    expected_sha = spec["sha256"]

    dest = out_dir / filename
    out_dir.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        actual = sha256_file(dest)
        if actual == expected_sha:
            print(f"[ok] {dest} already exists and matches SHA-256")
            return _maybe_extract(dest, out_dir, extract_zip)
        print(f"[re-downloading] {dest} checksum mismatch ({actual[:12]} != {expected_sha[:12]})")

    url = f"https://github.com/{REPO}/releases/download/{tag}/{filename}"
    tmp = dest.with_suffix(dest.suffix + ".part")
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
            f"Download failed ({exc}). You can also fetch via GitHub CLI:\n"
            f"  gh release download {tag} -R {REPO} -p {filename} -D {out_dir}"
        ) from exc

    actual = sha256_file(tmp)
    if actual != expected_sha:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"SHA-256 mismatch for {filename}: got {actual}, expected {expected_sha}")
    tmp.replace(dest)
    print(f"[verified] {dest} (sha256={actual[:16]}...)")
    return _maybe_extract(dest, out_dir, extract_zip)


def _maybe_extract(path: Path, out_dir: Path, extract_zip: bool) -> Path:
    if extract_zip and path.suffix.lower() == ".zip":
        print(f"Extracting {path} -> {out_dir}")
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(out_dir)
        extracted = out_dir / "ipad-bundle-v2"
        if extracted.is_dir():
            print(f"[ready] {extracted}")
            return extracted
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=tuple(ARTIFACTS.keys()),
        default="v2",
        help="Artifact to download: v2, ipad-v2, inkforge-v2, inkforge-v2-styles, or v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "checkpoints",
        help="Destination folder (default: <repo>/checkpoints)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Keep .zip archives compressed instead of extracting automatically",
    )
    args = parser.parse_args()
    download(args.output_dir, artifact=args.model, extract_zip=not args.no_extract)


if __name__ == "__main__":
    main()
