# InkTeX

**InkTeX** is an open-source, on-device **bimodal (online ink + rendered vision) handwritten mathematical expression recognition (HMER)** system that turns handwritten strokes (or math images) into clean LaTeX locally with zero cloud calls.

## Recommended Checkpoint & Evaluation Setup

> **Important for evaluators:** Always evaluate with **Beam Search (`beam_width=5`) + Right-to-Left (R2L) bidirectional reranking (`r2l_weight=0.5`, `length_penalty=0.7`)**. The checkpoint includes a jointly trained 3-layer R2L decoder; enabling Beam-5 + R2L adds **+1.75 pp** on CROHME 2019, **+2.70 pp** on CROHME 2023, and **+2.76 pp exact / −0.56 pp CER** on MathWriting over single-pass decoding.

The pretrained `.pt` checkpoint is hosted in GitHub Release [`v0.1.0-models`](https://github.com/Wosca/inktex/releases/tag/v0.1.0-models):

| Checkpoint | Size | Params | Modes |
|---|---:|---:|---|
| **`inktex-75m-stage26-best.pt`** | 320 MB | 71.2M (+12.6M R2L) | Bimodal (online ink + strip) & vision-only (offline images) |

### Official Held-Out Benchmark Results (`inktex-75m-stage26-best.pt`, Unmodified Ground Truth)

| Benchmark | Greedy | Beam-5 | **Beam-5 + R2L (Recommended)** |
|---|---:|---:|---:|
| **CROHME 2019** (official SymLG/LgEval ExpRate, 1,197) | 80.87% (968) | 81.20% (972) | **82.62%** (989) · ≤1 err: 88.81% |
| **CROHME 2023** (official SymLG/LgEval ExpRate, 2,299) | 77.77% (1,788) | 78.03% (1,794) | **80.47%** (1,850) · ≤1 err: 87.86% |
| **MathWriting Test** (7,644 expressions) | — | 66.95% exact / 6.509% CER | **69.71% exact / 5.952% CER** |

### Quick Validation (Python)

```bash
pip install -r ml/requirements.txt

# 1. Download the checkpoint (verifies SHA-256)
python ml/download_checkpoints.py

# 2. Run inference with Beam-5 + R2L reranking (default!) on any .inkml file/folder
python ml/infer.py \
  --checkpoint checkpoints/inktex-75m-stage26-best.pt \
  --inkml path/to/sample_or_dir \
  --beam-width 5 --r2l-weight 0.5 --length-penalty 0.7 --show-nbest

# Or on offline math images (.png / .jpg -> vision-only mode):
python ml/infer.py \
  --checkpoint checkpoints/inktex-75m-stage26-best.pt \
  --image path/to/formula.png

# 3. Run full official CROHME 2019/2023 + MathWriting benchmark evaluation
python ml/eval_benchmarks.py \
  --name stage26_val \
  --checkpoint checkpoints/inktex-75m-stage26-best.pt \
  --out-dir ./eval_out \
  --beam-width 5 --r2l-weight 0.5 --length-penalty 0.7 \
  --mathwriting-test /path/to/mathwriting-2024/test \
  --crohme-test /path/to/crohme23/test.jsonl
```

See [`ml/README.md`](ml/README.md) for full training, evaluation, and checksum details.

## Architecture

InkTeX pairs two complementary encoders with a shared autoregressive LaTeX decoder (`BimodalMathModel`, 71.2M inference / 83.8M with R2L):

- **Vision Strip Encoder (`ConvNeXt-Tiny`):** Reads a normalized 2-channel `128 × W` strip — Channel 0 is anti-aliased ink intensity, Channel 1 encodes monotonic stroke-order progress (`1..255`).
- **Stroke Encoder (8-layer Relative Transformer + Stroke Structure):** Reads 10-channel glyph-normalized online ink features (`x, y, dx, dy, Δt, stroke boundaries, stroke index`) pooled into stroke-level contextual tokens.
- **Shared LaTeX Decoder (6-layer ALiBi Transformer + Visibility-Gated Coverage) + 3-layer R2L Decoder:** Cross-attends over concatenated `[stroke_memory, vision_memory]` tokens with KV-cached Beam-5 search and parallel R2L bidirectional rescoring.

## Repository Layout

```text
inktex/
├── apps/
│   └── ipad/         # Native SwiftUI + PencilKit + offline KaTeX iPadOS app (ONNX Runtime)
├── export/           # Self-contained PyTorch (.pt) -> split KV-cached ONNX exporter + parity tests
├── ml/               # Training, canonicalizer, checkpoint downloader, inference & benchmark eval
└── docs/
    └── ipad-setup.md # Step-by-step Mac / Xcode / M1 iPad Pro setup guide
```

## Quick Start (iPad Pro)

1. Download `inktex-75m-stage26-best.pt` and export it to ONNX (including the R2L scorer for on-device Beam + R2L) using [`export/export_onnx.py`](export/export_onnx.py).
2. Open [`apps/ipad/InkTeX.xcodeproj`](apps/ipad/InkTeX.xcodeproj) in Xcode on your Mac.
3. Follow [`docs/ipad-setup.md`](docs/ipad-setup.md) to drop in `model-bundle/`, select your iPad, and hit **Run**.
