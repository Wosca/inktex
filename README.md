# InkTeX

**InkTeX** is an open-source, on-device **bimodal (online ink + rendered vision) handwritten mathematical expression recognition (HMER)** system that turns handwritten strokes or math images into clean LaTeX locally on iPadOS and desktop with zero cloud calls.

## Highlights (InkTeX v2)

- **Bimodal Architecture (`InkTeXV2Model`, 98.3M params / 199 MB FP16 ONNX bundle):**
  - **Sub-Stroke Chunk Encoder (13L, `d=576`):** Encodes 20-channel glyph-normalized online ink features in overlapping 8-point chunks across static buckets `{128, 256, 512, 768}`.
  - **Vision Strip Encoder (`ConvNeXt-V2 + FPN`):** Reads a normalized 2-channel `192 × W` strip (anti-aliased ink intensity + monotonic stroke-order progress) across static width buckets `{256, 512, 768, 1024}`.
  - **Bimodal Cross-Attention Fusion & Shared Bidirectional Decoder (6L ALiBi):** Single decoder handles both L2R KV-cached beam search and parallel R2L bidirectional reranking with zero extra R2L weights.
- **On-Device 3-Tier Handwriting Personalization:**
  - **Tier 1 (Zero-Shot FiLM):** `WriterStyleEncoder` extracts a 64-d writer vector from 5–20 calibration strokes (`<0.5 ms` on iPad) with guaranteed bit-for-bit invariance when uncalibrated.
  - **Tier 2 (Hot-Swappable Ink-Only LoRA):** Adapts only `q/v/o` projections in stroke blocks `9–12` (`r=8`, `alpha=16`, **288 KB** `.inktexlora`) with zero model reload and zero vision/decoder drift.
  - **Tier 3 (Correction Capture & Export):** Save corrections directly on iPad, optionally enable instant exact-geometry caching, and export `corrections.jsonl` for fast `<60s` personal LoRA fine-tuning.

## Benchmark Results

### InkTeX v2 (Stage C Soup, GitHub Release [`v0.2.0-models`](https://github.com/Wosca/inktex/releases/tag/v0.2.0-models), Beam-5 + R2L `α=0.4`, Unmodified Ground Truth)

| Benchmark | Metric | Score |
|---|---|---:|
| **CROHME 2019** (1,197) | Official SymLG/LgEval ExpRate (`D_B=0`) | **82.21%** (`≤1 err: 87.89%`) |
| **CROHME 2023** (2,299) | Official SymLG/LgEval ExpRate (`D_B=0`) | **78.56%** (`≤1 err: 86.43%`) |
| **MathWriting Test** (7,644) | Std v2 Exact / Google Ref CER | **68.03%** / **5.96% CER** |
| **Held-Out Personal Handwriting** (500 disjoint eqs) | Base → **150-eq Personal LoRA (288 KB)** | **71.80% → 78.80%** (`+7.00 pp`, CER `3.67% → 2.90%`) |

### InkTeX v1 (`inktex-75m-stage26-best.pt`, GitHub Release [`v0.1.0-models`](https://github.com/Wosca/inktex/releases/tag/v0.1.0-models))

| Benchmark (Beam-5 + R2L) | ExpRate / Exact |
|---|---:|
| **CROHME 2019** (SymLG/LgEval, 1,197) | **82.62%** (`≤1 err: 88.81%`) |
| **CROHME 2023** (SymLG/LgEval, 2,299) | **80.47%** (`≤1 err: 87.86%`) |
| **MathWriting Test** (7,644) | **69.71% exact / 5.952% CER** |

## Repository Layout

```text
inktex/
├── apps/
│   └── ipad/         # Native SwiftUI + PencilKit + offline KaTeX iPadOS app (ONNX Runtime)
├── export/           # v1 PyTorch (.pt) -> split KV-cached ONNX exporter & parity tests
└── ml/               # Training, evaluation, ONNX v2 export, and personalization pipelines
    ├── configs/      # YAML configs for v1 and v2 training stages
    ├── v2/           # InkTeX v2 model, training, eval, ONNX/LoRA exporter, personalization
    └── tests/        # Unit tests
```

## Quick Start (iPad App in Xcode)

1. **Download the pre-exported FP16 iPad bundle** from GitHub Release [`v0.2.0-models`](https://github.com/Wosca/inktex/releases/tag/v0.2.0-models) (`inktex-v2-ipad-bundle-fp16.zip`, or via `python ml/download_checkpoints.py --model ipad-v2`), or export from a PyTorch checkpoint:
   ```bash
   cd ml
   python -m v2.export_onnx_v2 \
     --checkpoint ../checkpoints/inktex_v2_release.pt \
     --output ./ipad-bundle-v2 \
     --name "InkTeX v2 (Stage C)" \
     --quantize fp16 --stage 3
   ```
2. **Open [`apps/ipad/InkTeX.xcodeproj`](apps/ipad/InkTeX.xcodeproj)** in Xcode on macOS and wait for Swift Package Manager to resolve `onnxruntime`.
3. Select the **InkTeX** target → **Signing & Capabilities** → choose your Apple ID team.
4. **Add the model bundle:**
   - **Bundled in app:** Drag `ipad-bundle-v2/` into the `InkTeX` folder in Xcode's sidebar, choosing **Copy files to destination** + **Create folder references** (blue folder icon) and target **InkTeX**.
   - **Or hot-swappable via Files/AirDrop:** Build without bundling, AirDrop `ipad-bundle-v2/` to your iPad, then tap **Settings → Import `model-bundle` Folder from Files…**.
5. Set the scheme Build Configuration to **Release** for full M1/M-series speed, select your iPad, and press **⌘R**.
6. *(Optional)* **Load a Personal LoRA (`.inktexlora`):** AirDrop your `.inktexlora` file to iPad, open the **Profile** sheet (top-left), and tap **Import `.inktexlora` Adapter…** for instant hot-swapping.

## Quick Start (Python ML & Personal LoRA)

See [`ml/README.md`](ml/README.md) for training, evaluation, ONNX bundle verification, and personal LoRA fine-tuning commands.
