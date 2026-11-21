# InkTeX

**InkTeX** is an open-source, on-device **bimodal (online ink + rendered vision) handwritten mathematical expression recognition (HMER)** system that turns Apple Pencil strokes into clean LaTeX locally with zero cloud calls.

## Architecture

InkTeX pairs two complementary encoders with a shared autoregressive LaTeX decoder (`BimodalMathModel`, ~75M parameters):

- **Vision Strip Encoder (`ConvNeXt-Tiny`):** Reads a normalized 2-channel `128 × W` strip — Channel 0 is anti-aliased ink intensity, Channel 1 encodes monotonic stroke-order progress (`1..255`).
- **Stroke Encoder (8-layer Relative Transformer + Stroke Structure):** Reads 10-channel glyph-normalized online ink features (`x, y, dx, dy, Δt, stroke boundaries, stroke index`) pooled into stroke-level contextual tokens.
- **Shared LaTeX Decoder (6-layer ALiBi Transformer + Visibility-Gated Coverage):** Cross-attends over concatenated `[stroke_memory, vision_memory]` tokens with KV-cached incremental decoding.

## Repository Layout

```text
inktex/
├── apps/
│   └── ipad/         # Native SwiftUI + PencilKit + offline KaTeX iPadOS app (ONNX Runtime)
├── export/           # Self-contained PyTorch (.pt) -> split KV-cached ONNX exporter + parity tests
├── ml/               # Training pipeline & datasets (landing zone for Stage 1–3 code)
└── docs/
    └── ipad-setup.md # Step-by-step Mac / Xcode / M1 iPad Pro setup guide
```

## Quick Start (iPad Pro)

1. Export your `bimodal-checkpoint-v1` checkpoint (`best.pt`) using [`export/export_onnx.py`](export/export_onnx.py) (or copy a pre-exported `model-bundle/` folder).
2. Open [`apps/ipad/InkTeX.xcodeproj`](apps/ipad/InkTeX.xcodeproj) in Xcode on your Mac.
3. Follow [`docs/ipad-setup.md`](docs/ipad-setup.md) to drop in `model-bundle/`, select your iPad, and hit **Run**.
