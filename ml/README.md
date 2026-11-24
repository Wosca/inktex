# InkTeX ML (`ml/`)

Self-contained training, dataset packing, LaTeX canonicalization, inference, and official benchmark evaluation pipeline for **InkTeX-75M** (`BimodalMathModel`).

## Validating the Checkpoint

### 1. Recommended Checkpoint & Decoding

Download the checkpoint from GitHub Release [`v0.1.0-models`](https://github.com/Wosca/inktex/releases/tag/v0.1.0-models):

```bash
pip install -r requirements.txt
python download_checkpoints.py
```

| File | SHA-256 | Recommended Decoding |
|---|---|---|
| **`inktex-75m-stage26-best.pt`** | `d55df173d3d40205fea272fb4f874cbb7bd7cadf4587af91f09dd7d03b871528` | **Beam-5 + R2L** (`--beam-width 5 --r2l-weight 0.5 --length-penalty 0.7`) |

**Why Beam-5 + R2L matters:** `inktex-75m-stage26-best.pt` ships with both a 6-layer L2R decoder and a jointly trained 3-layer R2L decoder (and embeds its 907-token tokenizer inside the `.pt` file; a standalone copy is also at [`tokenizer/bimodal_tokenizer.json`](tokenizer/bimodal_tokenizer.json)). Reranking the top-5 L2R beams with the R2L decoder (`0.5 * l2r_norm + 0.5 * r2l_norm`, length penalty `0.7`) yields substantial gains on every held-out benchmark:

| Benchmark (Unmodified Ground Truth) | Greedy | Beam-5 | **Beam-5 + R2L (Recommended)** |
|---|---:|---:|---:|
| **CROHME 2019** (SymLG/LgEval, 1,197) | 80.87% (968) | 81.20% (972) | **82.62%** (989) |
| **CROHME 2023** (SymLG/LgEval, 2,299) | 77.77% (1,788) | 78.03% (1,794) | **80.47%** (1,850) |
| **MathWriting Test** (7,644) | — | 66.95% / 6.509% CER | **69.71% / 5.952% CER** |

### 2. Inference on `.inkml` or Math Images

`infer.py` defaults to **Beam-5 + R2L** (`--beam-width 5 --r2l-weight 0.5 --length-penalty 0.7`):

```bash
# Online handwriting (.inkml file or directory -> full bimodal mode)
python infer.py --checkpoint ../checkpoints/inktex-75m-stage26-best.pt --inkml path/to/file_or_dir --show-nbest

# Offline math images (.png/.jpg -> vision-only mode)
python infer.py --checkpoint ../checkpoints/inktex-75m-stage26-best.pt --image path/to/formula.png

# Fast single-pass comparison
python infer.py --checkpoint ../checkpoints/inktex-75m-stage26-best.pt --inkml path/to/file.inkml --greedy
```

### 3. Official Benchmark Reproduction

```bash
# MathWriting test (7,644 .inkml files)
python eval_benchmarks.py --name stage26_mw \
  --checkpoint ../checkpoints/inktex-75m-stage26-best.pt \
  --out-dir ./eval_mw \
  --datasets mathwriting \
  --mathwriting-test /path/to/mathwriting-2024/test \
  --beam-width 5 --r2l-weight 0.5 --length-penalty 0.7

# CROHME 2019 + 2023 (requires organizer SymLG/LgEval tools)
python eval_benchmarks.py --name stage26_crohme \
  --checkpoint ../checkpoints/inktex-75m-stage26-best.pt \
  --out-dir ./eval_crohme \
  --datasets crohme \
  --crohme-test /path/to/crohme23/test.jsonl \
  --crohme-tools /path/to/crohme-eval \
  --crohme-archive /path/to/CROHME23.zip \
  --beam-width 5 --r2l-weight 0.5 --length-penalty 0.7
```

## Training Pipeline Overview

1. **Shard Packing (`data/bimodal_pack.py`):** Leakage-audited tar shards (`stage1_train/`, `stage2_train/`, `val_dev/`) with 2-channel strips (`data/bimodal_raster.py`) and 10-channel glyph-normalized stroke features (`data/preprocessing.py`).
2. **Stage 1 (`configs/bimodal_g4.yaml`):** Vision + decoder pretraining (`modality_mode: vision`).
3. **Stage 2 (`configs/bimodal_g4.yaml`):** End-to-end bimodal training with modality dropout and auxiliary stroke-structure supervision.
4. **Stage 2.5 (`configs/bimodal_stage25_3.yaml`):** Stroke wake-up & CROHME-balanced polish on clean human online ink with canonical LaTeX (`tokenizer/standardize.py`) and differential learning rates.
5. **Stage 2.6 (`configs/bimodal_stage26.yaml`):** Coupled image+stroke geometric augmentation, vision strip band masking, and agreement-gated logit distillation.

## Unit Tests

```bash
pytest tests/ -q
```

Inference-only model definitions live in [`../export/inktex_model/`](../export/inktex_model/) so the ONNX exporter and iPad app stay self-contained.
