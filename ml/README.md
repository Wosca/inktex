# InkTeX Training (`ml/`)

Self-contained training, dataset packing, LaTeX canonicalization, and official benchmark evaluation pipeline for **InkTeX-75M** (`BimodalMathModel`).

## Pipeline Overview

1. **Shard Packing (`data/bimodal_pack.py`):** Builds leakage-audited tar shards (`stage1_train/`, `stage2_train/`, `val_dev/`) with 2-channel rendered strips (`data/bimodal_raster.py`) and 10-channel normalized online stroke features (`data/preprocessing.py`).
2. **Stage 1 (`configs/bimodal_g4.yaml`):** Vision + decoder pretraining on rendered & offline math strips (`modality_mode: vision`).
3. **Stage 2 (`configs/bimodal_g4.yaml`):** End-to-end bimodal training (online ink + rendered strips) with modality dropout and auxiliary stroke-structure supervision.
4. **Stage 2.5 (`configs/bimodal_stage25_3.yaml`):** Stroke wake-up & CROHME-balanced polish on clean human online ink using single-standard canonical LaTeX (`tokenizer/standardize.py`) and differential learning rates (`vision_lr`, `lr`, `stroke_lr`).
5. **Stage 3 (`configs/bimodal_stage25_3.yaml`):** Clean-human hard-example mining (`--mine-hard`) and anchor-regularized GRPO fine-tuning.
6. **Official Evaluation (`eval_benchmarks.py`):** Unmodified ground-truth evaluation on CROHME 2019 / 2023 (greedy, Beam-5, Beam-5 + R2L with organizer SymLG/LgEval graph scoring) and MathWriting test (7,644).

## Quick Commands

```bash
# Run unit tests (from inktex/ml/)
pytest tests/ -q

# Stage 2.5 polish
python train_bimodal.py --stage 2 --config configs/bimodal_stage25_3.yaml \
  --shards <shards_dir> --output <out_dir> --init-from <stage2_ckpt.pt>

# Official CROHME 2019 / 2023 + MathWriting evaluation
python eval_benchmarks.py --name stage25_best \
  --checkpoint <best.pt> --out-dir <eval_dir>
```

Inference-only model definitions live in [`../export/inktex_model/`](../export/inktex_model/) so the ONNX exporter and iPad app stay self-contained.
