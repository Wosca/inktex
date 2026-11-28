# InkTeX ML (`ml/`)

Self-contained training, dataset packing, LaTeX canonicalization, evaluation, ONNX/LoRA export, and personalization pipelines for **InkTeX v2** (`ml/v2/`) and **InkTeX v1** (`BimodalMathModel`).

## InkTeX v2 Pipeline (`ml/v2/`)

Download the pre-trained **InkTeX v2** PyTorch checkpoint (`inktex_v2_release.pt`) or ready-to-use iPad FP16 ONNX bundle (`inktex-v2-ipad-bundle-fp16.zip`) from GitHub Release [`v0.2.0-models`](https://github.com/Wosca/inktex/releases/tag/v0.2.0-models):

```bash
pip install -r requirements.txt
python download_checkpoints.py --model v2       # -> ../checkpoints/inktex_v2_release.pt
python download_checkpoints.py --model ipad-v2  # -> ../checkpoints/ipad-bundle-v2/
```

### 1. Shard Packing & Training
- **Shard Packing (`v2/pack_v2.py`):** Builds leakage-audited `inktex-shards-v2` tars (`train_online`, `train_offline`, `val_crohme`, `val_mw`).
- **Stage A (`configs/v2_stage_a_stroke.yaml`):** Online sub-stroke chunk encoder + shared bidirectional decoder pretraining.
- **Stage B0 (`configs/v2_stage_b0_vision.yaml`):** ConvNeXt-V2 + FPN vision branch alignment.
- **Stage B (`configs/v2_stage_b_joint.yaml`):** High-resolution joint bimodal fusion with coupled augmentation and mutual stroke KD.
- **Stage C (`configs/v2_stage_c_cooldown.yaml`):** Human-only cooldown (zero synthetic/printed rows).

```bash
cd ml
python -m v2.train_v2 --config configs/v2_stage_c_cooldown.yaml --shards-root /path/to/v2-shards --out-dir /path/to/out --init-from /path/to/stage_b_best.pt
```

### 2. Evaluation
```bash
python -m v2.eval_v2 \
  --checkpoint /path/to/inktex_v2_release.pt \
  --datasets crohme,mathwriting,native \
  --beam-size 5 --r2l-alpha 0.4 \
  --out-dir ./eval_v2_out
```

### 3. Personalization (Tier 1 FiLM & Tier 2 Ink-Only LoRA)
```bash
# Train Tier 1 zero-shot WriterStyleEncoder + FiLM (frozen base model, multi-writer shards)
python -m v2.train_writer_style \
  --checkpoint /path/to/base.pt \
  --shards-root /path/to/v2-shards \
  --output /path/to/inktex_v2_release.pt

# Train a personal ~288 KB `.inktexlora` adapter from native JSONL or iPad-exported corrections.jsonl
python -m v2.train_personal_lora \
  --checkpoint /path/to/inktex_v2_release.pt \
  --data /path/to/inktex_Oscar_corrections.jsonl \
  --output /path/to/oscar_personal.inktexlora \
  --adapter-name "Oscar Personal LoRA"

# Run controlled zero-overlap held-out personalization benchmark
python -m v2.benchmark_personalization \
  --checkpoint /path/to/inktex_v2_release.pt \
  --out-dir ./personalization_bench
```

### 4. iPad FP16 ONNX Bundle Export & Parity Verification
```bash
# Export 5-graph FP16 ONNX bundle + manifest.json (dynamic LoRA + 4x4 static buckets)
python -m v2.export_onnx_v2 \
  --checkpoint /path/to/inktex_v2_release.pt \
  --output ./ipad-bundle-v2 \
  --name "InkTeX v2 (Stage C)" \
  --quantize fp16 --stage 3

# Verify Swift manifest/LoRA contracts, all dynamic bucket shapes, and PyTorch <-> ORT decode parity
python -m v2.verify_ipad_bundle \
  --bundle ./ipad-bundle-v2 \
  --checkpoint /path/to/inktex_v2_release.pt \
  --lora /path/to/oscar_personal.inktexlora
```

---

## InkForge v2 (`ml/inkforge/`)

Writer-style-conditioned 62.7M Transformer-MDN math handwriting generator + recursive 2D equation composer + local InkTeX v2 atomic-cell verifier + blind VLM auditor. Download from GitHub Release [`v0.3.0-inkforge`](https://github.com/Wosca/inktex/releases/tag/v0.3.0-inkforge):

```bash
cd ml
python download_checkpoints.py --model inkforge-v2         # -> ../checkpoints/inkforge_v2_release.pt
python download_checkpoints.py --model inkforge-v2-styles  # -> ../checkpoints/inkforge_v2_writer_styles.npz

# Generate InkTeX-verified multi-line / matrix / cases / STEM synthetic shards (`stroke-latex-v1`)
python -m inkforge.generate_overnight \
  --checkpoint ../checkpoints/inkforge_v2_release.pt \
  --style-bank ../checkpoints/inkforge_v2_writer_styles.npz \
  --judge-ckpt ../checkpoints/inktex_v2_release.pt \
  --out-dir ./inkforge_synth \
  --target-count 50000

# Optional: blind VLM quality audit (Gemini free tier or OpenRouter) with 960x320 single-image recheck
python -m inkforge.verify_openrouter \
  --provider gemini \
  --input ./inkforge_synth/inkforge_synth_00000.jsonl \
  --out-verified ./inkforge_synth/verified_00000.jsonl \
  --mode blind --tiles-per-call 3
```

---

## InkTeX v1 (`inktex-75m-stage26-best.pt`)

Download the v1 checkpoint from GitHub Release [`v0.1.0-models`](https://github.com/Wosca/inktex/releases/tag/v0.1.0-models):

```bash
pip install -r requirements.txt
python download_checkpoints.py --model v1

# Online handwriting (.inkml) or offline math images (.png/.jpg)
python infer.py --checkpoint ../checkpoints/inktex-75m-stage26-best.pt --inkml path/to/file_or_dir --show-nbest
python infer.py --checkpoint ../checkpoints/inktex-75m-stage26-best.pt --image path/to/formula.png
```

## Unit Tests

```bash
pytest tests/ -q
```
