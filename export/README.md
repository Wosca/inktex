# InkTeX ONNX Exporter

> **Note:** For **InkTeX v2** (5-graph FP16 bundle with `WriterStyleEncoder` + hot-swappable `.inktexlora`), use [`ml/v2/export_onnx_v2.py`](../ml/v2/export_onnx_v2.py) and [`ml/v2/verify_ipad_bundle.py`](../ml/v2/verify_ipad_bundle.py).

This directory contains the legacy **InkTeX v1** (`bimodal-checkpoint-v1`) exporter, producing a 4-graph KV-cached ONNX bundle (`vision_encoder.onnx`, `stroke_encoder.onnx`, `memory_proj.onnx`, `decoder_step.onnx`, optional `r2l_scorer.onnx`, and `manifest.json`).

## Usage (v1)

```bash
pip install -r requirements.txt

# Export bundle
python export_onnx.py --checkpoint /path/to/best.pt --output ./model-bundle

# Verify numerical & decode parity across bimodal, vision-only, and stroke-only modes
python test_parity.py --checkpoint /path/to/best.pt --bundle ./model-bundle
```
