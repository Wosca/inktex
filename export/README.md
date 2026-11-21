# InkTeX ONNX Exporter

Exports any `bimodal-checkpoint-v1` PyTorch checkpoint (`best.pt`) into a 4-graph KV-cached ONNX bundle (`vision_encoder.onnx`, `stroke_encoder.onnx`, `memory_proj.onnx`, `decoder_step.onnx`, and `manifest.json`) ready for the iPadOS app (and future web/desktop runtimes).

## Usage

```bash
pip install -r requirements.txt

# Export FP32 bundle (exact PyTorch parity)
python export_onnx.py --checkpoint /path/to/best.pt --output ./model-bundle

# Verify exact numerical & decode parity across bimodal, vision-only, and stroke-only modes
python test_parity.py --checkpoint /path/to/best.pt --bundle ./model-bundle
```

### Optional INT8 Weight Quantization

```bash
python export_onnx.py --checkpoint /path/to/best.pt --output ./model-bundle-int8 --quantize int8
```
