# InkTeX Training (`ml/`)

This directory is reserved for the clean open-source migration of the 3-stage `InkTeX-75M` training pipeline:

1. **Stage 1:** Vision + decoder pretraining on rendered & offline math strips (`modality_mode: vision`).
2. **Stage 2:** End-to-end bimodal training (online ink + rendered strips) with modality dropout and auxiliary stroke-structure supervision.
3. **Stage 3:** Hard-example structure-reward GRPO fine-tuning.

Inference-only model definitions live in [`../export/inktex_model/`](../export/inktex_model/) so the ONNX exporter and iPad app stay self-contained.
