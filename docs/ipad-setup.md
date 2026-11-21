# Running InkTeX on Your M1 iPad Pro

## 1. Clone the Repo on Your Mac

```bash
git clone https://github.com/Wosca/inktex.git
cd inktex
```

## 2. Export (or Copy) the `model-bundle/`

You have two options:

### Option A — Copy the pre-exported `model-bundle/` from your Windows PC (Fastest)
Copy `E:\math-recogniser-training\bimodal_75m_v1\stage2\model-bundle\` (5 files: `manifest.json`, `vision_encoder.onnx`, `stroke_encoder.onnx`, `memory_proj.onnx`, `decoder_step.onnx`) to your Mac via USB, SMB, or iCloud Drive.

### Option B — Export from any `best.pt` on your Mac or PC
```bash
cd export
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python export_onnx.py --checkpoint /path/to/best.pt --output ./model-bundle
python test_parity.py --checkpoint /path/to/best.pt --bundle ./model-bundle
```
*(Add `--quantize int8` to shrink the bundle from ~289 MB to ~115 MB with minimal quality impact.)*

## 3. Open in Xcode & Install on Your M1 iPad Pro

1. Open `apps/ipad/InkTeX.xcodeproj` in Xcode.
2. Wait ~15 seconds for Swift Package Manager to resolve `onnxruntime` (1.24.2).
   > **Note:** Upstream `microsoft/onnxruntime-swift-package-manager` went offline in Nov 2026; this project pins the verified byte-for-byte mirror `thesagezm/onnxruntime-swift-package-manager` (commit `7dbd851c`, SHA-256 verified against Microsoft's official 1.24.2 pod archive).
3. Click the **InkTeX** project → **InkTeX** target → **Signing & Capabilities**:
   - Select your Apple ID under **Team** (free Apple ID works fine for personal testing!).
   - If Xcode says `com.wosca.inktex` is taken, change Bundle Identifier to something unique like `com.wosca.inktex.dev`.
4. **Add the model weights (choose either):**
   - **Bundled in app (zero setup on launch):** Drag the `model-bundle` folder into the `InkTeX` group in Xcode's sidebar. Make sure **"Create folder references"** (blue folder icon) and target **"InkTeX"** are checked!
   - **Or hot-swappable via Files / AirDrop (no rebuild needed when Stage 2 finishes or Stage 3 arrives!):** Build & run without bundling `model-bundle`, AirDrop `model-bundle` to your iPad (Save to Files), then tap **Import…** inside InkTeX and pick the `model-bundle` folder.
5. Plug in your M1 iPad Pro, select it as the run destination, and press **⌘R** (tip: switch Scheme to **Release** for maximum speed on M1!).

## Tips While Testing
- **Toolbar Modality Switcher:** Compare **Bimodal** (ink + strip), **Vision** (strip only), and **Stroke** (ink only) live on the exact same drawing.
- **Inspector Button** (waveform icon): Shows the exact 2-channel `128×W` vision strip (colored cyan→magenta by stroke order), encode/decode latency breakdown, and individual LaTeX tokens.
- **CPU Button:** Lets you toggle **CoreML EP** for the vision encoder or hot-swap a newer exported `model-bundle` from Files at any time.
