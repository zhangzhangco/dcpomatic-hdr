# DCP-o-matic SDR-to-HDR Neural Pipeline

Integrates a neural network-based SDR-to-HDR upscaling pipeline, supporting GPU acceleration, enabling the creation of DCI-compliant HDR DCPs (PQ, ST 2084).

## 🚀 Quick Start

The HDR functionality is now fully integrated, supporting **automatic model downloading** and **GPU acceleration** for a zero-configuration, out-of-the-box experience.

### 1. Enable Feature (One-time Setup)

1. Go to **Edit -> Preferences**.
2. Select the **Neural HDR** tab.
3. **Enable Neural HDR Processing**: Check this box. This will intercept the video processing pipeline and convert SDR signals to HDR using the neural network.
4. **Enable Hue Lock Strategy**: Recommended. This forces the HDR output to maintain the exact chromaticity of the SDR input, preventing AI-induced color shifts.

**🤖 Automatic Model Download**:
When enabled and run for the first time, the application will automatically detect and download the required `.onnx` model files from GitHub Releases. No manual path configuration is needed.

### 2. Preview & Create

1. **Preview**: After loading your content, the preview window may appear "Dark". This is normal because the system is outputting PQ (ST 2084) signals, which look washed out or dark on standard SDR monitors.
2. **Make DCP**: Click **Jobs -> Make DCP**. The output DCP will automatically include the correct HDR metadata.

### 3. Hardware Acceleration

The pipeline includes built-in **ONNX Runtime (CUDA)** support:
*   **GPU Mode**: If an NVIDIA GPU and CUDA environment are detected, inference will automatically use the GPU for significantly faster processing.
*   **CPU Fallback**: If the GPU is unavailable (e.g., out of memory or missing drivers), the application seamlessly switches to CPU mode to ensure the task continues without interruption.

## ⚙️ Core Technology

### Hue Lock
*   **Principle**: The neural network predicts only the Luminance Gain, while locking the chromaticity to the original SDR values.
*   **Benefit**: Completely resolves color shifts or temporal flickering often associated with generative models, ensuring "true-to-source" color style while expanding dynamic range.

### Gamma Processing
*   The pipeline defaults to using **Gamma 2.4** to decode SDR input, ensuring that the data fed into the neural network is Linear Light.

## 📊 Debugging & Verification

When running CLI tools (`dcpomatic2_create` or `dcpomatic2_cli`) with Debug mode enabled, per-frame statistics (e.g., Peak Nits, Hue Shift) are output to the console.

## 🛠️ Build & Deployment

This project depends on the `onnxruntime` C++ library.

1. **Libraries**: Ensure `deps/onnxruntime/lib` contains `libonnxruntime.so` (and `_cuda.so` for GPU support).
2. **Environment**:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/deps/onnxruntime/lib
   ```
3. **Model Files**: Hardcoded to look for `neural_hdr.onnx` relative to the executable. If missing, it attempts auto-download.

## ✅ Verify HDR Compliance

To verify that the generated DCP contains HDR (ST 2084 / PQ) metadata, use `asdcp-info`:

```bash
# 1. Set library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/deps/asdcplib/src/.libs

# 2. Run check
./deps/asdcplib/src/.libs/asdcp-info -d path/to/video.mxf | grep -A 5 "Transfer"
```
You should see `Transfer Characteristic: SMPTE ST 2084`.

---
**Author**: zhangxin
**Last Updated**: 2026-01-13 (v4.0 Auto-Download & GPU)
