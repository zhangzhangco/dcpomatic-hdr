# DCP-o-matic SDR-to-HDR Neural Pipeline

High-efficiency neural network pipeline with GPU acceleration for generating DCI-compliant HDR DCPs (PQ, ST 2084).

## 🚀 Key Features

- **Zero Configuration**: Automatic model download; no manual setup required.
- **High Performance**: Auto-detects NVIDIA GPU (CUDA) for acceleration with seamless CPU fallback.
- **High Fidelity**: Enforced **Hue Lock** strategy ensures zero color shifting.
- **Automated Workflow**: Built-in Gamma 2.4 decoding automatically handles SDR inputs correctly.

## 🛠️ User Guide

1. **Enable Feature**: 
   - Go to **Edit -> Preferences -> Neural HDR**.
   - Check **Enable Neural HDR Processing**.

2. **Make DCP**:
   - Import SDR content as usual.
   - Click **Jobs -> Make DCP**.
   - *Note: It is normal for the preview window to appear dark (PQ signal on SDR monitor).*

## ⚙️ Technical Specs

- **Model**: Neural HDR v4 (Opset 17) via ONNX Runtime.
- **Deployment**: Libraries in `deps/onnxruntime`; model auto-downloaded relative to executable.

## ✅ Verification (Optional)

Check output MXF for SMPTE ST 2084 metadata:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/deps/onnxruntime/lib
./deps/asdcplib/src/.libs/asdcp-info -d path/to/video.mxf | grep "Transfer"
# Expected: Transfer Characteristic: SMPTE ST 2084
```

---
**Author**: zhangxin
**Last Updated**: 2026-01-13 (v4.0)
