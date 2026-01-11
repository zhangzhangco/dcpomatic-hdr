# DCP-o-matic SDR-to-HDR Neural Pipeline

Integration of a neural network-based SDR-to-HDR upscaling pipeline into DCP-o-matic, enabling the creation of DCI-compliant HDR DCPs (PQ, ST 2084).

## 🚀 Quick Start

To encode an HDR DCP from an SDR source:

```bash
# 1. Set Local Library Path (Critical for Linux)
export LD_LIBRARY_PATH=$(pwd)/build/src/lib:$(pwd)/build/src/wx:$(pwd)/local_target/lib:$(pwd)/deps/onnxruntime/lib:$LD_LIBRARY_PATH

# 2. Enable HDR Pipeline
export ZHANGXIN_HDR_ENABLE=1

# 3. Point to your ONNX Model
export ZHANGXIN_HDR_MODEL="/home/zhangxin/models/v6_latest.onnx"

# 4. (Optional) Set Input Transfer Function. 
#    Default is 2.4 (Safe for most cases). 
#    Only change if you know your source is DCI-graded (2.6) or Scene Linear (REC709).
# export ZHANGXIN_HDR_GAMMA=2.6 

# 5. Run DCP-o-matic CLI (Single threaded recommended for inference stability)
./build/src/tools/dcpomatic2_create -o MyHDRProject input.mov
./build/src/tools/dcpomatic2_cli -t 1 MyHDRProject
```

## ⚙️ Configuration (Environment Variables)

The pipeline is entirely controlled via environment variables.

| Variable | Values | Default | Description |
| :--- | :--- | :--- | :--- |
| `ZHANGXIN_HDR_ENABLE` | `1` | Unset | **Required**. Activates the HDR processing. If unset, runs standard SDR pipeline. |
| `ZHANGXIN_HDR_MODEL` | File Path | Unset | **Required**. Path to the `.onnx` model file. Process aborts if missing. |
| `ZHANGXIN_HDR_GAMMA` | `2.4`, `2.6`, `REC709` | `2.4` | **Critical**. Defines how the SDR input RGB values are decoded before entering the neural network. See guide below. |
| `ZHANGXIN_HDR_DEBUG` | `1` | Unset | Detailed per-frame logs (Luminance stats, Hue shift, Semantics). |
| `ZHANGXIN_HDR_DUMP` | `1` | Unset | Dumps intermediate frames to `/tmp` for visual debugging. |

### 🎨 Transfer Function Guide (Input Gamma)

Choosing the correct `ZHANGXIN_HDR_GAMMA` is vital for correct brightness and shadow detail.

*   **`2.4` (Default, Recommended)**
    *   **Logic**: Pure Power 2.4. Approximates BT.1886 display reference.
    *   **Use Case**: Most mixed sources, ProRes masters, HD TV/Web content. 
    *   **Effect**: Balanced brightness. Prevents crushed shadows. Safest engineering choice.

*   **`2.6` (Cinema Mode)**
    *   **Logic**: Pure Power 2.6 (DCI Gamma).
    *   **Use Case**: Inputs that are strictly graded for DCI Cinema (dark rooms).
    *   **Effect**: Darker shadows. If used on Rec.709 content, it will crush shadow details into black.

*   **`REC709` (Scene Linear)**
    *   **Logic**: Rec.709 Inverse OETF (contains a linear segment near black).
    *   **Use Case**: Camera original footage or "Scene Linear" workflows.
    *   **Effect**: Lifts shadows significantly. Can be used if the default 2.4 feels too dark.

## 📊 Debugging & Interpretation

When `ZHANGXIN_HDR_DEBUG=1` is set, the CLI outputs frame statistics:

```text
[Frame 101] HDR(med=7.5 p99=10.5 max=105 nits)
[Frame 101] Hue(mean=3.1° p95=6.2°) Chroma(×1.9) ValidPx=1540624
[Frame 101] SDR(DD=28% Sh=70% Mid=1% Hi=0%) HDR(DD=28% Sh=71% Mid=0% Hi=0%)
```

*   **Hue (p95)**: 95th percentile hue shift. Should ideally be < 5°. High values indicate color skew.
*   **Chroma (x1.9)**: Ratio of HDR saturation to SDR saturation.
*   **Cinema Semantic Zones**:
    *   **DeepDark (DD)**: Black level floor.
    *   **Shadow (Sh)**: Textures and dark details.
    *   **Mid / Hi**: Midtones and Highlights.
    *   *Check*: If `SDR DD` is very high (>80%), your input might be crushed (try checking Gamma settings).

## 🛠️ Build & Patching

Requires `libdcp` patching for metadata support.

```bash
cd deps/libdcp
git apply ../../libdcp_hdr_integration.patch
# Rebuild dcpomatic:
# ./waf configure && ./waf build -j4
```

---
**Author**: zhangxin
**Last Updated**: 2026-01-11 (v2.0 Transfer Function Update)
