# DCP-o-matic HDR Integration

This document outlines the custom integration of a Neural Network-based SDR-to-HDR upscaling pipeline into DCP-o-matic, enabling the creation of DCI-compliant HDR DCPs.

## Overview

We have enhanced DCP-o-matic with a new processing pipeline that:
1.  **Intercepts** standard SDR video frames during the encoding process.
2.  **Upscales** them to HDR (targeting 300 nits peak luminance) using an ONNX-based neural network model.
3.  **Encodes** the result using the PQ (Perceptual Quantizer) curve (ST 2084).
4.  **Signals** the content as HDR in both the MXF and CPL metadata, ensuring compatibility with DCI HDR-capable cinema servers.

## Features

*   **ONNX Runtime Integration**: Loads and runs custom `.onnx` models for frame-by-frame inference.
*   **Strict Validation**: Requires a valid ONNX model path. If missing, the encoding process will terminate with an error to prevent invalid output.
*   **DCI Compliance**:
    *   **CPL Metadata**: Injects `<ExtensionMetadata>` with `EOTF=ST 2084` into the CPL.
    *   **MXF Metadata**: Sets the correct Transfer Characteristic UL (ST 2084) in the Picture Essence Descriptor.
    *   **Digital Signature**: All metadata is injected *before* signing, ensuring passing validation (Clairmeta check: Success).
*   **Performance Monitoring**: Logs real-time pixel statistics (Nit levels, PQ code values) for verification.

## Usage

### 1. Environment Setup

The HDR pipeline is controlled via environment variables.

| Variable | Value | Description |
| :--- | :--- | :--- |
| `ZHANGXIN_HDR_ENABLE` | `1` | **Required.** Activates the HDR processing path. If unset, DCP-o-matic behaves normally (SDR behavior). |
| `ZHANGXIN_HDR_MODEL` | `/path/to/model.onnx` | **Required.** Path to your trained ONNX model. If unset or invalid, the process will **FAIL**. |
| `ZHANGXIN_HDR_DEBUG` | `1` | Optional. Enables detailed per-frame statistics logging (min/max nits, histogram) to stdout. |
| `LD_LIBRARY_PATH` | `...` | **Required.** Must include: `build/src/lib`, `local_target/lib`, and `deps/onnxruntime/lib`. |

### 2. Making an HDR DCP (CLI Example)

You can use the standard DCP-o-matic CLI tools. No changes to CLI arguments are needed; simply set the environment variables.

```bash
# Set up environment
export ZHANGXIN_HDR_ENABLE=1
export ZHANGXIN_HDR_MODEL="/home/zhangxin/models/v6_latest.onnx"
# export ZHANGXIN_HDR_DEBUG=1  # Uncomment for pixel stats

# CRITICAL: Point to local libraries
export LD_LIBRARY_PATH=$(pwd)/build/src/lib:$(pwd)/build/src/wx:$(pwd)/local_target/lib:$(pwd)/deps/onnxruntime/lib:$LD_LIBRARY_PATH

# 1. Create the project
./build/src/tools/dcpomatic2_create \
  -o My_HDR_Project \
  --name "My HDR Movie" \
  input_video.mp4

# 2. Transcode (Make DCP)
# IMPORTANT: Use -t 1 (single thread) to ensure stable inference without OOM.
./build/src/tools/dcpomatic2_cli -t 1 My_HDR_Project
```

### 3. Verification

To verify the generated DCP is correctly signaled as HDR:

**Using Clairmeta (Recommended):**
```bash
python3 -m clairmeta.cli probe -type dcp My_HDR_Project/My_HDR_Movie_..._OV
# Check output for:
# - Transfer Characteristic in MXF (should indicate ST 2084 / PQ)
# - ExtensionMetadata in CPL (should present EOTF: ST 2084)
```

**Using asdcp_info (Low-level):**
```bash
# Check Picture MXF
asdcp-info -H My_HDR_Project/.../j2c_....mxf | grep "TransferCharacteristic"
# Should output UL ending in: ...04.01.01.01.01.0A.00.00 (ST 2084)
```

## System Integration Summary

This integration modifies the core `dcpomatic` and `libdcp` libraries to strictly enforce DCI HDR compliance:

1.  **Frame Processing**: Intercepts video frames in `dcp_video.cc` and routes them through the ONNX model (`zhangxin_hdr.cc`).
2.  **Metadata Injection**: 
    *   **MXF**: Patched `libdcp` writes SMPTE ST 2084 Universal Labels.
    *   **CPL**: `writer.cc` injects `<ExtensionMetadata>` before digital signing.
3.  **Validation**: All logic is hard-coded to fail-safe (abort) if model loading fails, preventing accidental SDR leakage.



---
**Author**: zhangxin
**Date**: 2026-01-11
