# DCP-o-matic SDR-to-HDR Neural Pipeline

集成高效的神经网络流水线，支持 GPU 加速，可直接生成符合 DCI 标准的 HDR DCP (PQ, ST 2084)。

## 🚀 核心特性

- **零配置**: 自动下载模型，无需手动部署。
- **高性能**: 自动检测并使用 GPU (CUDA) 加速，支持无缝回退 CPU。
- **高保真**: 强制启用色相锁定 (Hue Lock)，确保色彩零偏移。
- **自动化**:内置 Gamma 2.4 解码流程，自动适配 SDR 输入格式。

## 🛠️ 使用指南

1. **全局开启**: 
   - 前往 **Edit -> Preferences -> Neural HDR**。
   - 勾选 **Enable Neural HDR Processing**。

2. **制作 DCP**:
   - 正常导入 SDR 视频。
   - 点击 **Jobs -> Make DCP**。
   - *注：预览窗口画面偏暗属正常现象（PQ 信号在 SDR 屏幕上的特性）。*

## ⚙️ 技术规格

- **模型**: Neural HDR v4 (Opset 17)，运行于 ONNX Runtime。
- **部署**: 依赖库位于 `deps/onnxruntime`，模型文件自动下载至同级目录。

## ✅ 验证 (可选)

检查输出 MXF 是否包含 SMPTE ST 2084 标识：

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/deps/onnxruntime/lib
./deps/asdcplib/src/.libs/asdcp-info -d path/to/video.mxf | grep "Transfer"
# 预期输出: Transfer Characteristic: SMPTE ST 2084
```

---
**Author**: zhangxin
**Last Updated**: 2026-01-13 (v4.0)
