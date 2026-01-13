# DCP-o-matic SDR-to-HDR 神经网络流水线

集成基于神经网络的 SDR 到 HDR 上变换流水线，支持 GPU 加速，可创建创建符合 DCI 标准的 HDR DCP (PQ, ST 2084)。

## 🚀 快速开始

现在的 HDR 功能已完全集成，支持**模型自动下载**和**GPU 加速**，实现零配置开箱即用。

### 1. 启用功能 (一次性设置)

1. 打开 **Edit -> Preferences** (编辑 -> 偏好设置)。
2. 选择 **Neural HDR** (神经网络 HDR) 标签页。
3. **Enable Neural HDR Processing (启用神经网络 HDR 处理)**: 勾选此项。这将接管所有的视频处理流水线，将 SDR 信号转换为 HDR。
4. **Enable Hue Lock Strategy (启用色相锁定)**: 建议勾选。此选项强制 HDR 输出保持与 SDR 一致的色相，防止 AI 产生的色彩偏移。

**🤖 自动模型下载**：
首次启用并运行任务时，程序会自动检测并从 GitHub Release 下载所需的 `.onnx` 模型文件。无需手动配置路径。

### 2. 预览与制作

1. **预览**: 加载素材后，在预览窗口中看到的画面可能会偏灰暗（Dark），这是正常的。因为系统正在输出 PQ (ST 2084) 信号，而您的普通显示器无法正确解码 HDR。
2. **制作 DCP**: 点击 **Jobs -> Make DCP**。输出的 DCP 将自动包含正确的 HDR 元数据。

### 3. 硬件加速

程序内置了 **ONNX Runtime (CUDA)** 支持：
*   **GPU 模式**: 如果检测到 NVIDIA 显卡和 CUDA 环境，将自动使用 GPU 加速推理，大幅提升速度。
*   **CPU 回退**: 如果 GPU 不可用（如显存不足或驱动缺失），程序会自动无缝切换到 CPU 模式，保证任务不中断。

## ⚙️ 核心技术说明

### 色相锁定 (Hue Lock)
*   **原理**: 仅使用神经网络预测亮度增益 (Luminance Gain)，将色度信息 (Chromaticity) 锁定为原始 SDR 值。
*   **作用**: 彻底解决生成式模型可能产生的色偏或时间闪烁，确保 "原汁原味" 的色彩风格，仅扩展动态范围。

### Gamma 处理
*   流水线默认使用 **Gamma 2.4** 解码 SDR 输入，确保输入到神经网络的数据是线性光 (Linear Light)。

## 📊 调试与验证

在运行 CLI 工具 (`dcpomatic2_create` 或 `dcpomatic2_cli`) 时，如果开启了 Debug 模式，控制台会输出逐帧统计信息（如 Peak Nits, Hue Shift 等）。

## 🛠️ 构建与部署

本项目依赖 `onnxruntime` C++ 库。

1. **库文件**: 确保 `deps/onnxruntime/lib` 下包含 `libonnxruntime.so` (以及 GPU 版所需的 `_cuda.so`)。
2. **运行环境**:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/deps/onnxruntime/lib
   ```
3. **模型文件**: 硬编码为相对路径 `neural_hdr.onnx`。如果文件缺失，程序会尝试自动下载。

## ✅ 验证 HDR 合规性

验证生成的 DCP 是否包含 HDR (ST 2084 / PQ) 元数据，推荐使用 `asdcp-info`：

```bash
# 1. 设置库路径
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/deps/asdcplib/src/.libs

# 2. 运行检查
./deps/asdcplib/src/.libs/asdcp-info -d path/to/video.mxf | grep -A 5 "Transfer"
```
应看到 `Transfer Characteristic: SMPTE ST 2084`。

---
**Author**: zhangxin
**Last Updated**: 2026-01-13 (v4.0 Auto-Download & GPU)
