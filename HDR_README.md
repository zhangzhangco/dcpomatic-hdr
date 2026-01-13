# DCP-o-matic SDR-to-HDR 神经网络流水线

集成基于神经网络的 SDR 到 HDR 上变换流水线，支持创建符合 DCI 标准的 HDR DCP (PQ, ST 2084)。

## 🚀 快速开始

现在的 HDR 功能已完全集成到图形界面中，无需通过命令行环境变量配置。

### 1. 全局配置 (一次性设置)

首先配置模型路径和全局默认策略：
1. 打开 **Edit -> Preferences** (编辑 -> 偏好设置)。
2. 选择 **Neural HDR** (神经网络 HDR) 标签页。
3. **Model Path (模型路径)**: 选择你的 `.onnx` 模型文件 (例如 `v6_latest.onnx`)。
4. **Enable Hue Lock Strategy (启用色相锁定)**: 建议勾选。此选项强制 HDR 输出保持与 SDR 一致的色相，防止色彩偏移。

### 2. 项目启用 (每个项目单独设置)

针对你想要制作 HDR 版的具体项目：
1. 加载你的 SDR 视频素材。
2. 进入 **DCP** 标签页 -> **Video** (视频) 子标签页。
3. 勾选 **Enable Neural HDR Processing (启用神经网络 HDR 处理)**。
4. 此时，预览窗口 (Preview) 可能会显示 "Dark" 的画面，这是正常的，因为 HDR (PQ) 信号在 SDR 显示器上看起来会偏灰暗。

### 3. 创建 DCP

正常点击 **Jobs -> Make DCP** (任务 -> 制作 DCP)。
*   建议使用单线程或较少的编码线程以保证推理稳定性（可在 Preferences -> General 中设置线程数）。
*   输出的 DCP 将自动包含 PQ Transfer Function (ST 2084) 和相关的 HDR 元数据。

## ⚙️ 核心功能说明

###色相锁定 (Hue Lock Only For Pro)
*   **原理**: 仅使用神经网络预测亮度增益 (Luminance Gain)，而将色度信息 (Chromaticity) 锁定为原始 SDR 的值。
*   **作用**: 彻底解决 AI 模型（基础版）可能产生的色偏或时间闪烁问题，确保 "原汁原味" 的色彩风格，仅仅扩展动态范围。

### Gamma 处理
*   流水线内部默认使用 **Gamma 2.4** 解码 SDR 输入。这是最通用的标准，适用于大多数 ProRes 母版和高清素材。
*   程序会自动识别 DCP-o-matic 的色彩转换设置，确保输入到神经网络的数据是线性光 (Linear Light)。

## 📊 调试与验证

在运行 CLI 工具 (`dcpomatic2_create` 或 `dcpomatic2_cli`) 时，如果开启了 Debug 模式 (需源码级开启)，控制台会输出逐帧统计信息：

```text
[Frame 101] HDR(med=7.5 p99=10.5 max=105 nits)
[Frame 101] Hue(mean=3.1° p95=6.2°) Chroma(×1.9) ValidPx=1540624
```

*   **Hue (p95)**: 95% 像素的色相偏移量。启用 Hue Lock 后此值应接近 0。
*   **HDR Max**: 峰值亮度 (Nits)。

## 🛠️ 构建说明

本项目依赖 `onnxruntime` C++ 库。

1. 确保 `deps/onnxruntime` 存在且包含 `lib` 和 `include`。
2. 编译时需使用 C++17 标准 (已在 wscript 中配置)。
3. 运行时需确保 `libonnxruntime.so` 在库路径中：
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/deps/onnxruntime/lib
   ```

## ✅ 验证 HDR 合规性

验证生成的 DCP 是否包含正确的 HDR (ST 2084 / PQ) 元数据：

### 方法 1: 使用 asdcp-lib (推荐)

使用 `asdcp-info` 工具检查 MXF 文件头中的 **Transfer Characteristic UL** (标识符: `...04 01 01 01 01 02`)。

```bash
# 1. 设置库路径
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/deps/asdcplib/src/.libs

# 2. 运行检查 (替换为你的 MXF 路径)
./deps/asdcplib/src/.libs/asdcp-info -d path/to/j2c_video.mxf | grep -A 5 "Transfer"
```

如果看到类似 `Transfer Characteristic: SMPTE ST 2084` 或对应的 UL 值，则表示 HDR 元数据正确。

### 方法 2: 使用 ClairMeta

可以使用 Python 工具 ClairMeta 进行整体合规性检查（需安装 mediainfo）：

```bash
pip install clairmeta
python3 -m clairmeta.cli check path/to/dcp_folder -type dcp
```

---
**Author**: zhangxin
**Last Updated**: 2026-01-12 (v3.0 GUI Integration)
