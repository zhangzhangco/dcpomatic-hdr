# DCP-o-matic HDR Neural Pipeline
# 智能 HDR 电影打包系统

A specialized fork of [DCP-o-matic](https://dcpomatic.com/) embedding a **Neural SDR-to-HDR Conversion Pipeline**.
本项目基于 [DCP-o-matic](https://dcpomatic.com/) 深度定制，嵌入了 **SDR-to-HDR 神经网络转换流水线**。

> **Goal**: Provide a low-cost, high-efficiency solution for creating DCI-compliant HDR Master DCPs.
> **目标**：为影院级 HDR 母版制作提供低成本、高效率的智能化解决方案。

### ❓ Why this fork? / 为什么选择此版本？

**For Creators / 电影制作人**: 
Bridge the gap between SDR budget and HDR quality. You don't need expensive color grading suites or dedicated HDR monitors; our Neural Engine handles the luminance reconstruction for you.
缩短 SDR 预算与 HDR 质量之间的差距。无需昂贵的调色软件或专业 HDR 监视器，内置的神经网络引擎将自动为您完成亮度重建。

**For Engineers / 技术人员**: 
Zero-config deployment. The integration of native libdcp patches ensures that the generated HDR MXF is fully compliant with DCI standards, passing digital cinema server validations with ease.
零配置部署。底层 libdcp 补丁确保生成的 HDR MXF 完全符合 DCI 标准，轻松通过各类影院服务器的合规性校验。

---

## 🚀 Key Features / 核心特性

- **Zero Config / 零配置**
  - Automates model loading; no complex Python environment setup required.
  - 自动加载预训练模型，无需繁琐的环境部署。

- **GPU Acceleration / GPU 加速**
  - Deep integration with ONNX Runtime (CUDA). **20x faster** than CPU.
  - 深度集成 ONNX Runtime，利用 GPU 张量核心大幅提升处理效率 (提速 20+ 倍)。

- **High Fidelity / 高保真**
  - Implements **Hue Lock** to ensure zero color shift during dynamic range expansion.
  - 强制启用色相锁定 (Hue Lock) 技术，确保亮度动态扩展时色彩零偏移。

- **Standard Compliance / 标准兼容**
  - Automates **SMPTE ST 2084 (PQ)** curve and Rec.2020 color space transformation.
  - 自动应用 ST 2084 (PQ) 光电转换曲线，生成符合 DCI 标准的 HDR DCP。

- **HDR Packaging Logic / HDR 打包支持**
  - Includes a custom **libdcp patch** to enable native HDR metadata (ST 2084) in MXF wrapping.
  - 包含自定义 **libdcp 补丁**，使 MXF 封装层级原生支持 HDR 元数据 (ST 2084) 注入。

---

## 🛠️ Usage Guide / 使用指南

### 1. Enable Neural Engine / 开启 HDR 引擎

1.  Launch **DCP-o-matic 2**.
    - *启动程序。*
2.  Go to **Edit** -> **Preferences**.
    - *进入菜单 **Edit** -> **Preferences**。*
3.  Select the **Neural HDR** tab.
    - *切换至 **Neural HDR** 标签页。*
4.  Check **Enable Neural HDR Processing**.
    - *勾选 **Enable Neural HDR Processing**。*

> ✅ **Status Check**: Verify **"Neural Engine: Ready (GPU)"** in the bottom status bar.
> *状态确认：底部状态栏应显示 "Neural Engine: Ready"*。

### 2. Make HDR DCP / 制作 HDR DCP

1.  **Create Project**: Start a new Film project as usual.
    - *新建项目：创建一个 DCP 项目。*
2.  **Import Media**: Drag & drop your SDR video source (Rec.709).
    - *导入素材：直接拖入您的 SDR 视频文件。*
3.  **Make DCP**: Click **Jobs** -> **Make DCP**.
    - *生成 DCP：点击 **Jobs** -> **Make DCP**。*

> 💡 **Note**: The preview window may appear dark or low-contrast. This is normal behavior for PQ signals on SDR screens.
> *注意：预览画面偏暗属于 PQ 信号在 SDR 屏幕上的正常特性。*

---

## ⚙️ Validation / 验证

You can verify that the output MXF contains the correct HDR metadata:
您可以使用以下命令验证生成的 MXF 文件是否包含正确的 HDR 元数据：

```bash
# 1. Set library path / 设置运行库路径
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/deps/onnxruntime/lib

# 2. Check Transfer Characteristic / 检查光电转换特性
./deps/asdcplib/src/.libs/asdcp-info -d <Your_DCP_Video_MXF> | grep "Transfer"

# Expected Output / 预期输出:
# Transfer Characteristic: SMPTE ST 2084
```

---

**Author**: zhangxin
**Model**: Neural HDR v4 (Opset 17)
**Base Project**: [dcpomatic.com](https://dcpomatic.com/)
