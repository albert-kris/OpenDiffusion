# OpenDiffusion

课题组扩散模型基础代码库 | A foundational diffusion model codebase for research and development.

## 📖 简介

OpenDiffusion 是基于 PyTorch 实现的扩散模型框架，为课题组提供**模块化、易扩展**的代码基础。支持 2D 图像和 1D 特征的扩散建模，涵盖 DDPM / DDIM 采样、多种噪声调度策略、多种预测目标，以及混合精度训练等功能。

### 主要特性

- 🔄 **多种噪声调度**：支持 Linear、Cosine、Sigmoid、Sqrt 四种 beta schedule
- 🎯 **多种预测目标**：支持 `pred_noise`（预测噪声）、`pred_x0`（预测原始数据）、`pred_v`（v-prediction）
- ⚡ **双采样器**：同时支持 DDPM 和 DDIM 采样，DDIM 可加速推理
- 🧩 **多架构支持**：内置 2D UNet（含注意力机制 / 交叉注意力）和两种 1D UNet 变体
- 📉 **Min-SNR 加权**：实现 [Min-SNR loss weighting](https://arxiv.org/abs/2303.09556) 策略
- 🔁 **自条件机制**：支持 Self-Conditioning 提升生成质量
- ⚡ **混合精度训练**：内置 FP16 训练工具，节省显存
- 📊 **灵活日志**：支持 stdout / CSV / JSON / TensorBoard 多种日志格式

## 📁 项目结构

```
OpenDiffusion/
├── zhou_diffusion/            # 核心代码包
│   ├── __init__.py
│   ├── diffusion_model.py     # 扩散过程核心（前向加噪、DDPM/DDIM去噪、损失计算）
│   ├── unet.py                # 2D UNet（含 ResBlock、AttentionBlock、CrossAttention）
│   ├── unet_1d.py             # 1D UNet（MLP-based，适用于 ViT 特征的扩散/聚类）
│   ├── unet1d.py              # 1D UNet（Encoder-Decoder 结构，带 skip connection）
│   ├── nn.py                  # 通用工具（timestep embedding、GroupNorm、EMA 更新等）
│   ├── fp16_util.py           # FP16 混合精度训练工具
│   ├── resample.py            # 时间步采样策略（均匀采样、Loss-aware 采样）
│   ├── uniform.py             # 权重初始化方法集合（Xavier、Kaiming、Orthogonal 等）
│   └── logger.py              # 多格式日志系统
├── setup.py                   # 安装配置
├── .gitignore
├── LICENSE                    # MIT License
└── README.md
```

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/albert-kris/OpenDiffusion.git
cd OpenDiffusion
```

### 2. 创建环境

```bash
conda create -n opendiffusion python=3.10
conda activate opendiffusion
```

### 3. 安装依赖

```bash
pip install torch torchvision torchaudio
pip install numpy tqdm einops
```

### 4. 安装本包（必须）

```bash
pip install -e .
```

> ⚠️ **这一步不能跳过！** 执行后才能在任意位置通过 `from zhou_diffusion.xxx import xxx` 导入模块。每人只需运行一次。

### 验证安装

```python
from zhou_diffusion.diffusion_model import diffusion
from zhou_diffusion.unet import UNetModel
print("安装成功 ✅")
```

## 📝 使用示例

### 2D 图像扩散

```python
import torch
from zhou_diffusion.unet import UNetModel
from zhou_diffusion.diffusion_model import diffusion

# 1. 定义 UNet
model = UNetModel(
    in_channels=3,
    model_channels=128,
    out_channels=3,
    num_res_blocks=2,
    attention_resolutions=[16, 8],
    channel_mult=(1, 2, 4),
    dims=2,
)

# 2. 创建扩散模型
diff = diffusion(
    eps_model=model,
    timesteps=1000,
    schedule='cosine',          # 'linear' | 'cosine' | 'sigmoid' | 'sqrt'
    objective='pred_noise',     # 'pred_noise' | 'pred_x0' | 'pred_v'
    min_snr_loss_weight=True,
)

# 3. 训练
x = torch.randn(8, 3, 64, 64)          # 输入图像 batch
t = torch.randint(0, 1000, (8,))        # 随机时间步
loss = diff.loss(x, t)
loss.backward()

# 4. 采样
samples = diff.denoise_loop_ddim(
    shape=(4, 3, 64, 64),
    number=50,       # DDIM 步数
)
```

### 1D 特征扩散（适用于 ViT 特征 / 聚类任务）

```python
import torch
from zhou_diffusion.unet1d import UNet1D
from zhou_diffusion.diffusion_model import diffusion

model = UNet1D(in_dim=384, embed_dim=384, self_condition=False)
diff = diffusion(eps_model=model, timesteps=1000, schedule='cosine')

features = torch.randn(32, 384)         # ViT 提取的特征
t = torch.randint(0, 1000, (32,))
loss = diff.loss(features, t)
```

## 🏗️ 核心模块说明

### `diffusion_model.py` — 扩散过程

| 方法 | 说明 |
|------|------|
| `add_noise(x_start, t)` | 前向扩散：对原始数据加噪 |
| `loss(x, t)` | 计算训练损失（支持一致性训练） |
| `denoise_loop_ddpm(shape)` | DDPM 采样（完整 T 步） |
| `denoise_loop_ddim(shape, number)` | DDIM 加速采样 |
| `cluster(x, model, ...)` | 基于扩散过程的聚类 |
| `get_z(x, t)` | 提取中间特征表示 |

### `unet.py` — 2D UNet

标准 U-Net 架构，支持多尺度特征、残差连接、自注意力和交叉注意力机制。核心组件包括 `ResBlock`、`AttentionBlock`、`CrossAttention`、`Upsample` / `Downsample`。

### `unet_1d.py` / `unet1d.py` — 1D UNet

两种面向 1D 特征的网络变体：

- **`unet_1d.py`**：基于 MLP 的轻量级结构，适用于 ViT 特征的扩散聚类任务
- **`unet1d.py`**：完整的 Encoder-Decoder 结构（含 skip connection），适用于需要多层次特征的任务

## 📝 开发规范

组内成员在此代码库上开发时，请遵守以下规范：

1. **新功能新建分支** — `git checkout -b feature/你的功能名`
2. **提交信息规范** — 使用 `feat:` / `fix:` / `docs:` / `refactor:` 前缀
3. **核心改动写注释** — 算法改动和非直觉实现请写清楚注释
4. **合并前提 PR** — 保持 `main` 分支稳定可用

## 📚 参考文献

- Ho et al. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPM), NeurIPS 2020
- Nichol & Dhariwal. [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672), ICML 2021
- Song et al. [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (DDIM), ICLR 2021
- Hang et al. [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556), CVPR 2024

## 📄 License

MIT License © haidong Hu
