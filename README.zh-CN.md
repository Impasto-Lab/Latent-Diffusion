# 论文与推理流程

本项目使用 LDM 公开预训练权重，文本编码器、U-Net、KL 自编码器和 DDIM 采样器均在本地实现。安装与运行见 [README](README.md)，理论背景见 [LDM 论文](https://arxiv.org/abs/2112.10752)。

## 推理流程

以下对应 [generate.py](generate.py) 的默认配置：**单张 256×256 图像、50 步、guidance=5、eta=0**。图像特征形状按 `[batch, channels, height, width]` 记。

### 整体步骤

**文本编码 + 初始噪声 →（U-Net → CFG → DDIM）× 50 → 解码 → PNG**

1. **编码文本。** 提示词和空文本分别转成 `[1, 77]` 的 token IDs，再经文本编码器得到 `context` 和 `unconditional`，各为 `[1, 77, 1280]`。按「空文本、提示词」顺序拼成 `model_context [2, 77, 1280]`。
2. **初始化潜变量。** 在 CPU 上用 `seed` 固定随机生成器，从标准正态分布采样 `latents [1, 4, 32, 32]`，再移到所选设备。
3. **设置时间步。** `scheduler.set_timesteps(50)` 生成 `[981, 961, …, 21, 1]`，按此顺序去噪。
4. **循环去噪。** 每个时间步执行下表中的三步，始终维护一份 `[1, 4, 32, 32]` 的 `latents`。
5. **解码并保存。** `autoencoder.decode(latents / 0.18215)` 输出 `[1, 3, 256, 256]`；经 `(decoded / 2 + 0.5).clamp(0, 1)` 映射并裁剪到 `[0, 1]`，保存为 PNG。

| 单步操作 | 核心逻辑 | 输出形状 |
| --- | --- | --- |
| U-Net 预测噪声 | 拼接两份 `latents`，连同时间步和 `model_context` 输入 U-Net | `[2, 4, 32, 32]` |
| CFG 合成 | 按 batch 拆出空文本与提示词两路预测，按下式合成 `noise` | `[1, 4, 32, 32]` |
| DDIM 更新 | `scheduler.step(...)` 更新 `latents`；默认 `eta=0`，不额外加噪 | `[1, 4, 32, 32]` |

```python
noise = noise_unconditional + guidance * (noise_conditional - noise_unconditional)
```

> `guidance=1` 时跳过空文本编码、batch 拼接和 CFG 合成，U-Net 直接使用条件预测，batch 为 1。下文的 batch=2 来自 CFG 的两路预测，最终仍只生成一张图。

### U-Net 内部：一次噪声预测

对照 [ConditionalUNet.forward](models/unet.py)，先看三个输入：

| 输入 | 形状与用途 |
| --- | --- |
| `sample` | `[2, 4, 32, 32]`，两份相同的带噪潜变量 |
| `timestep` | 当前时间步扩展到 batch=2；正余弦编码得到 `[2, 320]`，再经 `Linear → SiLU → Linear` 得到 `temb [2, 1280]` |
| `encoder_hidden_states` | `model_context [2, 77, 1280]`，为 cross-attention 提供文本条件 |

特征图依次经过下列模块。**输出形状均指整个模块执行完毕后的 `hidden`。**

| 阶段 | 模块 / 操作 | 输出形状 |
| --- | --- | --- |
| 输入 | `conv_in` | `[2, 320, 32, 32]` |
| 下行 1 | `CrossAttnDownBlock2D`（含下采样） | `[2, 320, 16, 16]` |
| 下行 2 | `CrossAttnDownBlock2D`（含下采样） | `[2, 640, 8, 8]` |
| 下行 3 | `CrossAttnDownBlock2D`（含下采样） | `[2, 1280, 4, 4]` |
| 下行 4 | `DownBlock2D`，不下采样 | `[2, 1280, 4, 4]` |
| 瓶颈 | `ResNet → SpatialTransformer → ResNet` | `[2, 1280, 4, 4]` |
| 上行 1 | `UpBlock2D`（含上采样） | `[2, 1280, 8, 8]` |
| 上行 2 | `CrossAttnUpBlock2D`（含上采样） | `[2, 1280, 16, 16]` |
| 上行 3 | `CrossAttnUpBlock2D`（含上采样） | `[2, 640, 32, 32]` |
| 上行 4 | `CrossAttnUpBlock2D`，不上采样 | `[2, 320, 32, 32]` |
| 输出 | `GroupNorm → SiLU → conv_out` | `[2, 4, 32, 32]` |

- **时间与文本条件：** ResNet 注入 `temb`，可改变通道数；带注意力的块再经 `SpatialTransformer`（self-attention → cross-attention → 前馈网络）融合文本，其输入、输出形状相同。
- **跳跃连接（skip connection）：** 保存输入卷积和下行各层的特征；上行每个 ResNet 先沿通道维拼接对应特征，再处理。每个下行块有 2 个 ResNet，每个上行块有 3 个。
- **采样位置：** 前三个下行块在末尾下采样，前三个上行块在末尾上采样；两侧最后一个块都保持空间尺寸。

## 原理与实现

### 1. 为什么把扩散过程放在潜空间

像素空间扩散反复处理整个 RGB 图像，很多计算花在不影响视觉语义的细节上。LDM 先学习感知压缩，把图像映射到保留空间结构的低维特征图，再在这个特征图上学习扩散模型。压缩太弱节省有限，太强又丢失细节，因此下采样倍率是质量与计算量的折中。

论文的训练分两阶段：先训练自编码器，再固定它、训练潜空间扩散模型。自编码器通过重建、感知与对抗目标保留视觉细节，并用 KL 或 VQ 约束潜表示。本项目采用的公开文生图模型属于 KL 正则化、下采样倍率 8 的版本。这里全部加载别人训练好的权重。

$$z=\mathcal E(x),\qquad \hat x=\mathcal D(z)$$

图像 `256×256×3` 对应潜表示 `32×32×4`。空间位置减少到 1/64；若连通道一起计数，元素数量从 196608 降至 4096，是 1/48。不能由此推断实际耗时刚好缩短 48 或 64 倍。

### 2. U-Net 学习什么

训练时，对真实图片编码后的潜变量 $z_0$ 加噪。这里 $z_0$ 指扩散模型使用的缩放后潜变量，本项目解码前用 `latents / 0.18215` 还原尺度：

$$z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon,\quad \epsilon\sim\mathcal N(0,I).$$

文本条件为 $c=\tau_\theta(y)$，噪声预测目标为：

$$\mathcal L=\mathbb E\left[\|\epsilon-\epsilon_\theta(z_t,t,c)\|_2^2\right].$$

时间步告诉 U-Net 当前噪声强度，文本条件告诉它要生成的内容。U-Net 的输出是噪声估计，DDIM 再利用这个估计更新潜变量。推理时没有反向传播，也不用计算这个训练损失。

### 3. 文本如何影响图像

论文原始模型使用 BERT tokenizer 和随模型学习的文本 Transformer，不应直接换成任意 BERT 或 CLIP 权重。Cross-attention 的 query 来自图像特征，key/value 来自文本特征：

$$Q=W_Q\varphi(z_t),\quad K=W_Kc,\quad V=W_Vc,$$

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^\top/\sqrt d)V.$$

这让图像的不同空间位置读取不同文本信息。论文对空间对齐条件也采用拼接，可用于超分辨率、修复与语义图生成；这些任务需要各自适配的条件与模型，不能仅改文生图脚本的参数完成。

### 4. CFG 与 DDIM 更新公式

CFG 将空条件与文本条件下的两份噪声预测组合起来：

$$\hat\epsilon=\epsilon_u+s(\epsilon_c-\epsilon_u).$$

这里 $s$ 是 `guidance`。$s=1$ 只保留条件预测；$s=0$ 对应无条件预测。增大 $s$ 往往增强文本影响，也可能降低多样性或产生过饱和。

训练时的 1000 个时间步定义噪声强度，并不要求推理调用 U-Net 1000 次。原始 DDPM 采样逐步更新；本项目使用 DDIM，从中选出 50 个时间步，每次按较大的跨度更新。DDPM 也可以通过重算跳步的转移参数减少采样步数，不能把原本逐步更新的公式直接用于跳步。

DDIM 将当前潜变量 $z_t$ 更新到较小的时间步 $z_p$。其中 $\bar\alpha_t$ 是从训练噪声日程累乘得到的保留信号比例。先由预测噪声估计干净潜变量：

$$\hat z_0=(z_t-\sqrt{1-\bar\alpha_t}\hat\epsilon)/\sqrt{\bar\alpha_t},$$

$$\sigma_t=\eta\sqrt{\frac{1-\bar\alpha_p}{1-\bar\alpha_t}\left(1-\frac{\bar\alpha_t}{\bar\alpha_p}\right)},$$

$$z_p=\sqrt{\bar\alpha_p}\hat z_0+\sqrt{1-\bar\alpha_p-\sigma_t^2}\hat\epsilon+\sigma_t\xi.$$

默认 `eta=0`，因此 $\sigma_t=0$，最后的随机项消失。`eta>0` 时，每一步还会采样 $\xi\sim\mathcal N(0,I)$；初始潜变量与这些附加噪声都由同一个固定 seed 的生成器控制。最后一步的目标时间步落到 0 以下时，本地 scheduler 使用 $\bar\alpha_0$ 完成更新。