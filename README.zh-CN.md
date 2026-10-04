# 论文与推理流程

本项目使用 LDM 公开预训练权重，文本编码器、U-Net、KL 自编码器和 DDIM 采样器均在本地实现。安装与运行见 [README](README.md)，理论背景见 [LDM 论文](https://arxiv.org/abs/2112.10752)。

## 推理流程

以下对应 [generate.py](generate.py) 的默认配置：**单张 256×256 图像、50 步、guidance=5、eta=0**。图像特征形状按 `[batch, channels, height, width]` 记。

### 整体步骤

**文本编码 + 初始噪声 →（U-Net → CFG → DDIM）× 50 → 解码 → PNG**

1. **编码文本。** 把 `["", prompt]`（空文本在前）一起转成 `[2, 77]` 的 token IDs，经文本编码器得到 `context [2, 77, 1280]`。
2. **初始化潜变量。** 在 CPU 上用 `seed` 固定随机生成器，从标准正态分布采样 `latents [1, 4, 32, 32]`，再移到所选设备。
3. **设置时间步。** `scheduler.set_timesteps(50)` 生成 `[981, 961, …, 21, 1]`，按此顺序去噪。
4. **循环去噪。** 每个时间步执行下表中的三步，始终维护一份 `[1, 4, 32, 32]` 的 `latents`。
5. **解码并保存。** `autoencoder.decode(latents / 0.18215)` 输出 `[1, 3, 256, 256]`；经 `(decoded / 2 + 0.5).clamp(0, 1)` 映射并裁剪到 `[0, 1]`，保存为 PNG。

| 单步操作 | 核心逻辑 | 输出形状 |
| --- | --- | --- |
| U-Net 预测噪声 | 拼接两份 `latents`，连同时间步和 `context` 输入 U-Net | `[2, 4, 32, 32]` |
| CFG 合成 | 按 batch 拆出空文本与提示词两路预测，按下式合成 `noise` | `[1, 4, 32, 32]` |
| DDIM 更新 | `scheduler.step(...)` 更新 `latents`；默认 `eta=0`，不额外加噪 | `[1, 4, 32, 32]` |

```python
noise = noise_empty + guidance * (noise_prompt - noise_empty)
```

> `guidance=1` 时跳过空文本编码、batch 拼接和 CFG 合成，U-Net 直接使用条件预测，batch 为 1。下文的 batch=2 来自 CFG 的两路预测，最终仍只生成一张图。

### U-Net 内部：一次噪声预测

对照 [ConditionalUNet.forward](models/unet.py)，先看三个输入：

| 输入 | 形状与用途 |
| --- | --- |
| `latents` | `[2, 4, 32, 32]`，两份相同的带噪潜变量 |
| `timestep` | 当前时间步扩展到 batch=2；正余弦编码得到 `[2, 320]`，再经 `Linear → SiLU → Linear` 得到 `temb [2, 1280]` |
| `context` | `[2, 77, 1280]`，为 cross-attention 提供文本条件 |

特征图依次经过下列模块。**输出形状均指整个模块执行完毕后的 `hidden`。** 配置中的 `CrossAttnDownBlock2D` / `CrossAttnUpBlock2D`，分别由本地 `DownBlock2D` / `UpBlock2D` 开启注意力实现。

| 阶段 | 模块 / 操作 | 输出形状 |
| --- | --- | --- |
| 输入 | `conv_in` | `[2, 320, 32, 32]` |
| 下行 1 | `DownBlock2D`：注意力、下采样 | `[2, 320, 16, 16]` |
| 下行 2 | `DownBlock2D`：注意力、下采样 | `[2, 640, 8, 8]` |
| 下行 3 | `DownBlock2D`：注意力、下采样 | `[2, 1280, 4, 4]` |
| 下行 4 | `DownBlock2D`，不下采样 | `[2, 1280, 4, 4]` |
| 瓶颈 | `ResNet → SpatialTransformer → ResNet` | `[2, 1280, 4, 4]` |
| 上行 1 | `UpBlock2D`（含上采样） | `[2, 1280, 8, 8]` |
| 上行 2 | `UpBlock2D`：注意力、上采样 | `[2, 1280, 16, 16]` |
| 上行 3 | `UpBlock2D`：注意力、上采样 | `[2, 640, 32, 32]` |
| 上行 4 | `UpBlock2D`：注意力，不上采样 | `[2, 320, 32, 32]` |
| 输出 | `GroupNorm → SiLU → conv_out` | `[2, 4, 32, 32]` |

- **时间与文本条件：** ResNet 注入 `temb`，可改变通道数；带注意力的块再经 `SpatialTransformer`（self-attention → cross-attention → 前馈网络）融合文本，其输入、输出形状相同。
- **跳跃连接（skip connection）：** 保存输入卷积和下行各层的特征；上行每个 ResNet 先沿通道维拼接对应特征，再处理。每个下行块有 2 个 ResNet，每个上行块有 3 个。
- **采样位置：** 前三个下行块在末尾下采样，前三个上行块在末尾上采样；两侧最后一个块都保持空间尺寸。

## 原理与实现

本节用一个数值例子说明推理的数学过程。例子只跟踪潜变量中的一个数。实际的潜变量有 4×32×32 = 4096 个数，每个数都用同样的公式计算。

- $\bar\alpha_t$ 的值来自本项目 scheduler 的配置，是真实值。
- U-Net 的输出是示例值，不是模型的真实输出。

### 符号

| 符号 | 名称 | 含义 |
| --- | --- | --- |
| $t$ | 时间步 | 训练时取 0 到 999。t 越大，噪声越多。 |
| $z_t$ | 潜变量 | 时间步 t 的带噪潜变量，即代码中的 `latents`。 |
| $z_0$ | 干净潜变量 | 没有噪声的潜变量，对应一张图像。 |
| $\epsilon$ | 噪声 | 标准正态分布的随机数，$\epsilon\sim\mathcal N(0,I)$。 |
| $\bar\alpha_t$ | 信号保留比例 | 时间步 t 的潜变量中保留的信号比例。 |
| $c$ | 文本条件 | 文本编码器的输出，即代码中的 `context`。 |
| $\hat\epsilon$ | 噪声预测 | U-Net 对 $\epsilon$ 的预测值。 |
| $\hat z_0$ | 干净潜变量估计 | 用 $\hat\epsilon$ 计算出的 $z_0$。 |

### 1. 为什么在潜空间做扩散

像素空间扩散每一步都处理完整的 RGB 图像。很多计算用于不影响视觉语义的细节。LDM 先用自编码器压缩图像，再在压缩后的潜空间做扩散：

$$z=\mathcal E(x),\qquad \hat x=\mathcal D(z)$$

- 论文分两个阶段训练：先训练自编码器，再固定自编码器，训练潜空间的扩散模型。
- 本项目使用 KL 正则化、下采样倍率为 8 的自编码器。所有权重都是预训练权重。
- 256×256×3 的图像对应 32×32×4 的潜变量。元素数量从 196608 减少到 4096，即 1/48。
- 实际耗时不一定减少到 1/48。

压缩倍率是质量与计算量的折中。压缩太弱，节省的计算量少。压缩太强，图像会丢失细节。

### 2. 加噪公式

扩散的全部计算都基于这个公式：

$$z_t=\sqrt{\bar\alpha_t}\,z_0+\sqrt{1-\bar\alpha_t}\,\epsilon$$

$\sqrt{\bar\alpha_t}$ 是信号的系数，$\sqrt{1-\bar\alpha_t}$ 是噪声的系数。两个系数的平方和等于 1。因此，在所有时间步，$z_t$ 的方差都约等于 1。

$\bar\alpha_t$ 是累乘积：

$$\bar\alpha_t=\prod_{s=0}^{t}(1-\beta_s)$$

scheduler 先在 $\sqrt{0.00085}$ 与 $\sqrt{0.012}$ 之间等距取 1000 个值，再把每个值平方，得到 $\beta_0,\dots,\beta_{999}$。本项目中几个时间步的系数如下：

| $t$ | $\bar\alpha_t$ | $\sqrt{\bar\alpha_t}$（信号） | $\sqrt{1-\bar\alpha_t}$（噪声） |
| --- | --- | --- | --- |
| 981 | 0.0058 | 0.076 | 0.997 |
| 961 | 0.0073 | 0.085 | 0.996 |
| 761 | 0.0514 | 0.227 | 0.974 |
| 21 | 0.9804 | 0.990 | 0.140 |
| 1 | 0.9983 | 0.999 | 0.041 |

**示例：** 设一个位置的干净值 $z_0=1.0$，噪声 $\epsilon=0.8$。在三个时间步加噪：

$$
\begin{aligned}
z_{761}&=0.227\times1.0+0.974\times0.8=1.006\\
z_{21}&=0.990\times1.0+0.140\times0.8=1.102\\
z_{981}&=0.076\times1.0+0.997\times0.8=0.874
\end{aligned}
$$

在 t = 981，$z_t$ 中的信号只占 7.6%。在 t = 21，信号占 99%。

### 3. U-Net 的训练目标

训练时，模型执行以下步骤：

1. 自编码器把一张真实图像编码为 $z_0$。
2. 模型随机选择一个时间步 t 和一个噪声 $\epsilon$。
3. 模型用加噪公式计算 $z_t$。
4. U-Net 接收 $z_t$、$t$ 和 $c$，输出噪声预测。
5. 损失函数比较噪声预测与真实噪声：

$$\mathcal L=\mathbb E\left[\|\epsilon-\epsilon_\theta(z_t,t,c)\|_2^2\right]$$

在上面的示例中，U-Net 的正确输出是 0.8。U-Net 预测噪声，不直接预测图像。

U-Net 需要输入 t。原因：同一个 $z_t$ 值在不同时间步中的噪声比例不同。在 t = 981，$z_t$ 主要是噪声。在 t = 21，$z_t$ 主要是信号。

这里的 $z_0$ 是缩放后的潜变量。训练时，自编码器的输出乘以 0.18215，使方差约等于 1。因此，解码前要用 `latents / 0.18215` 还原尺度。推理时没有反向传播，也不计算损失。

### 4. 推理的一步：t = 981 → 961

推理从纯噪声开始。设 `torch.randn` 在这个位置生成 $z_{981}=0.874$。这个值与第 2 节的示例相同，因此正确答案已知：$z_0=1.0$，$\epsilon=0.8$。

每一步有四个操作。

**① U-Net 预测两份噪声**（[generate.py:128](generate.py#L128)）

代码把同一份潜变量复制为 batch = 2。第一份使用空提示，第二份使用提示词：

$$
\begin{aligned}
\epsilon_u&=\epsilon_\theta(z_{981},\,981,\,c_{\text{empty}})=0.70\\
\epsilon_c&=\epsilon_\theta(z_{981},\,981,\,c_{\text{prompt}})=0.72
\end{aligned}
$$

**② CFG 合成噪声预测**（[generate.py:133](generate.py#L133)）

$$\hat\epsilon=\epsilon_u+s\,(\epsilon_c-\epsilon_u)=0.70+5\times(0.72-0.70)=0.80$$

$s$ 是 `guidance`。$\epsilon_c-\epsilon_u$ 是文本使噪声预测改变的方向，$s$ 放大这个方向。

- $s=1$：只使用 $\epsilon_c$。
- $s=0$：只使用 $\epsilon_u$，结果与文本无关。
- $s$ 增大：文本的影响增强。多样性可能降低，颜色可能过饱和。

在真实运行中，$|\epsilon_c-\epsilon_u|$ 的平均值约为 0.01。因此需要放大。

**③ 计算干净潜变量估计**（[scheduler.py](models/scheduler.py) 的 `step()`）

把加噪公式变形，求解 $z_0$：

$$\hat z_0=\frac{z_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}=\frac{0.874-0.997\times0.80}{0.076}=1.0$$

结果与正确答案 $z_0=1.0$ 相同。

**④ 用下一个时间步的系数重新混合**（[scheduler.py](models/scheduler.py) 的 `step()`）

下一个时间步 p = 961。DDIM 用 $\hat z_0$ 和 $\hat\epsilon$ 计算 $z_p$：

$$z_p=\sqrt{\bar\alpha_p}\,\hat z_0+\sqrt{1-\bar\alpha_p-\sigma_t^2}\,\hat\epsilon+\sigma_t\,\xi$$

$$\sigma_t=\eta\sqrt{\frac{1-\bar\alpha_p}{1-\bar\alpha_t}\left(1-\frac{\bar\alpha_t}{\bar\alpha_p}\right)}$$

默认 `eta = 0`，因此 $\sigma_t=0$，随机项 $\sigma_t\xi$ 消失：

$$z_{961}=0.085\times1.0+0.996\times0.80=0.882$$

`eta > 0` 时，每一步都采样 $\xi\sim\mathcal N(0,I)$。初始潜变量和 $\xi$ 都来自同一个使用固定 seed 的生成器。

**要点：** 第 ④ 步的公式就是第 2 节的加噪公式。区别有两个：

- 用 $\hat z_0$ 和 $\hat\epsilon$ 代替真实的 $z_0$ 和 $\epsilon$。
- 用下一个时间步的系数。下一个时间步的噪声更少。

信号系数从 0.076 增加到 0.085。50 步之后，信号系数增加到约 0.999。

### 5. 预测误差在不同时间步的影响

设 U-Net 的噪声预测有误差 $\Delta\epsilon$。由第 ③ 步的公式，$\hat z_0$ 的误差是：

$$\Delta\hat z_0=\frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}\,\Delta\epsilon$$

**示例：** 设 $\hat\epsilon=0.78$，误差为 0.02。

| 时间步 | 放大倍数 $\sqrt{1-\bar\alpha_t}/\sqrt{\bar\alpha_t}$ | $\hat z_0$ | $\hat z_0$ 的误差 |
| --- | --- | --- | --- |
| t = 981 | 0.997 / 0.076 ≈ 13 | 1.26 | 0.26 |
| t = 21 | 0.140 / 0.990 ≈ 0.14 | 1.003 | 0.003 |

在 t = 981，误差放大约 13 倍。因此，前几步解码 $\hat z_0$ 得到的图像只有模糊的色块。

$z_p$ 受到的影响小得多。在 t = 981 → 961，$\hat z_0$ 在 $z_p$ 中的系数只有 0.085：

$$z_{961}=0.085\times1.26+0.996\times0.78=0.885$$

正确值是 0.882，误差只有 0.003。前期的误差对潜变量的影响小，后续的步骤可以纠正它。后期的放大倍数小，所以细节在后期确定。

### 6. U-Net 怎样计算噪声预测

U-Net 的三个输入进入不同的位置。

**时间步 t：** 正余弦编码把 t 转换为 320 维向量。`Linear → SiLU → Linear` 再把它转换为 `temb [1280]`。每个 ResNet 把 `temb` 经过线性层后加到特征上：

$$h=\mathrm{conv}_1(x)+W\,\mathrm{temb}$$

每个通道加一个常数。同一通道的所有空间位置加相同的值。U-Net 用这个值判断当前的噪声强度。

**文本条件 c：** cross-attention 的 query 来自图像特征 $\varphi(z_t)$，key 和 value 来自文本特征：

$$Q=W_Q\,\varphi(z_t),\quad K=W_K\,c,\quad V=W_V\,c$$

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(QK^\top/\sqrt d\right)V$$

每个空间位置是一个 query。77 个文本 token 是 key 和 value。因此，不同的空间位置读取不同的文本信息。

**示例：** 一个空间位置与 3 个 token 的分数是 2.0、0.5、0.1：

| token | 分数 | $e^{\text{分数}}$ | softmax 权重 |
| --- | --- | --- | --- |
| fox | 2.0 | 7.39 | 0.73 |
| snowy | 0.5 | 1.65 | 0.16 |
| [PAD] | 0.1 | 1.11 | 0.11 |

输出是 $0.73\,V_{\text{fox}}+0.16\,V_{\text{snowy}}+0.11\,V_{\text{[PAD]}}$。这个位置主要读取 “fox” 的信息。

文本编码器（LDMBert）与 LDM 一起训练，tokenizer 是 BERT 的 WordPiece 词表。不能把文本编码器换成其他 BERT 或 CLIP 权重。论文对空间对齐的条件（例如超分辨率、修复、语义图）使用拼接。这些任务需要各自的模型，不能只改本项目的参数。

**下采样、上采样与跳跃连接：**

- 下行路径把分辨率从 32×32 降到 4×4。每个位置的感受野变大，U-Net 可以判断整体构图。
- 上行路径把分辨率从 4×4 升回 32×32。跳跃连接把下行路径的高分辨率特征拼接回来，补充细节。
- `conv_out` 输出 `[4, 32, 32]`。这是每个位置的噪声预测。

### 7. 完整流程

训练使用 1000 个时间步。推理不需要调用 U-Net 1000 次。DDIM 从中选出 50 个时间步：981, 961, …, 21, 1。每一步跨越 20 个训练时间步。DDPM 也可以减少步数，但必须重新计算跳步的转移参数。

```text
z = randn([1, 4, 32, 32])                       # t = 981，纯噪声
for t in [981, 961, ..., 21, 1]:
    ε_u, ε_c = U-Net([z, z], t, context)         # generate.py:128
    ε̂  = ε_u + s·(ε_c − ε_u)                      # generate.py:133
    ẑ₀ = (z − √(1−ᾱ_t)·ε̂) / √ᾱ_t                  # scheduler.step()
    z  = √ᾱ_p·ẑ₀ + √(1−ᾱ_p)·ε̂                     # scheduler.step()，eta = 0
image = decoder(z / 0.18215)                     # generate.py:138
```

最后一步从 t = 1 更新时，下一个时间步小于 0。scheduler 在这一步使用 $\bar\alpha_0$。

[流程动画](https://impasto-lab.github.io/Latent-Diffusion/examples/ldm-pipeline.html) 的第 4 阶段显示每一步的真实系数。第 1 步，$\sqrt{1-\bar\alpha_t}=0.997$，$\sqrt{\bar\alpha_t}=0.076$。最后一步，$\sqrt{1-\bar\alpha_t}=0.041$，$\sqrt{\bar\alpha_t}=0.999$。
