# 潜空间扩散模型：本地文生图学习项目

[英文版](README.md) · [论文](https://arxiv.org/abs/2112.10752v2) · [预训练权重](https://huggingface.co/CompVis/ldm-text2im-large-256)

本项目用论文作者公开的 256×256 文生图权重，只运行推理。BERT 文本编码器、条件 U-Net、KL 自编码器及 DDIM 采样器都在本地显式定义，方便沿着代码观察数据流。Diffusers 不参与生成，只用于测试时做数值对照；本项目不训练模型。

## 环境与运行

在 Apple Silicon Mac 上，先进入本地 conda 环境：

```bash
cd Latent-Diffusions
conda activate diffusion
python -m pip install -r requirements.txt
python -m scripts.download_model
python generate.py --prompt "A red fox in a snowy forest, oil painting"
```

现有 `diffusion` 环境已安装支持 MPS 的 PyTorch。新环境需先安装适合该机器的 PyTorch。权重约 5.73 GiB，下载脚本会校验文件哈希；下载完成后生成过程离线运行。默认 PNG 保存在 `outputs/`。如需指定文件名，可加 `--output outputs/fox.png`。

在 NVIDIA 机器上，先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装支持该显卡的 CUDA 版 PyTorch，再安装依赖、下载权重并运行：

```bash
conda activate diffusion
python -m pip install -r requirements.txt
python -m scripts.download_model
python generate.py --device cuda --steps 2 --output outputs/cuda-smoke.png
```

`--device auto` 会按 CUDA → MPS → CPU 选择；RTX 5070 Ti 的真实推理仍需在目标机器上验证。Mac 上文本编码器和解码器留在 CPU，重复执行的 U-Net 放在 MPS；CUDA 上三个模型都放在显卡。这是固定的内存放置策略，没有额外的调参开关。

常用参数：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--prompt` | 红狐示例 | 文本条件 |
| `--steps` | 50 | DDIM 去噪次数 |
| `--guidance` | 5 | Classifier-free guidance 强度；1 只算条件分支 |
| `--eta` | 0 | DDIM 的额外随机噪声；0 为确定性更新 |
| `--seed` | 42 | 初始噪声及 `eta>0` 时的随机噪声 |
| `--width`, `--height` | 256 | 输出分辨率，要求 64 的倍数 |
| `--device` | auto | `cpu`、`mps`、`cuda` 或 `cuda:N` |
| `--dtype` | auto | GPU 默认 float16，CPU 默认 float32 |
| `--trace-shapes` | 关闭 | 打印主要模型阶段的张量形状，便于调试 |

16 GiB M1 Air 已实测原生 256×256、CFG batch=2 的完整推理。更大分辨率会增加 attention 的临时内存占用，建议先从模型原生尺寸理解流程。

## 项目结构与阅读入口

先看根目录的 `generate.py`：解析参数与选择设备 → 加载权重 → 采样潜变量 → 解码并保存图像。然后看 `inference/sampler.py` 中的文本编码、初始噪声、CFG 和 DDIM 循环；各模型的计算进入 `models/`。

```text
generate.py
├── inference/sampler.py     # 文本编码、CFG、DDIM、图像解码
├── models/bert.py           # 文本编码器
├── models/unet.py           # 条件去噪网络
├── models/layers.py         # ResNet、attention、上下采样
├── models/vqvae.py          # KL 自编码器
├── models/scheduler.py      # DDIM 时间步与更新公式
├── models/loader.py         # 本地模型与预训练权重绑定
└── utils/                   # 命令行、设备、输出和形状调试
```

主要数据路径为：文本 token → `[1,77,1280]` 条件表示；初始噪声 → `[1,4,32,32]` 潜变量；U-Net 逐步预测噪声；最终潜变量由自编码器解码为 `[1,3,256,256]` 图像。

## 模型实现与调试

`models/` 包含推理所需的全部 PyTorch 网络和 DDIM 公式。外部 checkpoint 只提供参数张量和配置。

建议按下面顺序阅读：

1. `loader.py`：读取 JSON 配置，在 `meta` 设备创建空模型，再用 `strict=True` 绑定预训练权重。
2. `components.py`：声明 tokenizer、文本编码器、U-Net、自编码器和 scheduler 之间的依赖关系。
3. `bert.py`：token embedding、位置 embedding、32 层 self-attention/FFN 和最终 LayerNorm。
4. `unet.py`：时间步 embedding、四个 down block、cross-attention bottleneck、四个 up block 和 skip connection。
5. `layers.py`：ResNet、上下采样、GEGLU、self/cross-attention 与空间 Transformer 的逐步张量运算。
6. `vqvae.py`：完整 image encoder、对角高斯分布以及本项目推理实际使用的 latent decoder。
7. `scheduler.py`：噪声曲线、DDIM 时间步以及从 $z_t$ 到 $z_{t-1}$ 的显式更新。

一次 256×256 文生图的主要形状为：

```text
token ids                     [1, 77]
text embeddings               [1, 77, 1280]
initial latent                [1, 4, 32, 32]
U-Net down features           320 → 640 → 1280 → 1280 channels
predicted noise               [1, 4, 32, 32]
decoded image                 [1, 3, 256, 256]
```

使用下面的命令可以只在每个主要模块第一次执行时打印输入输出形状：

```bash
python generate.py --steps 2 --trace-shapes \
  --output outputs/debug-shapes.png
```

`unet.up.*` 的 trace 会按 `pop()` 顺序列出当前 up block 消费的 skip 形状。调试 `ConditionalUNet.forward` 时，观察 `skip_features`：从输入卷积开始保存，四个 down block 逐级加入，上采样的 ResNet 层再逐个取出。

适合设置断点的位置：

- 模块实例化：`ConditionalUNet.__init__` 中的 `down_block = ...`、`up_block = ...`
- 下采样特征保存：`DownBlock2D.forward` 和 `CrossAttnDownBlock2D.forward` 中的 `new_skips.append`
- 文本 self-attention：`BertSelfAttention.forward`
- 图像读取文本条件：`CrossAttention.forward`，其中 `context` 是 BERT 输出
- U-Net skip connection：`UpBlock2D.forward` 中的 `torch.cat`
- 完整 U-Net 数据流：`ConditionalUNet.forward`
- 潜变量转 RGB：`Decoder.forward`
- 权重绑定：`_load_weights`

模型 checkpoint 的键名与本地模块路径一一对应。例如：

```text
down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight
```

可以直接解释为第一个 down block、第一个空间 Transformer、其中第一个 Transformer block 的文本 cross-attention key 投影。`model.named_parameters()` 会显示同样的路径，便于在调试器中从权重追到计算。

Transformers 只负责读取 BERT tokenizer 词表。Diffusers 只存在于测试中，用来产生参考输出；实际生成的神经网络前向和 DDIM 更新均由本目录的 PyTorch 代码执行。

本地 BERT、U-Net、VAE 解码器和 DDIM 的一步更新都直接返回 tensor：文本特征、预测噪声、RGB 张量和下一步 latent。调试时可直接查看返回值，不需要再从 `.sample`、`[0]` 或 `.prev_sample` 中取值。

下载的 JSON 保留原始内容以便核对权重来源；模型实例只保存参与当前 checkpoint 推理的配置字段。U-Net 的 cross-attention 宽度从 BERT 输出宽度补齐，VAE 的 latent 数值缩放因子由 `models/config.py` 显式给出。DDIM 只实现作者这份模型实际使用的噪声曲线和更新公式。

## 论文与推理流程

论文：[Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models, arXiv:2112.10752v2](https://arxiv.org/abs/2112.10752v2)。以下区分论文方法、作者公开权重和本项目的工程实现。

### 按代码顺序走一遍（默认 256×256、50 步、guidance=5）

方括号表示张量形状，第一维为 batch。

1. **编码条件。** `encode(prompt)` 和 `encode("")` 各将文本转成 `[1, 77]` 的 token IDs，再经文本编码器返回 `context [1, 77, 1280]` 和 `unconditional [1, 77, 1280]`。`guidance=1` 时跳过 `unconditional`。
2. **生成初始噪声。** 用 `seed` 从标准正态分布采样 `latents [1, 4, 32, 32]`；32×32 是 256×256 经自编码器压缩 8 倍后的空间尺寸。
3. **设置时间步。** `scheduler.set_timesteps(50)` 生成 `[981, 961, …, 21, 1]`，之后按此顺序去噪。
4. **U-Net 预测噪声。** 对照 `ConditionalUNet.forward(sample, timestep, encoder_hidden_states)` 看三个输入：

   **a. `timestep` → `temb`：** 当前时间步扩展到 batch=2，经过正余弦编码 `[2, 320]`，再经 `Linear → SiLU → Linear` 得到 `temb [2, 1280]`。

   **b. `sample` → `hidden`：** `sample = cat([latents, latents]) [2, 4, 32, 32]`。`conv_in` 输出 `hidden [2, 320, 32, 32]`；随后经过四个 down blocks、bottleneck、四个 up blocks，并通过 `skip_features` 连接对应层。

   **c. `encoder_hidden_states`：** `cat([unconditional, context]) [2, 77, 1280]`，与 b 的 `hidden` 一起进入 down blocks 0–2、bottleneck 和 up blocks 1–3；这些 block 内部就有 cross-attention。down block 3 和 up block 0 没有 cross-attention。

   **合流：** 在上述 block 内，`hidden` 先经过 ResNet，并加上投影后的 `temb`；紧接着进入 `SpatialTransformer`，先做图像 self-attention，再做图像与文本的 cross-attention：

   - `hidden [2, 320, 32, 32]` 展平为 1024 个图像 token：`[2, 1024, 320]`。
   - **Q 来自图像，K/V 来自 `encoder_hidden_states`**。第一层的 8 个 head 产生注意力分数 `[16, 1024, 77]`；对 77 个文本位置做 softmax，再加权 V，得到每个图像位置读取的文本信息。
   - 结果恢复为空间特征图，最终由 `conv_out` 输出噪声预测 `[2, 4, 32, 32]`。

5. **CFG 合成。** 将 U-Net 输出拆成 `noise_unconditional` 和 `noise_conditional`，各为 `[1, 4, 32, 32]`；计算 `noise = noise_unconditional + guidance × (noise_conditional − noise_unconditional)`。`guidance=1` 时直接使用条件预测。
6. **DDIM 更新。** `scheduler.step(noise, timestep, latents)` 得到下一时间步的 `latents [1, 4, 32, 32]`；默认 `eta=0`，不额外加噪。公式见下文「CFG 与 DDIM 更新公式」。
7. **重复步骤 4–6**，遍历全部 50 个时间步。
8. **解码。** `autoencoder.decode(latents / 0.18215)` 得到 `[1, 3, 256, 256]`；映射到 `[0, 1]` 后保存 PNG。

#### `new_skips`

`conv_in` 的输出先存入 `skip_features`。每个 down block 返回 `hidden`（传给下一个 block）和 `new_skips`（当前 block 保存的各层输出）；`skip_features.extend(new_skips)` 把后者逐个加入总列表。前三个 block 的 `new_skips` 各含两层 ResNet/attention 输出和一次下采样输出；最后一个 block 不下采样。

| down block | `new_skips` 中各项的形状（按保存顺序） | 返回给下一层的 `hidden` |
| --- | --- | --- |
| 0 | `[2,320,32,32]`、`[2,320,32,32]`、`[2,320,16,16]` | `[2,320,16,16]` |
| 1 | `[2,640,16,16]`、`[2,640,16,16]`、`[2,640,8,8]` | `[2,640,8,8]` |
| 2 | `[2,1280,8,8]`、`[2,1280,8,8]`、`[2,1280,4,4]` | `[2,1280,4,4]` |
| 3 | `[2,1280,4,4]`、`[2,1280,4,4]`；最后一个 block 不下采样 | `[2,1280,4,4]` |

`hidden` 与 `new_skips[-1]` 是同一张量。连同 `conv_in` 的输出，共有 `1 + 3 + 3 + 3 + 2 = 12` 份 skip；四个 up blocks 各调用 3 次 `skip_features.pop()`，按后进先出顺序取完。

### 1. 为什么把扩散过程放在潜空间

像素空间扩散反复处理整个 RGB 图像，很多计算花在不影响视觉语义的细节上。LDM 先学习感知压缩，把图像映射到保留空间结构的低维特征图，再在这个特征图上学习扩散模型。压缩太弱节省有限，太强又丢失细节，因此下采样倍率是质量与计算量的折中。

论文的训练分两阶段：先训练自编码器，再固定它、训练潜空间扩散模型。自编码器通过重建、感知与对抗目标保留视觉细节，并用 KL 或 VQ 约束潜表示。本项目采用的公开文生图模型属于 KL 正则化、下采样倍率 8 的版本。这里全部加载别人训练好的权重。

$$z=\mathcal E(x),\qquad \hat x=\mathcal D(z)$$

图像 `256×256×3` 对应潜表示 `32×32×4`。空间位置减少到 1/64；若连通道一起计数，元素数量从 196608 降至 4096，是 1/48。不能由此推断实际耗时刚好缩短 48 或 64 倍。

### 2. U-Net 学习什么

训练时，从编码后的真实图片潜变量加噪：

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

### 5. 论文、权重与本机设置的对应关系

| 概念 | 论文/作者实现 | 本项目 |
| --- | --- | --- |
| 感知压缩 | KL 自编码器，f=8 | `models/vqvae.py` 的本地 `AutoencoderKL` |
| 文本条件 | tokenizer + Transformer | tokenizer + `models/bert.py` |
| 噪声预测 | 条件 U-Net | `models/unet.py` 的本地 `ConditionalUNet` |
| 采样 | DDIM + CFG | `inference/sampler.py` 中的 `sample_latents()` |
| 输出解码 | 一次调用 decoder | `inference/sampler.py` 中的 `decode_latents()` |
| 参数规模 | 文生图约 1.45B | 来自作者公开转换权重，另外加载图像自编码器 |
| 本机执行 | 非论文实验环境 | Mac：文本/解码 CPU float32、U-Net MPS float16；NVIDIA：默认全组件 CUDA float16 |

默认 50 步、CFG=5、eta=0 是方便本机实验的设置。论文 v2 Figure 5 用 200 步、CFG=10、eta=1；Table 2 的定量评测又采用另一组设置，不能将两者混为一谈。要复现 FID/IS，还需要完整评测数据、样本数、预处理和指标实现。

本地 scheduler 显式使用作者的 sqrt 插值 beta 曲线和 DDIM 更新。MPS/CUDA、半精度和算子实现仍不保证得到论文中的完全相同图片。本项目验证的是算法与本机推理链路，不声称复现训练或完整基准指标。

建议先看论文 Figure 3 和 §3.1–3.3，再对照 `inference/sampler.py` 中 `sample_latents()` 的四个阶段，按上文「模型实现与调试」的顺序进入本地 BERT、U-Net、attention 和 KL 自编码器。可用 `--trace-shapes` 核对张量形状；随后固定 seed，只改变 `guidance` 或 `steps`，观察条件强度和采样长度对结果的影响。

实现依据：[作者噪声/时间步代码](https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/util.py)、[作者 DDIM 采样器](https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddim.py)。测试使用 [Diffusers 0.35.1 LDM 实现](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py) 作为独立数值参考。

## 验证记录

本机为 16 GiB Apple Silicon Mac，conda 环境为 `/Users/xiang/Documents/app/miniconda3/envs/diffusion`。预训练权重固定为 `CompVis/ldm-text2im-large-256@30de525ca11a880baea4962827fb6cb0bb268955`，三个大权重文件通过官方 SHA256 校验。

测试需要额外安装 Diffusers 0.35.1：

```bash
python -m pip install diffusers==0.35.1
python -m unittest discover -s tests -v
```

本机的 14 项测试通过。测试用随机初始化的小模型，对照参考实现检查 BERT、U-Net、VAE 编解码、CFG、DDIM 公式与随机种子；Diffusers 不参与实际生成。

真实 MPS 运行使用原生 256×256、float16 U-Net、CPU float32 文本编码器和解码器。此前同机完成了 batch=2 CFG、20 步推理，未发生 swap。精简后又用真实权重、`--steps 2 --trace-shapes` 完成端到端生成，得到 256×256 PNG；与精简前同参数生成的图片逐像素完全一致。形状输出覆盖 BERT、U-Net 下采样与 skip 连接、瓶颈、上采样和自编码器解码。

RTX 5070 Ti 的 CUDA 设备选择由单元测试覆盖，但当前机器没有该显卡。需要在目标机器运行 `python generate.py --device cuda --steps 2` 才能确认端到端兼容性。
