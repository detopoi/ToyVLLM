# Toy vLLM：从零实现一个大模型推理框架

这个项目的目标不是复刻完整的 vLLM，而是用尽量少、尽量清晰的代码，
一步一步实现现代大模型推理引擎中最重要的概念。

我们使用本地的 `Qwen3-1.7B` 作为唯一模型，先保证结果正确，再逐步提高速度和并发能力。
核心模型前向计算由我们自己用 PyTorch 实现，不直接调用
`transformers.AutoModelForCausalLM` 完成推理。`transformers` 可以用于分词和结果对照，
但不会替代我们要学习的模型与引擎代码。

## 当前硬件与模型

- GPU：NVIDIA GeForce RTX 4060 Ti，8 GB 显存
- 模型：Qwen3-1.7B，28 层 Transformer
- 权重：BF16，约 3.78 GiB
- 注意力：16 个 Query Head，8 个 KV Head，属于 GQA
- Head Dimension：128
- 模型最大上下文：40960 token

模型虽然声明支持 40960 token，但 8 GB 显存不能把“支持的最大长度”直接当成
“适合本机运行的长度”。仅 BF16 KV Cache 每个 token 就大约需要：

```text
28 层 × 2(K 和 V) × 8 个 KV Head × 128 × 2 字节
= 114688 字节
= 112 KiB/token
```

因此，4096 个缓存 token 约占 448 MiB。还要给模型权重、临时张量和 CUDA 工作区留空间。
项目初期默认使用较短的上下文，先从 512 或 1024 token 开始，稳定后再逐步增加。

## 总体架构

最终的代码大致分为五层：

```text
用户输入
  ↓
Tokenizer：文本与 token id 互相转换
  ↓
LLM Engine：接收请求，驱动 prefill 和 decode
  ↓
Scheduler + KV Cache：安排请求，管理 token 和缓存块
  ↓
Qwen3 Model：Embedding、Attention、MLP、LM Head
  ↓
PyTorch / CUDA
```

计划采用下面的目录结构：

```text
toyvllm/
├── config.py                 # 读取模型配置和引擎配置
├── tokenizer.py              # 文本编码、聊天模板和解码
├── sampling.py               # greedy、temperature、top-k、top-p
├── sequence.py               # 一条生成请求的状态
├── engine.py                 # 推理引擎主循环
├── scheduler.py              # 请求调度
├── kv_cache.py               # KV Cache 存储
├── block_manager.py          # 分页缓存的块分配
├── weight_loader.py          # safetensors 权重加载
├── models/
│   └── qwen3.py              # Qwen3 模型结构
└── layers/
    ├── rms_norm.py           # RMSNorm
    ├── rotary_embedding.py   # RoPE
    ├── attention.py          # GQA 与因果注意力
    └── mlp.py                # SwiGLU MLP
```

目录会随着实现逐步建立，不会一次生成所有空文件。

## 实现顺序

### 第 0 步：环境与项目骨架

目标：

- 建立可重复安装的 Python 环境
- 安装支持 RTX 4060 Ti 的 PyTorch
- 添加最小包结构、命令行入口和测试目录
- 写一个环境检查程序，打印 Python、PyTorch、CUDA、GPU 和模型目录信息

当前系统默认的 `python` 命令指向 Python 3.14，该解释器没有安装 PyTorch。
但系统还安装了可直接使用的 Python 3.10：

```text
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe
```

该解释器已经安装 PyTorch 2.2.2+cu121，并通过实际 CUDA 矩阵乘法确认可以使用
RTX 4060 Ti。它还包含 `transformers`、`tokenizers` 和 `safetensors`，因此后续优先
使用这个解释器，不重复下载大型 CUDA 依赖。当前只需要补充测试工具等轻量依赖。

这一阶段不加载模型。验收标准是：环境检查能够识别 CUDA 和 RTX 4060 Ti。

### 第 1 步：读懂模型输入

目标：

- 读取 `config.json`
- 加载 tokenizer
- 将一段文本转换为 token id，再还原为文本
- 解释 Qwen3 的聊天模板、特殊 token 和 thinking 模式

作用：

模型不直接读取字符串，只读取整数 token。Tokenizer 是用户文本和神经网络之间的桥梁。

验收标准是：文本编码再解码后语义保持一致，并能打印 token id。

### 第 2 步：实现 Qwen3 的基础层

按从简单到复杂的顺序实现：

1. RMSNorm
2. RoPE 旋转位置编码
3. GQA 因果自注意力
4. SwiGLU MLP
5. Decoder Layer

这一阶段先用小尺寸随机张量测试，不加载 1.7B 模型。这样可以把“数学实现错误”和
“权重加载错误”分开排查。

验收标准是：每个层的输入、输出形状正确，数值中没有 NaN，并通过单元测试。

### 第 3 步：组装完整模型并加载权重

目标：

- 实现 Embedding、28 个 Decoder Layer、最终 RMSNorm 和 LM Head
- 从两个 safetensors 分片加载权重
- 正确处理 Qwen3 权重名称和 tied embedding
- 尽量避免先在 CPU 完整复制一份权重再搬到 GPU，减少内存峰值

作用：

这一步把“模型结构”和“训练得到的参数”组合成真正可用的 Qwen3。

验收标准是：输入 token 后能得到最后一个位置的 logits，并和 Hugging Face
参考实现的结果在允许误差内一致。

### 第 4 步：实现最朴素的文本生成

目标：

- 选择 logits 最大的 token，也就是 greedy decoding
- 将新 token 追加到输入末尾
- 每生成一个 token，都重新计算完整序列
- 遇到 EOS 或达到最大生成长度时停止

作用：

这是第一个真正“能聊天”的版本。它很慢，但逻辑最直观，是后续优化必须保留的正确性基线。

验收标准是：给定固定提示词，可以稳定生成可读文本。

### 第 5 步：加入连续 KV Cache

目标：

- 区分 prefill 和 decode
- prefill 一次处理整段提示词
- decode 每次只输入一个新 token
- 每层保存历史 Key 和 Value，避免重复计算

作用：

没有 KV Cache 时，第 N 步会重复计算前 N-1 个 token 的 Key 和 Value。
有了缓存后，decode 阶段只计算新 token，生成速度会明显提升。

验收标准是：开启和关闭 KV Cache 得到相同 token，并记录两者生成耗时。

### 第 6 步：加入采样策略

目标：

- temperature
- top-k
- top-p
- 随机种子
- repetition penalty（可选）

作用：

Greedy 每次只选概率最高的 token，输出固定但容易单调。采样策略在合理概率范围内引入随机性。

验收标准是：相同随机种子结果可复现，不同参数会改变候选 token 分布。

### 第 7 步：静态批处理

目标：

- 一次处理多条提示词
- 使用 attention mask 隔离 padding
- 处理不同长度请求

作用：

GPU 擅长并行计算。批处理可以提高吞吐量，但静态批处理中，短请求仍要等待长请求完成。

验收标准是：批量结果与逐条生成结果一致，并比较吞吐量。

### 第 8 步：连续批处理与调度器

目标：

- 用 Sequence 保存每个请求的状态
- Scheduler 区分 waiting、running 和 finished 请求
- 已完成请求立即退出，新请求可以进入运行批次
- 每轮 decode 动态重组 batch

作用：

连续批处理是推理服务和普通离线批处理的重要区别。它减少 GPU 空转，并降低请求排队时间。

验收标准是：不同长度的请求可以交错执行，短请求不必等最长请求结束后才返回。

### 第 9 步：Paged KV Cache

目标：

- 把 KV Cache 切成固定 token 数的 block
- 用逻辑 block table 映射物理显存块
- 按需分配和回收 block
- 先用纯 PyTorch 实现清晰版本，不急于编写 CUDA/Triton 内核

作用：

连续 KV Cache 需要预留大块连续空间，容易浪费显存，也不方便请求动态增长。
分页管理让请求只占用实际需要的缓存块，是 vLLM 的核心思想之一。

验收标准是：请求增长时能追加 block，请求结束后 block 能被复用，并且生成结果不变。

### 第 10 步：性能测量与针对性优化

目标：

- 分别测量首 token 延迟、decode tokens/s 和总吞吐量
- 使用 PyTorch Profiler 找出瓶颈
- 尝试 SDPA、`torch.compile`、减少 Python 循环和张量复制
- 记录每项优化前后的数据

作用：

优化必须由测量驱动。只有先建立基线，才能判断某个改动是真优化还是增加了复杂度。

Windows 原生环境下不把 Triton 自定义内核作为前置条件。先把引擎机制学清楚，
以后可在 Linux/WSL2 环境中增加融合内核。

### 第 11 步：可选的 HTTP 服务

目标：

- 提供简单生成接口
- 支持流式返回 token
- 将服务层和推理引擎分离

作用：

这一层展示推理框架如何变成可被其他程序调用的服务，但不属于理解 vLLM 核心机制的前置步骤。

## 开发原则

后续每一步都遵循以下顺序：

1. 先在 `TUTORIAL.md` 中说明本次要解决的问题。
2. 编写最小实现，并用有教学意义的中文注释解释“为什么这样做”。
3. 添加针对当前功能的测试或对照程序。
4. 实际运行验证，而不是只保证语法正确。
5. 在本文件追加本次实现、作用、原理、运行方式和验证结果。

中文注释重点解释张量形状、算法目的、显存影响和容易出错的地方，不逐行翻译代码。

## 实现日志

### 2026-06-06：完成第 0 步，建立可运行骨架

本次增加：

- `pyproject.toml`：声明 Python 版本和 tokenizer 相关依赖
- `toyvllm/environment.py`：检查 Python、PyTorch、CUDA、GPU、BF16 和模型目录
- `toyvllm/config.py`：读取并校验 Qwen3 的关键结构参数
- `python -m toyvllm`：统一的教学命令行入口
- `tests/`：使用 Python 内置 `unittest`，不额外依赖 pytest

配置读取没有直接把整个 JSON 字典传遍项目，而是转换成 `ModelConfig`。这样后续
Attention、MLP 和 KV Cache 使用的每个维度都有明确名称，配置缺失或维度冲突也能尽早报错。

环境检查实际确认：

```text
Python       : 3.10.2
PyTorch      : 2.2.2+cu121
PyTorch CUDA : 12.1
GPU          : NVIDIA GeForce RTX 4060 Ti
GPU 显存     : 8188 MiB
支持 BF16    : True
```

运行方式：

```powershell
$PYTHON = "C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
& $PYTHON -m toyvllm --model Qwen3-1.7B env
```

配置程序还会输出 Qwen3-1.7B 的 KV Cache 预算：每 token 112 KiB，1024 token
约 112 MiB。以后设置最大并发数和缓存块数量时会直接用到这个计算。

### 2026-06-06：完成第 1 步，读懂模型输入

本次增加 `toyvllm/tokenizer.py`，支持：

- 普通文本的 encode 和 decode
- Qwen3 聊天模板展开
- 聊天消息直接编码成 token id
- 开启或关闭 thinking 模式
- 对空消息和非法角色提前报错

普通字符串不能直接送进神经网络。Tokenizer 会把文本切成词元，并映射成词表中的整数。
聊天模型还多一层模板：它使用 `<|im_start|>` 和 `<|im_end|>` 标记 user、assistant
等角色。因此，同一句用户输入作为普通文本编码和作为聊天消息编码，得到的 token 序列不同。

关闭 thinking 时，Qwen3 模板仍会在 assistant 开头放入一个空的：

```text
<think>

</think>
```

这表示明确跳过思考内容，不是模板失效。开启 thinking 后，模板只给出 `<think>` 起始标记，
模型可以继续生成思考 token。

运行方式：

```powershell
& $PYTHON -m toyvllm --model Qwen3-1.7B tokenize "你好，请介绍一下 KV Cache。"
& $PYTHON -m toyvllm --model Qwen3-1.7B tokenize --thinking "你好"
```

### 2026-06-06：建立 benchmark 记录机制

本次增加：

- `toyvllm/benchmark.py`：统一计时、平均时延、P50 和吞吐计算
- `bench.py`：项目统一性能测试入口
- JSONL 结果追加功能：每个阶段用 label 标记，历史结果不会被覆盖

当前模型前向尚未实现，所以 `bench.py` 只测试 tokenizer，并明确标注它不是模型推理性能。
本机串行测试一次短文本编码的 P50 时延约为 0.076 ms，平均时延约为 0.088 ms。
这个结果只验证 benchmark 管线可用，不能拿来代表 Qwen3 的生成速度。

模型能够生成 token 后，固定记录以下指标：

- TTFT（Time To First Token）：从提交请求到得到第一个输出 token 的时间
- TPOT（Time Per Output Token）：首 token 之后，每生成一个 token 的平均时间
- Decode tokens/s：单请求解码速度
- Total tokens/s：包含输入 token 和输出 token 的整体吞吐
- Peak VRAM：该轮测试的峰值显存

为了让不同阶段可比，基准测试固定使用相同提示词、相同输入 token、greedy decoding、
相同 `max_new_tokens`、相同 warmup 次数和重复次数。功能不同但输入不同的结果不能直接比较。
Benchmark 必须单独串行运行，不能同时跑测试、模型下载或其他 GPU 任务。短操作尤其容易受
系统调度抖动影响，因此同时记录平均值和 P50，不根据单次最快结果下结论。

保存当前结果的示例：

```powershell
& $PYTHON bench.py --save benchmarks/results.jsonl --label stage-01-tokenizer
```

后续至少保存以下推理阶段：

```text
stage-04-naive       每步重新计算完整序列
stage-05-kv-cache    使用连续 KV Cache
stage-07-static-batch
stage-08-continuous-batch
stage-09-paged-kv
```

### 2026-06-06：完成第 2 步，Qwen3 基础层

本次实现：

- `RMSNorm`：使用 FP32 计算均方根，再转回输入 dtype
- `RotaryEmbedding`：根据 token 位置旋转 Query 和 Key
- `Qwen3Attention`：16 个 Query Head、8 个 KV Head 的 GQA 结构
- `Qwen3MLP`：使用 `SiLU(gate) * up` 的 SwiGLU
- `Qwen3DecoderLayer`：Pre-Norm、Attention、MLP 和两次残差连接

Qwen3-1.7B 中，一层的数据流可以简化为：

```text
hidden_states
  ├─ residual ───────────────────────────────┐
  └─ RMSNorm → GQA Attention → 相加残差 ──────┘
                     │
                     ├─ residual ────────────┐
                     └─ RMSNorm → SwiGLU → 相加残差
```

GQA 不为每个 Query Head 单独保存 Key 和 Value。这个模型中两个 Query Head 共享一组
Key/Value，因此 KV Cache 只需要按 8 个 KV Head 保存，而不是按 16 个 Query Head 保存。

当前 PyTorch 2.2.2 的 SDPA 接口不能直接接收 GQA，所以代码先用 `repeat_interleave`
把 K/V 临时展开到 16 个头。这保证了数学结果正确，也让张量形状容易观察，但会增加临时显存。
后续性能优化会以 benchmark 为依据替换这一实现。

本阶段使用小尺寸随机模型测试，没有加载 1.7B 权重。验证内容包括：

- RMSNorm 与公式直接计算一致
- RoPE 在位置 0 不改变向量，并保持向量范数
- 修改未来 token 不会改变过去 token 的 Attention 输出
- MLP 和 Decoder Layer 输出形状正确且没有 NaN/Inf
- 把相同随机权重装入 Transformers 官方 Qwen3 Decoder Layer 后，所有权重键匹配
- 与官方 Decoder Layer 的最大输出误差约为 `2.38e-7`

最后一项参考对照很重要：形状正确不代表模型结构一定正确。RoPE 维度排列、Q/K Norm
位置或残差顺序只要有一个不一致，加载真实权重后都可能输出看似合理但实际错误的文本。

### 2026-06-06：完成第 3 步，组装模型并加载权重

本次增加：

- `Qwen3Model`：Embedding、28 个 Decoder Layer 和最终 RMSNorm
- `Qwen3ForCausalLM`：把隐藏状态投影到 151936 维词表
- `weight_loader.py`：读取 safetensors 索引并按两个分片加载参数

加载器先在 meta device 上构造模型。meta tensor 只记录形状和 dtype，不分配真实数据。
随后，两个 safetensors 分片直接加载到 CUDA，并使用 `assign=True` 绑定给模型参数。

如果直接普通构造模型，PyTorch 会先生成一份 FP32 随机参数，约占权重 BF16 版本的两倍；
再加载真实权重时还会短暂同时保留两份数据。这种加载峰值对 8 GB 显卡不合适。

加载器还会做三层检查：

1. 模型参数名和 safetensors 索引必须完全一致。
2. 每个分片的实际 tensor 必须符合索引记录。
3. 加载完成后不能有任何参数仍停留在 meta device。

RoPE 的 `inv_freq` 可以由配置计算，因此没有保存在权重中。meta 构造完成后，
加载器会在 GPU 上重建这个 buffer。

实测完整模型加载时间约 4.0 秒，随后可以在 RTX 4060 Ti 上完成 BF16 前向。

### 2026-06-06：完成第 4 步，朴素 greedy 生成

本次增加 `generate_greedy_naive`：

1. 把完整 prompt 输入 28 层模型。
2. 只保留最后位置的 logits，避免创建整段序列的巨大词表张量。
3. 使用 `argmax` 选择概率最高的 token。
4. 把新 token 追加到输入，下一步重新计算整个序列。
5. 遇到 EOS 或达到 `max_new_tokens` 后停止。

运行方式：

```powershell
& $PYTHON -m toyvllm --model Qwen3-1.7B generate `
    --max-new-tokens 16 "用一句话解释 KV Cache。"
```

第一次真实输出：

```text
KV Cache 是在大语言模型中
```

这证明 tokenizer、聊天模板、权重名称、28 层模型结构和 LM Head 已经连通。

当前算法故意不使用 KV Cache。假设 prompt 有 P 个 token，要生成 N 个 token，
每一步处理的序列长度依次为 `P, P+1, ..., P+N-1`。历史 token 的 K/V 和 MLP
结果被反复计算，所以输出越长，每一步通常越慢。

固定配置的 `stage-04-naive` 基线：

```text
GPU              RTX 4060 Ti 8 GB
输入             Qwen3 聊天模板，关闭 thinking
解码             greedy
max_new_tokens   16
warmup           1
iterations       3
平均 TTFT        61.19 ms
平均 TPOT        70.77 ms
输出吞吐         14.25 tokens/s
峰值显存         3886.4 MiB
模型加载         4.04 s（不计入 TTFT）
```

本机 PyTorch 2.2.2 的 Windows 构建没有包含 Flash Attention，SDPA 会退回其他 CUDA
实现。当前阶段接受这一点，因为下一步要测量的是 KV Cache 消除重复计算所带来的收益。

### 2026-06-06：完成第 5 步，连续 KV Cache

本次让 Attention、Decoder Layer 和完整模型都能接收并返回 `past_key_values`。
每层缓存两个张量：

```text
Key   [batch, num_kv_heads, cached_tokens, head_dim]
Value [batch, num_kv_heads, cached_tokens, head_dim]
```

Qwen3-1.7B 的具体形状是：

```text
[batch, 8, cached_tokens, 128]
```

缓存保持 8 个原始 KV Head，不保存 GQA 展开后的 16 个 Head，否则 KV Cache 显存会翻倍。

生成现在分成两个阶段：

```text
Prefill:
  一次输入完整 prompt
  为每一层计算并保存 prompt 的 K/V
  得到第一个输出 token

Decode:
  每次只输入刚生成的 1 个 token
  读取历史 K/V，并追加当前 token 的 K/V
  得到下一个输出 token
```

位置编码和 mask 有一个容易出错的细节。假设缓存中已有 100 个 token，decode 输入的第一个
token 的绝对位置是 100，而不是 0。若对形状 `[query=1, key=101]` 直接使用普通左上角
causal mask，query 可能只能看到第 0 个 key。当前实现根据 `past_length` 计算绝对
`position_ids`；单 token decode 允许访问全部历史，多 token chunk 则构造带位置偏移的 mask。

正确性验证：

- 小随机模型的 cached logits 与完整序列 logits 一致
- 一次输入多个 cached chunk 时结果也一致
- cached greedy 与 naive greedy 的输出 token 完全一致
- 真实 Qwen3-1.7B 的 16 个输出 token 逐个一致

真实输出片段：

```text
“KV Cache” 是 Key-Value Cache 的缩写，是一种
```

#### Benchmark 结果

所有数据均为 greedy、16 个输出 token、warmup 1、iterations 3。TTFT 包含 prompt
prefill；Decode 吞吐排除首 token，只衡量后续 token。

| 输入长度 | 后端 | TTFT | TPOT | Decode 吞吐 | 峰值显存 |
|---:|---|---:|---:|---:|---:|
| 18 token | naive | 26.84 ms | 30.24 ms | 33.07 tokens/s | 3886.5 MiB |
| 18 token | cached | 28.51 ms | 29.00 ms | 34.48 tokens/s | 3892.1 MiB |
| 252 token | naive | 69.62 ms | 68.36 ms | 14.63 tokens/s | 3900.0 MiB |
| 252 token | cached | 71.35 ms | 71.84 ms | 13.92 tokens/s | 3945.1 MiB |
| 1020 token | naive | 103.89 ms | 104.74 ms | 9.55 tokens/s | 3941.3 MiB |
| 1020 token | cached | 103.35 ms | 72.05 ms | 13.88 tokens/s | 4119.1 MiB |

1020-token 上下文中，KV Cache 让 Decode 吞吐提高约 45%，TPOT 降低约 31%。
但在 18 token 时提升很小，252 token 时甚至略慢。这说明“计算量减少”不必然等于
“实际耗时立刻减少”。

naive 每轮重新处理整段序列，矩阵较大，GPU 利用率较高。cached 每层只处理一个新 token，
把大矩阵乘法变成了大量小矩阵向量计算，还增加 Python 调度、缓存拼接和内存访问。
在单请求、短上下文下，这些固定开销可能抵消节省的 FLOPs。上下文足够长后，避免重复计算
历史 token 的收益才超过这些开销。

这也解释了为什么真实推理引擎还需要连续批处理：把多个请求的新 token 合并成一个 batch，
可以让 cached decode 重新形成较大的矩阵计算，提高 GPU 利用率。

当前缓存仍有两个刻意保留的低效点：

1. 每轮使用 `torch.cat` 重新分配更长的连续 K/V。
2. Attention 临时把 8 个 KV Head 展开到 16 个 Query Head。

后续 Paged KV Cache 和 Attention 优化会继续处理这些问题。

### 2026-06-06：完成第 6 步，采样策略

此前生成过程固定使用 greedy：

```text
next_token = argmax(logits)
```

它每次选择 logits 最大的 token，因此相同输入总是得到相同结果。Greedy 适合作为性能
benchmark 和正确性对照，但容易让开放式文本变得单调。

本次新增 `SamplingParams` 和独立采样器，支持：

- `temperature`
- `top-k`
- `top-p`
- 固定随机种子

处理顺序如下：

```text
原始 logits
  ↓ 除以 temperature
调整概率分布的尖锐程度
  ↓ top-k
只保留 logits 最高的 k 个 token
  ↓ top-p
保留累计概率刚达到 p 的最小候选集合
  ↓ softmax + multinomial
按剩余概率随机抽取一个 token
```

#### Temperature

采样概率来自：

```text
softmax(logits / temperature)
```

- temperature 小于 1：概率分布更尖锐，更偏向高概率 token
- temperature 大于 1：概率分布更平坦，输出更随机
- temperature 等于 0：项目约定为 greedy，不进行除法

#### Top-k

`top_k=20` 表示每一步只允许 logits 最高的 20 个 token 参与采样，其余 token 的
logits 被设置为负无穷，softmax 后概率变成 0。

#### Top-p

Top-p 也叫 nucleus sampling。它先按概率从高到低排序，再保留累计概率刚好达到阈值的
最小集合。候选数量不是固定的：模型很确定时可能只留下少量 token，不确定时会留下更多。

例如概率为：

```text
[0.64, 0.24, 0.09, 0.03]
```

使用 `top_p=0.8` 时保留前两个 token，因为 `0.64 + 0.24 = 0.88`，第一次超过 0.8。

#### 随机种子

随机数生成器在一次请求开始时创建，并在整个生成过程中持续使用。不能每生成一个 token
就重新设置 seed，否则每一步都会从同一个随机状态开始，破坏正常的随机序列。

相同模型、输入、参数和 seed 会产生相同 token 序列。真实 Qwen3 验证结果：

```text
seed=123 第一次 == seed=123 第二次
seed=123 输出 != seed=456 输出
sampling 输出 != greedy 输出
```

Qwen3 本地 `generation_config.json` 的推荐参数是：

```text
temperature = 0.6
top_k       = 20
top_p       = 0.95
```

默认命令仍使用 greedy，保证性能数据可重复比较。增加 `--sample` 才会采用上述参数：

```powershell
& $PYTHON -m toyvllm --model Qwen3-1.7B generate --sample --seed 123 `
    --max-new-tokens 32 "写一句关于夜空的短句。"
```

#### Benchmark 结果

相同 cached 后端、18-token 输入、16-token 输出、warmup 1、iterations 3：

| 策略 | TTFT | TPOT | Decode 吞吐 | 峰值显存 |
|---|---:|---:|---:|---:|
| greedy | 29.95 ms | 29.96 ms | 33.40 tokens/s | 3892.1 MiB |
| sampling | 31.17 ms | 30.60 ms | 32.68 tokens/s | 3893.7 MiB |

本次实现中采样让 Decode 吞吐下降约 2.2%，峰值显存增加约 1.6 MiB。这是为了获得更多样
输出所支付的计算成本，不是性能优化。跨阶段性能 benchmark 仍固定使用 greedy。

测试覆盖：

- Greedy 一定选择最大 logits
- Top-k 恰好保留 k 个候选
- Top-p 保留最小 nucleus
- 非法 temperature、top-k、top-p 会提前报错
- 相同 seed 可复现采样序列
- Naive 和 cached 后端在相同 seed 下输出一致

采样超参数可以在启动时分别指定，不要求一定使用 `--sample`：

```powershell
& $PYTHON -m toyvllm generate `
    --temperature 0.8 `
    --top-k 40 `
    --top-p 0.9 `
    --seed 123 `
    "写一句关于夜空的短句。"
```

只传其中任意一个也会进入采样模式，未指定项使用模型 `generation_config.json` 的默认值。
例如只传 `--top-k 50` 时，temperature 和 top-p 分别使用 0.6 和 0.95。

### 2026-06-06：完成第 7 步，静态批处理

静态批处理把多条请求放进同一个固定 batch，一次模型前向同时计算多条序列。GPU 的矩阵计算
通常能从更大的 batch 中获得更高利用率。

#### 不同长度与左 Padding

Tensor 的每一维必须是规则矩形，但 prompt 长度可能不同，例如：

```text
请求 A: [a1, a2, a3, a4]
请求 B:         [b1, b2]
```

当前实现从左侧补 pad：

```text
input_ids:
A: [a1,  a2,  a3, a4]
B: [pad, pad, b1, b2]

attention_mask:
A: [1, 1, 1, 1]
B: [0, 0, 1, 1]
```

使用左 padding 后，每一行的最后位置都是 prompt 的真实末 token，可以统一取最后位置 logits。
Attention mask 保证真实 token 不会关注 pad。

RoPE 位置不能直接使用 padded 数组下标。请求 B 的 `b1, b2` 应使用位置 `0, 1`，而不是
`2, 3`。因此位置由下面公式得到：

```text
position_ids = attention_mask.cumsum(-1) - 1
```

padding 位置会被截断为 0，真实 token 从 0 递增。

实现时还遇到一个数值边界：padding query 的全部 key 都被屏蔽时，一些 SDPA 后端会返回
NaN。解决方式是仅让无效 padding query 看到自己的 padding key，保证该行数值有限；
真实 query 仍然看不到任何 padding key。

#### 批量 KV Cache

每层缓存形状从单请求：

```text
[1, num_kv_heads, tokens, head_dim]
```

变为：

```text
[batch_size, num_kv_heads, tokens, head_dim]
```

Prefill 一次处理整个 batch。之后每轮 decode 的输入形状是 `[batch_size, 1]`，
一次前向为每条请求生成一个 token。

每条请求独立检查 EOS。已经结束的请求不再追加输出，但静态 batch 的形状不能缩小，
它仍要用 pad 占住原来的槽位，直到 batch 中最长请求结束。这是静态批处理的主要浪费，
也是下一阶段连续批处理要解决的问题。

#### 批量采样

每条请求使用独立随机数生成器。给定基础 seed 时，第 i 条请求使用 `seed+i`。
这样某条请求不会因为同 batch 中另一条请求多生成一个 token 而改变随机序列，
第一条请求单独运行或放入 batch 时也能保持结果一致。

运行不同长度请求：

```powershell
& $PYTHON -m toyvllm batch --max-new-tokens 16 `
    "你好" `
    "用一句话解释 KV Cache。" `
    "请用三个词描述夏天。"
```

也可以同时独立指定全部采样参数：

```powershell
& $PYTHON -m toyvllm batch `
    --temperature 0.7 --top-k 30 --top-p 0.9 --seed 123 `
    "你好" "介绍 KV Cache。"
```

真实三请求测试的输入长度为 `[13, 18, 19]`，输出分别保持独立，padding 没有改变结果。

#### Benchmark 结果

固定 18-token prompt、greedy、16 个输出 token、warmup 1、iterations 3：

| Batch size | Batch TTFT | 每轮 TPOT | 总输出吞吐 | 请求吞吐 | 峰值显存 |
|---:|---:|---:|---:|---:|---:|
| 1 | 38.36 ms | 37.37 ms | 26.71 tokens/s | 1.67 req/s | 3892.1 MiB |
| 2 | 36.83 ms | 37.34 ms | 53.61 tokens/s | 3.35 req/s | 3899.8 MiB |
| 4 | 37.81 ms | 37.13 ms | 107.61 tokens/s | 6.73 req/s | 3915.2 MiB |

Batch 从 1 增至 4 后，总输出吞吐提高约 4.03 倍，而一轮 decode 的耗时基本不变。
这表示 GPU 在相近时间内从一次产出 1 个 token 变成产出 4 个 token。

这里的 TPOT 是整个 batch 完成一轮 decode 的耗时，不是用它除以 batch size 后的单请求
延迟。静态 batch 提升的是系统总吞吐，单条请求仍约每 37 ms 得到一个 token。

当前测试的 prompt 和输出较短，所以 batch=4 仅比 batch=1 增加约 23 MiB 峰值显存。
长上下文下，每条请求都要保存独立 KV Cache，显存会随 batch size 和 token 数明显增长。

测试覆盖：

- 左 padding batch 的 logits 与逐条模型前向一致
- 静态 batch greedy 输出与逐条 cached 输出一致
- batch 第一条采样流与单请求同 seed 一致
- 每条请求独立遇到 EOS 并停止追加输出
- padding query 不产生 NaN

### 2026-06-06：完成第 8 步，连续批处理与 Scheduler

这一阶段需要先区分两个经常被混在一起的概念：

```text
Scheduler：决定本轮哪些请求运行
Executor：把这些请求转换成 Tensor，并执行模型
```

Scheduler 本身不负责矩阵乘法，也不应该知道 KV Cache 是连续 Tensor 还是分页 Block。
这种边界让下一阶段替换缓存布局时，不必重写请求状态机。

#### 为什么需要 Scheduler

假设一个静态 batch 中有四条请求：

```text
请求 A：生成 4 token
请求 B：生成 16 token
请求 C：生成 4 token
请求 D：生成 16 token
```

第 4 轮后 A、C 已结束，但静态 batch 的形状仍是 4。直到 B、D 完成前，A、C 的槽位只能
填 pad，不能放入等待中的新请求。

连续批处理希望做到：

```text
第 4 轮：
A、C 完成
  ↓ 立即释放两个槽位
等待队列中的 E、F 进入
  ↓
下一批 GPU 工作变成 [B, D, E, F]
```

它优化的是整个请求流，而不是单独某一条序列。

#### Sequence：请求的状态载体

`sequence.py` 中每条请求保存：

- `request_id`
- prompt token
- 已生成 token
- 最大生成长度
- EOS token
- SamplingParams
- 当前状态
- 进入运行和完成时的调度轮次
- 完成原因

状态只有三种：

```text
                 有空闲槽位
WAITING  -------------------------->  RUNNING
                                          |
                                          | EOS 或达到 max_new_tokens
                                          v
                                      FINISHED
```

状态转换只能单向发生。FINISHED 请求不会重新回到 RUNNING，这让缓存释放和完成回调更容易
推理，也能尽早暴露重复调度同一请求的错误。

#### 三个队列

教学版 Scheduler 内部维护：

```text
waiting  : deque，尚未获得运行槽位
running  : 有 KV Cache、每轮可以生成 token
finished : 已完成，用于返回最终结果
```

`waiting` 使用 FIFO：

```text
先到的请求先获得空槽
```

FIFO 很容易理解，也能避免后来的短请求不断插队导致旧请求饥饿。它不是所有场景的最优策略：
生产系统还可能考虑请求优先级、prompt 长度、SLA、剩余 token 估计或抢占，但应先建立一个
行为明确且不会饿死请求的基线。

#### 一个调度轮次做什么

当前 `engine.step()` 的顺序是：

```text
1. 取出当前所有 RUNNING 请求
2. 执行一次 decode，每条请求前进一步
3. 检查 EOS 和长度上限
4. 把完成请求移到 FINISHED，并释放 KV Cache
5. 按 FIFO 从 WAITING 填满空闲槽位
6. 对新进入 RUNNING 的请求执行 prefill，生成首 token
```

所以一个轮次最多会执行两次模型调用：

```text
一次 decode：服务原有 running 请求
一次 prefill：服务刚被接纳的新请求
```

这样做的优点是语义清晰：decode 不会因为新 prompt 很长而被拼成难以理解的混合 Tensor。
生产级引擎会进一步使用 chunked prefill、token budget 或统一 token packing 平衡首 token
延迟与 decode 延迟。

#### 容量限制

当前 Scheduler 只有一个容量参数：

```text
max_num_seqs
```

它限制同时处于 RUNNING 的请求数量。例如值为 4 时，即使 waiting 中有 100 条请求，
也最多同时维护 4 条请求的 KV Cache。

仅限制请求数还不够构成生产级显存保护。四条 100-token 请求和四条 10000-token 请求的
KV Cache 大小差异巨大。完整 vLLM Scheduler 还需要根据可用 KV Block、每轮 token budget
和最大模型长度决定能否接纳请求。Paged KV Cache 完成后才能可靠加入这些资源约束。

#### 在线请求到达

除了一次性 `run()`，引擎还暴露 `step()`：

```python
engine.add_request(first_prompt, ...)
engine.step()

# 模拟服务运行过程中收到新请求
engine.add_request(second_prompt, ...)
engine.step()
```

第二条请求先进入 waiting。下一轮如果有空槽，就会被接纳并 prefill。HTTP 服务以后只需要
在事件循环中不断接收请求、调用 `step()`、返回新 token，而不必改变 Scheduler 状态机。

#### 不同长度 KV Cache 如何组成动态 Batch

每条请求的紧凑 KV Cache 长度不同：

```text
A: [heads, 100, head_dim]
B: [heads, 137, head_dim]
C: [heads,  52, head_dim]
```

PyTorch 的普通 batch Tensor 必须是矩形。当前 Executor 每轮找到最大长度 137，左侧补零：

```text
A: [37 pad | 100 valid]
B: [         137 valid]
C: [85 pad |  52 valid]
```

Attention mask 屏蔽 pad。模型返回后，再按 mask 把每行拆回紧凑缓存。

这保证数学结果正确，但代价很高：

- 28 层都要创建 padding Tensor
- 每轮都要 `torch.cat`
- 每轮都要把 batch cache 拆回独立请求
- 请求加入或退出时持续发生显存分配与复制

Scheduler 决定了“应该动态重组 batch”，但连续 Tensor 布局让重组本身很昂贵。

#### 正确性与公平性验证

测试覆盖：

- WAITING → RUNNING → FINISHED 单向状态转换
- FIFO admission
- 请求结束后槽位立即复用
- EOS 和长度上限都能结束请求
- running 数量永远不超过 `max_num_seqs`
- 连续批处理输出与逐条 cached 输出一致
- 短请求先完成，后续 waiting 请求提前进入
- 可以在两个 `step()` 之间添加在线请求

一个 tiny model 的调度轨迹为：

```text
step=00  decode=[]       prefill=[0, 1]
step=01  decode=[0, 1]   prefill=[2]
step=02  decode=[1, 2]   prefill=[]
```

请求 0 在 step 1 完成，请求 2 同一轮获得空槽。它不必等请求 1 完成。

真实模型可以这样查看轨迹：

```powershell
& $PYTHON -m toyvllm continuous --max-num-seqs 2 --show-schedule `
    "你好" "解释 KV Cache" "描述夏天" "什么是 GPU"
```

#### Benchmark：结果为什么暂时变差

测试工作负载：

```text
请求数       8
最大并发     4
prompt       每条 18 token
生成上限     [4, 16, 4, 16, 4, 16, 4, 16]
解码         greedy
warmup       1
iterations   3
```

结果：

| 模式 | 总输出吞吐 | 平均请求完成延迟 | 峰值显存 |
|---|---:|---:|---:|
| 静态 batch | 65.51 tokens/s | 688.88 ms | 3915.2 MiB |
| 连续 batch | 54.86 tokens/s | 790.42 ms | 3939.8 MiB |

连续模式只有静态吞吐的约 0.84 倍，平均请求延迟也更高。这不是 Scheduler 没有回收槽位，
而是当前 Executor 每轮 pack/unpack KV Cache 的成本超过了减少空槽带来的收益。

这个结果说明：

```text
好的调度策略 + 不适合动态增长的缓存布局
不等于
高性能推理引擎
```

静态 batch 虽然浪费短请求槽位，但整个 KV Tensor 保持连续，GPU 可以反复在稳定形状上计算。
当前连续引擎减少了一部分无效计算，却增加了大量显存复制、分配和 Python 循环。

因此，Scheduler 阶段的成果主要是：

- 请求生命周期已经正确
- 动态 admission 已经正确
- FIFO 公平性已经建立
- 在线 arrival 接口已经建立
- 性能瓶颈被明确定位到 KV Cache 数据结构

下一阶段 Paged KV Cache 的目标，不只是“再加一个功能”，而是让每条请求通过 block table
引用固定物理块。请求加入、增长和退出时只修改 block 映射，不再每轮搬动整段历史缓存。

### 2026-06-06：第 9A 步，Paged KV Cache 内存管理层

Paged Attention 是一个大工程，本项目拆成三段：

```text
9A  Block Manager + 物理 KV Block 池
    先把分配、追加、回收和逻辑映射做正确

9B  Engine 接入 Paged KV Cache
    Prefill/Decode 把 K/V 写入物理块，请求结束回收块

9C  Paged Attention
    Attention 根据 Block Table 直接读取离散物理块
    消除“先 gather 成连续 Cache 再调用 Attention”的中间复制
```

本次只完成 9A。现在已经具备分页存储，但模型 Attention 还没有走这条路径，因此本阶段不做
端到端 tokens/s 对比，也不声称已经加速。

#### 为什么连续 Tensor 不适合动态请求

之前一条请求的每层 KV Cache 是：

```text
[1, num_kv_heads, sequence_length, head_dim]
```

请求每生成一个 token，就要把新 K/V 追加到 sequence 维。动态 batch 中不同请求长度不同，
还需要每轮补齐、拼接和拆包。

分页方案把显存预先切成等大的物理块。例如 `block_size=4`：

```text
物理 Block 0: 4 个 token 的 K/V 空间
物理 Block 1: 4 个 token 的 K/V 空间
物理 Block 2: 4 个 token 的 K/V 空间
...
```

请求不再拥有一整段连续显存，只拥有一张 Block Table。

#### 逻辑块与物理块

假设一条请求有 10 个 token，block size 为 4，它需要三个逻辑块：

```text
逻辑块 0：token 0..3
逻辑块 1：token 4..7
逻辑块 2：token 8..9
```

物理块不要求连续。Block Table 可能是：

```text
logical block 0 -> physical block 5
logical block 1 -> physical block 2
logical block 2 -> physical block 11
```

那么 token 6 的地址计算为：

```text
logical_block = 6 // 4 = 1
block_offset  = 6 % 4  = 2
physical_block = block_table[1] = 2

最终位置 = physical block 2 的 offset 2
```

这和操作系统虚拟内存的页表思想相似：程序看到连续的逻辑地址，底层物理页可以离散。

#### BlockManager 管什么

`block_manager.py` 只管理整数物理块号，不持有 K/V Tensor：

```text
free blocks   : 当前可分配的物理块编号
block tables  : request_id -> 物理块编号列表
num_tokens    : 请求当前有效 token 数
```

它负责：

1. 注册请求。
2. 根据 prompt 长度计算需要多少块。
3. Decode 跨过块边界时追加一个块。
4. 请求结束后回收全部块。
5. 容量不足时抛出 `OutOfBlocksError`。

分配公式：

```text
num_blocks = ceil(num_tokens / block_size)
           = (num_tokens + block_size - 1) // block_size
```

例如 block size 为 4：

```text
0 token -> 0 block
1 token -> 1 block
4 token -> 1 block
5 token -> 2 block
```

#### 原子分配

请求从 4 token 增长到 9 token 时，可能一次需要新增两个块。如果空闲池只剩一个块，
不能“先拿走一个再失败”，否则请求会处于半更新状态。

当前 `reserve` 顺序是：

```text
1. 计算总共需要多少块
2. 检查空闲块是否足够
3. 完整取出所需块
4. 一次性替换不可变 BlockTable
```

因此 Out Of Blocks 后：

- 原 Block Table 不变
- `num_tokens` 不变
- 空闲队列不变

Scheduler 下一阶段可以根据 `can_reserve` 决定等待、抢占或拒绝，而不会修复半分配请求。

#### 物理 KV Block 池

`kv_cache.py` 一次性预分配两个大 Tensor：

```text
key_cache:
[num_layers, num_blocks, block_size, num_kv_heads, head_dim]

value_cache:
[num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

物理块号直接索引第二维。写入新 token 时，BlockManager 返回：

```text
(physical_block_id, block_offset)
```

PagedKVCache 把模型产生的：

```text
[1, num_kv_heads, num_new_tokens, head_dim]
```

转换为物理池需要的：

```text
[num_new_tokens, num_kv_heads, head_dim]
```

并写入对应槽位。

当前 `read(BlockTable)` 可以按逻辑顺序把离散物理块 gather 回紧凑连续 Cache。这个接口主要
用于测试和 9B 过渡接入。最终 9C Paged Attention 应直接按照 Block Table 读取物理池，
不再先 gather 整段历史。

#### Block 大小的取舍

Block 太大：

- Block Table 更短
- 分配管理次数更少
- 最后一个未填满 Block 的内部碎片更多

Block 太小：

- 内部碎片更少
- Block Table 更长
- 分配和地址映射次数更多
- Paged Attention 需要跨更多块读取

Qwen3-1.7B 的 BF16 KV Cache 每 token 为 112 KiB。使用 16-token Block：

```text
一个物理 Block = 112 KiB * 16 = 1792 KiB = 1.75 MiB
```

如果提供 1 GiB KV Cache 预算：

```text
大约 585 个 Block
全体请求合计约 585 * 16 = 9360 个 token
```

这 9360 token 可以属于一条长请求，也可以动态分给许多短请求，不需要按每条请求的最大长度
提前预留。

#### 物理 Block 与 CUDA Shared Memory 不是一回事

这两个概念名字中都有“块/共享”，但层级完全不同：

```text
Paged KV Physical Block
  位置：GPU global memory（显存）
  大小：当前模型 16 token 约 1.75 MiB
  生命周期：跨很多次 Prefill/Decode kernel
  用途：持久保存历史 token 的 K/V

CUDA Thread Block Shared Memory
  位置：SM 上的片上共享内存
  大小：通常是几十到几百 KiB 级别
  生命周期：一次 kernel 执行
  用途：线程协作、复用当前计算 tile、减少 global memory 访问
```

Paged Attention kernel 未来可能把一小片 K/V 从 global memory 搬到寄存器或 shared memory
参与点积，但不可能把完整请求的 KV Cache 长期放在 shared memory 中。

#### 当前不做活跃请求间共享

当前不实现两条 RUNNING 请求同时引用同一个物理块。`free` 后的块可以被后续请求复用，
但活跃块只有一个所有者，因此不需要引用计数。

Prefix Cache 会让相同前缀的请求共享只读 Block，并在修改时使用 Copy-on-Write。这需要：

- Block 引用计数
- 前缀哈希
- 只读共享规则
- Copy-on-Write
- 淘汰策略

它属于 Paged KV Cache 之上的独立优化，不在第一版 Block Manager 中提前混入。

#### 本阶段验证

新增测试覆盖：

- Prompt 初始分配
- Decode 跨 Block 边界追加
- 释放后物理 Block 被其他请求复用
- 非连续物理 Block 组成连续逻辑序列
- Out Of Blocks 时状态完全不变
- `can_reserve` 不修改状态
- 多层 K/V 写入后按 Block Table 完整还原
- 清零释放块
- K/V 形状检查
- Block 显存大小计算

测试还发现一个 PyTorch 高级索引细节：

```python
tensor[:, ids].zero_()
```

LongTensor 高级索引返回副本，`zero_()` 不会清零原物理池。最终使用：

```python
tensor.index_fill_(dim=1, index=ids, value=0)
```

在原 Tensor 上完成写回。

### 当前验证结果

```text
43 个测试全部通过
Python compileall 通过
CUDA、BF16、真实权重加载通过
Tokenizer 普通文本往返和聊天模板测试通过
单层结果与 Transformers 官方 Qwen3 对齐
真实 Qwen3-1.7B greedy 生成通过
Naive 与 cached 生成结果一致
Greedy 与 sampling 均可运行，采样结果可复现
不同长度静态 batch 与逐请求结果一致
Scheduler 状态转换、FIFO、在线 arrival 和动态 admission 通过
Paged Block 分配、回收、离散映射和 K/V 读写通过
多种上下文长度的 benchmark 已写入 benchmarks/results.jsonl
```

## 当前进度

- [x] 第 0 章：确认硬件、模型配置并设计整体路线
- [x] 第 0 步：建立运行环境与项目骨架
- [x] 第 1 步：读懂模型输入
- [x] 第 2 步：实现 Qwen3 基础层
- [x] 第 3 步：组装模型并加载权重
- [x] 第 4 步：朴素文本生成
- [x] 第 5 步：连续 KV Cache
- [x] 第 6 步：采样策略
- [x] 第 7 步：静态批处理
- [x] 第 8 步：连续批处理与调度器
- [ ] 第 9 步：Paged KV Cache
  - [x] 第 9A 步：Block Manager 与物理 KV Block 池
  - [ ] 第 9B 步：Engine 接入分页缓存
  - [ ] 第 9C 步：Paged Attention 直接按 Block Table 读取
- [ ] 第 10 步：性能测量与优化
- [ ] 第 11 步：HTTP 服务（可选）
