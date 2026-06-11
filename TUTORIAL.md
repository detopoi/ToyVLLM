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
├── kv_cache.py               # KV Cache 存储
├── weight_loader.py          # safetensors 权重加载
├── engine/
│   ├── llm_engine.py         # Prefill/Decode 主循环与执行后端
│   ├── scheduler.py          # 请求调度
│   ├── sequence.py           # 一条生成请求的状态
│   ├── block_manager.py      # 分页缓存的块分配
│   └── __init__.py           # 公共 API 与延迟导出
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

`engine/sequence.py` 中每条请求保存：

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

`engine/block_manager.py` 只管理整数物理块号，不持有 K/V Tensor：

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

### 2026-06-06：第 9B 步，Engine 接入分页缓存

9A 只有独立的 BlockManager 和物理池。9B 把它们接入真实请求生命周期：

```text
请求进入 WAITING
  ↓ 不占 KV Block
Scheduler 接纳为 RUNNING
  ↓ 按 Prompt 长度分配 Block
Prefill
  ↓ 把 Prompt K/V 写入物理池
Decode
  ↓ 每轮 reserve(1)，只写新增 token 的 K/V
请求完成
  ↓ 释放 Block Table 中全部物理块
```

新增 `PagedContinuousBatchEngine`，复用已有的 Sequence、Scheduler、采样器和调度轮次。
它只替换缓存资源管理，不复制一套新的调度策略。

#### WAITING 请求为什么不提前分配

等待队列可能很长。如果请求刚进入系统就按 Prompt 分配显存，大量尚未运行的请求会占满
KV Block，反而让真正 RUNNING 的请求无法继续 Decode。

因此：

```text
WAITING：只有 CPU 上的 prompt token 和状态
RUNNING：拥有 Block Table 和物理 KV Block
FINISHED：Block 已回收到空闲池
```

Block 的生命周期与 RUNNING 状态对齐。

#### Admission 同时检查两个资源

旧 Scheduler 只检查：

```text
running 数量 < max_num_seqs
```

分页后还需要检查：

```text
Prompt 所需 Block 数 <= 当前空闲 Block 数
```

分页引擎按 FIFO 查看队首请求。如果队首 Prompt 暂时放不下：

- 存在 RUNNING 请求：停止接纳，等待它们释放 Block
- 没有 RUNNING 请求：抛出 `OutOfBlocksError`，因为系统不可能自行取得进展

当前严格 FIFO 不会绕过大请求去接纳后面的小请求。这样保证公平性，但可能产生
Head-of-Line Blocking。生产系统可以增加优先级或 backfilling 策略，但必须明确它们对
公平性和饥饿的影响。

#### Prefill 如何写入分页池

Scheduler 接纳请求后，BlockManager 根据完整 Prompt 长度创建 Block Table。
模型 Prefill 仍然用左 padding 组成规则 batch，返回：

```text
每层 K/V [batch, kv_heads, padded_prompt_length, head_dim]
```

`write_prefill_batch` 根据 attention mask 去掉每行 padding，然后按该请求 Block Table
写入物理池。Padding 从来不会占用长期 KV Block。

#### Decode 的读写顺序

本阶段 Attention 还不能直接读取物理块，所以每轮先执行：

```text
Block Table + 物理池
  ↓ read_batch
临时连续历史 Cache
  ↓ 模型 Decode
历史 Cache + 当前 token 的 Present Cache
```

然后只取 Present Cache 的最后一列，也就是当前 token 的 K/V，写入新物理槽位。

关键顺序是：

```text
1. 使用旧 Block Table gather 历史 Cache
2. reserve_many，为本轮新 token 分配槽位
3. 执行模型
4. 把 Present 最后一列写入新槽位
```

不能先 reserve 再 gather。reserve 会把 `num_tokens` 增加一，如果这时 read_batch，
它会尝试读取尚未写入的新槽位。

#### 为什么需要 reserve_many

一个 Decode Batch 中可能有四条请求同时跨过 Block 边界，每条都需要一个新 Block。
如果分别调用：

```text
can_reserve(A) -> True
can_reserve(B) -> True
can_reserve(C) -> True
can_reserve(D) -> True
```

它们可能都看到了同一批空闲 Block。`reserve_many` 会先汇总整个 batch 的新增块需求，
确认总量足够后再修改任何 Block Table。

因此容量不足时，所有请求保持原状，不会出现一半请求增长、一半请求失败的 batch。

#### 请求完成时回收什么

连续引擎完成请求时删除：

```text
request_id -> 连续 KV Tensor
request_id -> 随机数生成器
```

分页引擎完成请求时删除：

```text
request_id -> Block Table
物理 Block -> 回到 free queue
request_id -> 随机数生成器
```

真实模型测试使用 32 个 Block。四条请求全部完成后：

```text
结束后空闲 Blocks：32/32
```

说明逻辑运行槽、Block Table 和物理显存块的生命周期已经闭环。

#### 9B 与 9C 的边界

9B 已经做到：

- 请求不持有不断增长的连续 KV Tensor
- KV 显存由固定物理池统一预分配
- 每轮只写新增 token 的 K/V
- Block 容量参与 Scheduler admission
- 请求结束后 Block 可立即复用

9B 仍然没有做到：

- Attention 直接按照 Block Table 读取 K/V
- 避免每轮 `read_batch` gather
- 避免不同长度请求的临时 padding
- 使用专门的 Paged Attention CUDA/Triton kernel

因此 9B 是“分页存储 + 连续 Attention”的过渡版本。

#### Benchmark

工作负载与 Scheduler 阶段相同：

```text
请求数       8
最大并发     4
Prompt       18 token
生成上限     [4, 16, 4, 16, 4, 16, 4, 16]
Block Pool   64 blocks
Block Size   16 tokens
```

结果：

| 缓存后端 | 总输出吞吐 | 峰值显存 |
|---|---:|---:|
| 请求级连续 Cache | 58.77 tokens/s | 3939.8 MiB |
| Paged 9B | 60.73 tokens/s | 4026.0 MiB |

分页存储吞吐提高约 3%。它不再把模型返回的整段历史 Cache 拆回每条请求，只写最后一个
token，因此抵消了一部分物理池 gather 成本。

峰值显存增加约 86 MiB，因为分页池按 64 个 Block 预分配：

```text
64 * 1.75 MiB = 112 MiB KV Block Pool
```

当前工作负载很短，旧后端本来也只需要少量 Cache，因此固定池看起来更贵。服务系统中预分配
不是纯浪费：它把显存容量变成明确、稳定、无碎片的 Block Budget，避免运行时频繁申请显存。

这组 3% 提升不是 Paged Attention 的最终收益。9C 的核心验收指标是移除 `read_batch`
产生的 gather/padding，并重新测量 Scheduler 混合长度工作负载。

#### 本阶段验证

- 分页引擎输出与逐条 cached generation 一致
- Block 容量不足会让 FIFO 请求等待
- Decode Batch 的多请求 reserve 具有原子性
- Prefill Padding 不写入物理池
- Decode 每轮只写一个新增 K/V
- 非连续物理块可组成正确 Attention 历史
- 所有请求结束后物理块全部回收

### 2026-06-06：第 9C 步，Paged Attention 直接读取物理块

9B 已经把 KV 长期存储变成分页物理池，但每轮 Decode 前仍调用 `read_batch()`：

```text
离散物理块 -> Gather -> 左 Padding 连续 Tensor -> SDPA
```

这会产生一份临时历史 Cache，长度不同的请求还要补齐到 Batch 最大长度。9C 把 Decode
数据流改为：

```text
Query + BlockTable + 物理 KV Block 池
              |
              v
逐 Block 在线 softmax
              |
              v
Attention Output
```

#### 1. 新增 Paged Attention 元数据

`PagedAttentionMetadata` 只保存三类引用：

- 整个 Key 物理池
- 整个 Value 物理池
- 当前 Decode Batch 每条请求的 BlockTable

它不复制 K/V。Attention 根据自己的 `layer_index` 读取物理池当前层，再按照
`physical_block_ids` 的逻辑顺序扫描 Block。物理编号即使是 `[7, 2, 19]`，逻辑序列
仍然是第 0、1、2 块。

#### 2. 为什么不能对每个 Block 单独做普通 softmax

假设完整历史被分成 A、B 两块。分别计算：

```text
softmax(scores_A)
softmax(scores_B)
```

再把两个输出相加是错误的，因为两块使用了不同分母。完整 Attention 需要所有 token
共同参与同一次归一化。

9C 为每个 Query Head 维护三个在线状态：

```text
m   = 已扫描 score 的最大值
l   = sum(exp(score - m))
acc = sum(exp(score - m) * value)
```

读到新 Block 后，先计算该块最大值 `block_max`，再更新：

```text
new_m   = max(m, block_max)
old_mul = exp(m - new_m)
weight  = exp(block_score - new_m)

new_l   = l * old_mul + sum(weight)
new_acc = acc * old_mul + sum(weight * block_value)
```

全部 Block 和当前 token 扫描结束后：

```text
attention_output = acc / l
```

`old_mul` 会把旧状态换算到新的最大值基准，因此结果与一次性对完整 scores 做 softmax
数学等价，同时保持数值稳定。中间 K/V 和 scores 的大小只与 `block_size` 有关，不再
与完整上下文长度成正比。

#### 3. 当前 token 为什么不先写入物理池

Decode 的当前 token 也必须看到自己的 Key/Value，但模型的每一层要先计算出该层 K/V，
Engine 才能在整次前向结束后统一写回。因此本轮顺序是：

```text
1. 保存 reserve 前的旧 BlockTable 快照
2. 原子 reserve_many，为本轮新 token 保证物理槽位
3. Attention 扫描旧 BlockTable
4. 把当前层刚算出的 K/V 当作最后一个长度为 1 的临时块
5. 模型前向结束后，把各层当前 K/V 写入预留槽位
```

`BlockTable` 是不可变 dataclass。即使 BlockManager 在第 2 步生成了包含新 token 的表，
Attention 持有的旧快照仍只描述已经初始化的历史，因此不会读取未写入显存。

#### 4. RoPE 位置如何确定

9B 的连续 Cache 可以从 Tensor 长度推断当前位置。9C 不再传 `past_key_values`，所以
Engine 显式构造：

```text
position_id = old_block_table.num_tokens
```

每条请求拥有自己的位置，不受 Batch 中最长请求和 Padding 影响。

#### 5. GQA 为什么不需要复制 KV Head

Qwen3-1.7B 的 Query Head 多于 KV Head。旧 SDPA 路径通过 `repeat_interleave` 临时复制
K/V Head。分页实现把 Query reshape 为：

```text
[num_kv_heads, queries_per_kv, head_dim]
```

同一个 KV Head 直接服务它对应的一组 Query Head，避免在每个 Block 上真的复制 K/V。

#### 6. 为什么 9C 参考版反而更慢

标准混合长短请求 Benchmark：

```text
请求数       8
最大并发     4
Prompt       18 token
生成上限     [4, 16, 4, 16, 4, 16, 4, 16]
Block Pool   64 blocks
Block Size   16 tokens
```

结果：

| Attention 路径 | 总输出吞吐 | 峰值显存 |
|---|---:|---:|
| 9B Gather + SDPA | 56.93 tokens/s | 4026.0 MiB |
| 9C PyTorch 在线 softmax | 27.60 tokens/s | 4008.2 MiB |

9C 吞吐只有 9B 的 `0.48x`，但峰值显存降低约 17.8 MiB。原因是当前参考实现包含：

- Python 层的请求循环
- Python 层的物理 Block 循环
- 每个 Block 多次 `einsum`、`exp`、`sum` GPU kernel 启动
- Attention 各步骤之间没有 Kernel Fusion

9B 虽然做了 Gather，但随后能进入 PyTorch 已高度优化的 fused SDPA。GPU 更擅长少量大
Kernel，而不是大量由 Python 发射的小 Kernel。

因此 9C 的完成标准不是“纯 PyTorch 马上加速”，而是：

- Attention 已真正消费 BlockTable
- Decode 不再调用 `read_batch`
- 不再创建带 Padding 的完整历史 KV Batch
- 在线 softmax 与连续 Attention 数学对齐
- Engine 的预留、写回、释放生命周期保持正确

下一阶段性能优化要把“请求循环 + Block 循环 + 在线 softmax”融合进一个 Triton/CUDA
Kernel。届时 GPU Thread Block 可协作加载一小块 K/V，使用寄存器或 shared memory
归约 `m/l/acc`，这才是生产级 Paged Attention 的性能来源。

#### 7. 如何运行和对比

默认 9C：

```powershell
& $PYTHON -m toyvllm continuous --cache-backend paged `
    --paged-attention paged --num-kv-blocks 64 --block-size 16 `
    --max-num-seqs 4 "你好" "解释 KV Cache"
```

保留的 9B A/B 对照：

```powershell
& $PYTHON -m toyvllm continuous --cache-backend paged `
    --paged-attention gather --num-kv-blocks 64 --block-size 16 `
    --max-num-seqs 4 "你好" "解释 KV Cache"
```

Benchmark：

```powershell
& $PYTHON bench.py --backend paged --batch-size 4 --num-requests 8 `
    --short-new-tokens 4 --max-new-tokens 16 --warmup 1 --iterations 3 `
    --save benchmarks/results.jsonl
```

### 2026-06-06：第 10A 步，第一个 Triton Paged Attention Kernel

9C 的 PyTorch 参考实现已经证明了分页 Attention 的数学和生命周期，但它在 Python 中
循环请求、物理 Block，并为每个小操作启动 GPU Kernel。10A 的目标很克制：

> 不追求生产级全功能，先把单 token Decode 的 Block 扫描和在线 softmax 放进一个
> Triton Kernel。

#### 1. 环境选择

当前环境保持：

```text
Python          3.10
PyTorch         2.2.2+cu121
GPU             RTX 4060 Ti 8GB
NVIDIA Driver   591.74
Triton Windows  3.1.0.post17
```

这里只安装 `triton-windows`，没有升级已经验证过的 PyTorch：

```powershell
& $PYTHON -m pip install triton-windows==3.1.0.post17
```

项目把它放在可选依赖 `triton-windows` 中。没有 Triton 时，CLI 的 `auto` 会回退到
PyTorch `paged` 后端；显式指定 `triton` 则会给出缺少依赖的错误。

#### 2. 一个 Triton Program 负责什么

Kernel 使用二维 Grid：

```text
grid = [batch_size, num_query_heads]
```

因此一个 Program 负责：

```text
一条请求 + 一个 Query Head + 当前模型层
```

它会：

1. 把当前 Query Head 加载进寄存器
2. 根据 GQA 映射算出对应的 KV Head
3. 读取该请求的 `context_length`
4. 顺序遍历 GPU BlockTable 中的逻辑块
5. 按物理块编号加载 K/V
6. 在寄存器中更新 `running_max/running_sum/accumulator`
7. 扫描结束后写出一个 Attention Head

Python 不再循环请求和物理块。模型每层只发射一次 Triton Kernel。

#### 3. BlockTable 为什么要变成 GPU Tensor

9C 的 `BlockTable` 是 CPU dataclass，适合 Scheduler 修改和单元测试。但 GPU Kernel
不能直接读取 Python 对象，因此 Engine 每轮 Decode 把本 Batch 的表打包为：

```text
block_table_tensor : [batch, max_num_blocks], int32
context_lengths    : [batch], int32
```

较短请求用 `-1` 补齐 BlockTable。Kernel 不依赖 `-1` 本身，而是根据
`context_lengths` 产生 token mask，所以不会读取补位地址。

这一步仍有小额 CPU 到 GPU 元数据复制。生产引擎通常维护常驻 GPU 的 BlockTable，
后续只增量修改变化部分；当前版本先保持数据流直观。

#### 4. 为什么先把当前 token 放入在线 softmax

Kernel 初始化时直接计算当前 token 的 score，并设置：

```text
running_max = current_score
running_sum = 1
accumulator = current_value
```

好处有两个：

- 当前 token 本来就必须参与因果 Attention
- 状态一开始就是有限值

如果从 `running_max=-inf` 开始，而某个补齐 Block 全部无效，就可能计算
`-inf - -inf` 产生 NaN。用当前 token 初始化后，无效 Block 的权重自然为 0。

#### 5. GQA 映射

每个 Program 已知自己的 `query_head`，对应 KV Head 为：

```text
kv_head = query_head // queries_per_kv
```

因此不需要 `repeat_interleave` 复制 K/V。当前简单 Kernel 仍会让属于同一 KV Head 的
多个 Query Program 各自读取一遍 K/V，后续可以让一个 Program 同时处理一组 Query Head。

#### 6. JIT 编译与缓存

Triton 第一次遇到新形状时需要 JIT 编译。冷启动不能计入稳态吞吐 Benchmark，所以：

- `bench.py` 先执行 Warmup
- `layer_index` 是运行时参数，28 层共享同一份 Kernel
- `head_dim/block_size/max_blocks` 才作为编译期常量
- JIT 缓存默认写入系统临时目录 `toy-vllm-triton-cache`

第一次真实模型命令包含编译时只有 `0.89 tokens/s`；缓存命中后，同类短任务恢复到
`18.50 tokens/s`。这说明测 GPU Kernel 必须区分冷启动和稳态。

#### 7. 标准 Benchmark

工作负载保持不变：

```text
请求数       8
最大并发     4
Prompt       18 token
生成上限     [4, 16, 4, 16, 4, 16, 4, 16]
Block Pool   64 blocks
Block Size   16 tokens
Warmup       1
Iterations   3
```

结果：

| Attention 路径 | 总输出吞吐 | 峰值显存 |
|---|---:|---:|
| 9B Gather + SDPA | 57.31 tokens/s | 4026.0 MiB |
| 9C PyTorch 在线 softmax | 26.76 tokens/s | 4008.2 MiB |
| 10A Triton Paged Attention | 77.69 tokens/s | 4008.2 MiB |

Triton 相对 PyTorch 参考版提升 `2.90x`，相对 Gather + SDPA 提升 `1.36x`。它保留了
不构造连续历史 Cache 的显存收益，同时通过 Kernel Fusion 消除了 Python 小算子开销。

#### 8. 当前 Kernel 的限制

为了保持第一版容易理解，当前只支持：

- 单 token Decode，不处理 Prefill
- `head_dim <= 256`
- 一个 Program 处理一个 Query Head
- 逻辑 Block 在一个 Program 内串行扫描
- 固定 `num_warps=4`，尚未 autotune
- 没有 Split-K、Tensor Core 专门优化或跨 Query Head 共享 KV

这不是 vLLM 生产 Kernel 的完整复刻，但已经建立了最关键的连接：

```text
Scheduler BlockTable
        -> GPU 元数据
        -> Triton Paged Attention
        -> 在线 softmax
        -> 模型输出
```

#### 9. 运行方式

自动选择 Triton，缺失时回退：

```powershell
& $PYTHON -m toyvllm continuous --cache-backend paged `
    --paged-attention auto --num-kv-blocks 64 --block-size 16 `
    --max-num-seqs 4 "你好" "解释 KV Cache"
```

三路径 Benchmark：

```powershell
& $PYTHON bench.py --backend paged --batch-size 4 --num-requests 8 `
    --short-new-tokens 4 --max-new-tokens 16 --warmup 1 --iterations 3 `
    --save benchmarks/results.jsonl
```

### 2026-06-06：第 10B 步，优化不能只盯着 Attention Kernel

10A 把 Paged Attention 从大量 PyTorch 小算子融合为 Triton Kernel 后，继续优化前先
检查整个 Decode 数据流，而不是默认瓶颈仍在 Attention 内部。

#### 1. 第一次尝试：按 KV Head 合并 GQA Query

10A 的 Grid 是：

```text
[batch, query_heads]
```

Qwen3 使用 GQA，多组 Query Head 会共享同一个 KV Head。直觉上的优化是改成：

```text
[batch, kv_heads]
```

一个 Triton Program 同时计算 `queries_per_kv` 个 Query Head。这样同一块 K/V 只加载
一次，再广播给组内 Query。

数学和输出都正确，但标准 Benchmark 结果是：

```text
逐 Query Head Kernel   84.72 tokens/s
GQA 分组 Kernel        83.59 tokens/s
倍率                   0.99x
```

它没有加速。原因是：

- 当前上下文很短，K/V 带宽还不是主要瓶颈
- 一个 Program 同时保存多组 Query、`m/l/acc`，寄存器压力更高
- 三维广播与归约比 10A 的二维计算更复杂
- 两种实现每层仍只发射一次 Grid，减少 Program 数不等于减少 Kernel launch 数

所以默认 `triton` 保留更简单、更快的逐 Query Head Kernel，分组版仅作为
`triton-grouped` 实验后端。这个负结果很重要：减少理论访存不保证真实 GPU 更快，
必须结合占用率、寄存器和工作负载测量。

#### 2. 真正瓶颈：Decode K/V 写回

Attention 算完后，每层都会返回当前 token 的 Key/Value。旧实现按以下顺序写回：

```text
for request in batch:
    for layer in 28 layers:
        write key
        write value
```

Batch 为 4 时，一轮 Decode 会产生大量细碎 Tensor 索引和赋值。即使每次写的数据很少，
GPU Kernel 启动与 Python 调度成本仍然存在。

独立微基准：

```text
Qwen3 层数       28
Batch            4
KV Heads         8
Head Dim         128

旧逐项写回       6.99 ms/轮
向量化写回       0.48 ms/轮
微基准倍率       14.6x
```

#### 3. 向量化写回的数据布局

每一层返回：

```text
[batch, kv_heads, sequence, head_dim]
```

Decode 只需要最后一个 token。先去掉 sequence 维，再沿层维 stack：

```text
key_rows   [num_layers, batch, kv_heads, head_dim]
value_rows [num_layers, batch, kv_heads, head_dim]
```

本轮每条请求的新槽位已经由 `reserve_many` 返回。把它们整理成：

```text
block_ids [batch]
offsets   [batch]
```

物理池布局为：

```text
[num_layers, num_blocks, block_size, kv_heads, head_dim]
```

因此可以直接完成两次批量赋值：

```python
key_cache[:, block_ids, offsets] = key_rows
value_cache[:, block_ids, offsets] = value_rows
```

Key 一次、Value 一次。层和请求不再由 Python 循环写入。

#### 4. 为什么微基准 14.6 倍，整模型只有 1.11 倍

标准工作负载最终结果：

| 路径 | 总输出吞吐 | 峰值显存 |
|---|---:|---:|
| Gather + SDPA | 61.02 tokens/s | 4026.4 MiB |
| PyTorch Paged Attention | 27.69 tokens/s | 4008.2 MiB |
| Triton + 旧逐项写回 | 76.25 tokens/s | 4008.2 MiB |
| Triton + 向量化写回 | 84.72 tokens/s | 4008.2 MiB |

向量化写回让整体吞吐提升 `1.11x`。微基准只测写回，而整模型还包含：

- Q/K/V 和输出线性层
- MLP
- RMSNorm 与 RoPE
- Triton Paged Attention
- LM Head 和采样
- Scheduler 与 Python 控制流

优化一个占总时间一部分的阶段，不可能让整条流水线也提高 14.6 倍。这就是
Amdahl 定律在推理系统中的直接体现。

#### 5. 为什么没有先优化 BlockTable 上传

同样做了独立测量：

```text
每轮构造并上传 Triton BlockTable 元数据约 40 微秒
```

它确实可以通过常驻 GPU Workspace 继续降低，但相比旧写回的约 6.99 ms 小得多。因此
本轮优先修复写回。优化顺序应该由测量决定，而不是由代码看起来是否“高级”决定。

#### 6. 正确性与参考路径

`write_decode_batch(..., vectorized=False)` 保留旧实现，专门用于 A/B Benchmark。
默认使用向量化路径，并增加测试验证两种实现写出的整个物理池完全一致。

本阶段验证：

- PyTorch、逐 Query Triton、GQA 分组 Triton 输出一致
- 不同请求长度和 BlockTable 补位正确
- 旧写回与向量化写回物理池内容一致
- 真实 Qwen3-1.7B 混合长度工作负载通过
- 所有请求结束后 Block 全部回收

下一步可以继续测量：

- BlockTable GPU 常驻 Workspace
- `num_warps` 和 Block Size autotune
- 把 K/V 写回也改成专门 Triton Kernel，避免 `torch.stack` 临时 Tensor
- 更长上下文下 GQA 分组和 Split-K 是否开始有收益

### 2026-06-06：第 10C 步，常驻 BlockTable 与带保护的 Kernel Autotune

10B 已经测得 BlockTable 元数据约几十微秒，远小于旧 K/V 写回，但它仍存在两个系统
问题：

- 每轮 Decode 创建新的 CUDA BlockTable Tensor
- Kernel 固定使用 `num_warps=4`，没有根据形状选择配置

10C 把这两项补齐，同时保留严格 A/B 路径验证真实收益。

#### 1. 常驻 GPU Workspace 的结构

`PagedAttentionWorkspace` 在 Engine 初始化时一次性分配：

```text
CPU block_tables      [max_num_seqs, num_kv_blocks] int32
CPU context_lengths   [max_num_seqs]                 int32
GPU block_tables      [max_num_seqs, num_kv_blocks] int32
GPU context_lengths   [max_num_seqs]                 int32
```

每轮返回的不是新 Tensor，而是有效区域的视图：

```text
gpu_block_tables[:batch_size, :max_blocks]
gpu_context_lengths[:batch_size]
```

测试会记录 `data_ptr()`，多次更新请求顺序、长度和物理块后，GPU Workspace 地址保持
不变。这说明 CUDA allocator 不再参与每轮 BlockTable 生命周期。

#### 2. 为什么不能逐格写预分配 Tensor

第一版 Workspace 对 Python tuple 做双重循环：

```python
for row, table in enumerate(tables):
    for column, block_id in enumerate(table.physical_block_ids):
        cpu_workspace[row, column] = block_id
```

微基准反而从瞬时路径的约 `55 us` 变成 `135 us`。预分配只消除了 allocator，却引入
大量 Python 标量赋值。常驻不自动等于更快。

最终实现缓存每一行上次的 `physical_block_ids`：

```text
Block 映射不变：不上传该行
跨 Block 边界：更新 CPU 行，并覆盖 GPU 有效切片
Batch 重排：只更新发生变化的行
```

普通 Decode 每轮只增加 `num_tokens`，物理块通常连续十几轮都不变化，因此 BlockTable
真正变成长期驻留数据。

#### 3. 复用 position_ids 作为 context length

Decode 的 RoPE 位置就是历史长度：

```text
position_id = block_table.num_tokens = context_length
```

旧数据流为同一组整数创建两份 GPU Tensor：

```text
position_ids       -> RoPE
context_lengths    -> Paged Attention
```

现在 Engine 只上传 `position_ids`，并把 `position_ids[:, 0]` 直接交给 Triton Kernel。
BlockTable 映射未变化时，Paged Attention 不再额外上传任何元数据。

#### 4. Workspace 为什么没有明显提高整模型吞吐

变化长度微基准中，增量 Workspace 曾把独立元数据操作从约 `56.8 us` 降到 `46.6 us`。
继续把 position/context 合并后，整条 Engine 路径的主要成本变成原本就需要的
`position_ids` 创建和上传；BlockTable 是否瞬时创建的差距已经很小。

最终五次迭代结果：

```text
瞬时 BlockTable + 4 warps   87.00 tokens/s
常驻 BlockTable + 4 warps   86.09 tokens/s
```

另一轮为 `83.72 vs 84.71 tokens/s`。方向会随系统噪声变化，合理结论是端到端基本持平，
而不是宣称 1% 的不稳定加速。

常驻 Workspace 的当前价值主要是：

- CUDA Tensor 地址稳定
- 避免高并发服务中持续的小对象分配
- 为 CUDA Graph 捕获准备固定地址
- 为后续增量 GPU BlockTable 更新建立接口

#### 5. Autotune 如何工作

首次遇到新的形状 key：

```text
(device, dtype, batch, query_heads, head_dim, block_size, max_blocks)
```

启动器依次测试：

```text
num_warps = 1, 2, 4, 8
```

每个候选使用 20 次 warmup 和 100 次正式测量，并取 median。结果保存在 Python 进程内，
后续 28 层和后续 Decode 轮直接复用选择；Triton 编译产物仍由磁盘 JIT Cache 保存。

`triton-fixed` 后端始终使用 4 warps，专门用于 A/B。

#### 6. 为什么需要 10% 基线保护

最初直接选择微基准最快配置时，候选只比 4 warps 快约 0.6% 到 3%，但端到端吞吐反而
下降约 2%。RTX 4060 Ti 在 Windows WDDM 下会受到频率、缓存和调度抖动影响，十几微秒
Kernel 的微小差距不够可靠。

因此最终规则是：

```text
候选延迟 <= 4-warps 延迟 * 0.90：采用候选
否则：继续使用 4 warps
```

标准工作负载中，微基准经常认为 1 warp 略快，但没有达到 10% 门槛，保护后统一选择
4 warps。最终：

```text
常驻 + 固定 4 warps    86.09 tokens/s
常驻 + 保护 autotune   85.97 tokens/s
```

两者基本一致，说明 autotune 没有制造回退。对于更长上下文或不同 GPU，如果某个候选
优势足够明确，它仍然可以自动生效。

#### 7. 最终标准 Benchmark

| 路径 | 总输出吞吐 | 峰值显存 |
|---|---:|---:|
| Gather + SDPA | 63.63 tokens/s | 4026.4 MiB |
| 瞬时 BlockTable + 4 warps | 87.00 tokens/s | 4008.2 MiB |
| 常驻 BlockTable + 4 warps | 86.09 tokens/s | 4008.2 MiB |
| 常驻 BlockTable + 保护 autotune | 85.97 tokens/s | 4008.2 MiB |

10C 没有得到新的显著吞吐倍数，但完成了生产推理引擎很重要的两类基础设施：

- 固定地址、增量更新的 GPU 调度元数据
- 可观察、有基线保护、不会强行采用噪声赢家的 Kernel 配置选择

下一步更值得测量的是 CUDA Graph：Workspace 地址稳定后，可以尝试捕获固定 Batch
形状的一整个 Decode 前向，减少 28 层模型中大量 Kernel 的 Python launch 开销。

### 2026-06-11：第 10D 步，长上下文 Serving 吞吐与时延总基线

在继续实现 chunked prefill 之前，先建立一套更接近在线服务的统一基准。否则后面即使
吞吐发生变化，也无法判断是 Prefill、Decode、调度还是测试口径造成的。

#### 1. 工作负载

```text
请求数：8
Prompt token：[128, 256, 512, 768, 128, 256, 512, 768]
输出上限：[4, 8, 4, 8, 4, 8, 4, 8]
最大并发 BS：[1, 2, 4, 8]
迭代：每个配置预热 1 次，正式测量 2 次
```

Prompt 被 tokenizer 构造成精确 token 长度，不使用 Padding 冒充输入 token。测试关闭
EOS，让 Gather 和 Triton 完成完全相同数量的生成工作。每个 BS 都单独预热，因为
Triton JIT/autotune 的缓存 key 包含 Batch 和输入形状；只预热 BS1 会污染较大 BS 的
首次 TTFT。

#### 2. 新增指标

- `TTFT`：请求进入 Engine 到产生第一个 token 的时间
- `TPOT`：首 token 之后，每个新增 token 的平均时间
- `E2E`：请求从进入 Engine 到完成的总时间
- `total tok/s`：输入 token 与输出 token 的总处理吞吐
- `out tok/s`：用户真正看到的生成吞吐
- `prefill step` / `decode step`：定位耗时来自哪一阶段

Engine 为每个请求记录首 token 和完成时刻。分位数使用线性插值实现，不引入 NumPy。

#### 3. RTX 4060 Ti 实测

| 后端 | BS | 总吞吐 tok/s | 输出 tok/s | TTFT P50/P95 ms | TPOT P50/P95 ms | E2E P95 ms | 峰值显存 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gather | 1 | 1682.5 | 23.9 | 901.1 / 1709.0 | 37.6 / 44.2 | 1991.4 | 4510.7 |
| Triton | 1 | 2161.5 | 30.7 | 700.7 / 1361.2 | 27.6 / 30.4 | 1553.6 | 4504.7 |
| Gather | 2 | 2459.5 | 35.0 | 488.4 / 1099.0 | 43.3 / 63.9 | 1370.4 | 4683.4 |
| Triton | 2 | 2958.5 | 42.1 | 440.2 / 927.8 | 34.5 / 54.7 | 1115.8 | 4504.7 |
| Gather | 4 | 2797.6 | 39.8 | 457.3 / 898.4 | 56.4 / 77.0 | 1201.9 | 5035.4 |
| Triton | 4 | 3416.8 | 48.6 | 435.0 / 774.5 | 39.5 / 57.8 | 986.6 | 4836.5 |
| Gather | 8 | 2894.0 | 41.1 | 777.5 / 780.2 | 62.4 / 67.6 | 1182.3 | 5729.2 |
| Triton | 8 | 3460.5 | 49.2 | 777.3 / 777.6 | 28.9 / 29.5 | 983.3 | 5340.5 |

#### 4. 如何读这张表

Triton 在所有 BS 上都比 Gather 更快。以 BS8 为例，总吞吐提高约 `19.6%`，输出吞吐
也从 `41.1` 提高到 `49.2 tokens/s`，峰值显存少约 `389 MiB`。

但最大吞吐不等于最佳服务体验：

```text
Triton BS4：TTFT P50 = 435 ms，total = 3417 tok/s
Triton BS8：TTFT P50 = 777 ms，total = 3460 tok/s
```

BS 从 4 增加到 8，总吞吐只增加约 `1.3%`，TTFT P50 却增加约 `79%`。原因是当前
Scheduler 把 8 条不同长度 Prompt 拼成一个完整 Prefill Batch，短 Prompt 也必须等待
最长的 768-token Prompt 完成。BS8 的 Prefill step 约 `734 ms`，在此期间所有 Decode
都被阻塞，这就是典型的队头阻塞。

#### 5. 为什么下一步先做 Chunked Prefill

Chunked prefill 给每轮设置 token budget，把长 Prompt 切成多个小块：

```text
当前：一次处理完整 768-token Prefill，然后恢复 Decode
分块：每轮只处理一部分 Prefill，并在轮次之间穿插 Decode
```

它的首要目标不是让单个 Prefill 算得更快，而是限制一次 Prefill 最长占用 GPU 的时间，
降低已有请求的 TPOT 和新请求的 TTFT 尾延迟。完成分块执行后，再实现 PD 混合调度：
Scheduler 才能在同一轮 token budget 内决定多少额度给 Prefill、多少给 Decode。

这组结果将作为验收基线。下一阶段重点比较：

- Triton BS8 的 TTFT P50/P95 是否低于当前 `777 ms`
- Prefill 插入时，运行中请求的 TPOT P95 是否保持稳定
- 吞吐是否接近当前 `3460 tok/s`，而不是用大幅吞吐下降换延迟

### 2026-06-11：第 10E 步，重构 Engine 子包

在加入 chunked prefill 和 PD 混合调度前，先整理 Engine 的代码边界。原来四个相关模块
散落在 `toyvllm/` 根目录：

```text
engine.py
scheduler.py
sequence.py
block_manager.py
```

其中 `engine.py` 已超过 800 行。后续如果继续直接加入 token budget、Prefill 分块状态和
混合调度决策，模型执行、请求状态与资源分配会更难区分。

#### 1. 新目录结构

```text
toyvllm/engine/
├── llm_engine.py
├── scheduler.py
├── sequence.py
├── block_manager.py
└── __init__.py
```

各模块的职责是：

- `sequence.py`：描述单条请求现在处于什么状态，不持有 GPU Tensor
- `scheduler.py`：决定请求何时从 WAITING 进入 RUNNING，何时完成
- `block_manager.py`：管理逻辑 token 到物理 KV Block 的映射
- `llm_engine.py`：把调度结果转换成 Tensor，执行 Prefill/Decode 并管理 GPU 资源
- `__init__.py`：提供稳定的公共导入入口

调用方仍然使用：

```python
from toyvllm.engine import (
    ContinuousBatchEngine,
    PagedContinuousBatchEngine,
    Scheduler,
)
```

因此这是内部结构重构，不改变 CLI 和已有调用方式。

#### 2. 为什么主文件叫 llm_engine.py

这个命名与 vLLM 的 Engine 层含义一致：它不是模型本身，也不是单独的 Scheduler，而是
协调请求状态、KV Cache 和模型执行的上层编排器。

当前没有立即把连续缓存和分页缓存拆成两个完全独立的执行器文件。原因是分页实现直接
继承连续引擎的请求生命周期、采样、计时和 `run/step` 循环。现在强拆会制造大量转发
接口。等 chunked prefill 明确“执行一个 token chunk”的契约后，再抽出 executor 边界
更合理。

#### 3. 为什么 __init__.py 使用延迟导出

依赖关系中存在一条环：

```text
llm_engine -> kv_cache -> engine.block_manager
```

如果 `engine/__init__.py` 一加载就立即导入 `llm_engine`，那么 `kv_cache` 为了取得
`BlockTable` 再进入 engine 包时，可能遇到尚未初始化完成的 `llm_engine`。

现在通过模块级 `__getattr__` 按名称延迟加载：

```text
导入 BlockManager：只加载轻量控制面
导入 ContinuousBatchEngine：此时才加载模型执行路径
```

这还带来一个教学和测试上的好处：Scheduler、Sequence、BlockManager 的 CPU 单元测试
不必顺带导入模型和 CUDA 执行代码。

#### 4. 这次重构的性能影响

没有修改任何 Prefill、Decode、Paged Attention 或 Scheduler 算法，所以预期性能变化
为零。它的收益是为下一步提供清楚的修改位置：

```text
Prefill 已处理到哪里        -> Sequence
本轮 Prefill token budget  -> Scheduler
Chunk 需要哪些 KV Block    -> BlockManager
如何执行一个 Prompt chunk  -> LLM Engine
```

验证结果：

```text
55 个单元测试通过
Python compileall 通过
真实 Qwen3-1.7B Paged Triton 连续批处理通过
请求结束后 64/64 KV Block 全部回收
```

### 2026-06-11：第 10F 步，Chunked Prefill 与 PD 混合调度

10D 的基线暴露了完整 Prefill 的队头阻塞：

```text
BS8 单次 Prefill 约 734 ms
在它结束前，8 条请求都拿不到首 token
```

本阶段把 Prompt 从“一次全部计算”改成“按 token budget 分多轮计算”，并允许同一调度轮
同时出现 Decode 和 Prefill。

#### 1. Sequence 为什么需要 Prefill 游标

过去 RUNNING 请求默认已经完成完整 Prompt。分块后，RUNNING 内部还要区分：

```text
num_prompt_tokens_computed < prompt_length  -> PREFILLING
num_prompt_tokens_computed = prompt_length  -> DECODING
```

`Sequence` 新增：

- `num_prompt_tokens_computed`
- `num_prompt_tokens_remaining`
- `is_prefill_complete`
- `next_prompt_chunk()`
- `advance_prefill()`

游标只在模型前向和 KV 写回都成功后推进。Scheduler 也禁止在 Prefill 完成前提交生成
token，避免“状态说已经完成，但缓存还没写好”的双重事实。

#### 2. Token budget 如何分配

每轮先统计 Decode：

```text
decode_tokens = 正在 Decode 的请求数
prefill_budget = max_num_batched_tokens - decode_tokens
```

每条 Decode 请求固定消耗 1 token。剩余预算按正在 Prefill 的请求数动态计算 fair share：

```text
fair_share = ceil(remaining_budget / remaining_prefill_requests)
chunk = min(prompt_remaining, max_prefill_chunk_size, fair_share)
```

这样第一条 768-token Prompt 不会独占整轮预算。短 Prompt 没用完的份额会自动留给后面的
请求。

调度轨迹示例：

```text
step=03 decode=[]  prefill=[(0, 1), (1, 4)]
step=04 decode=[0] prefill=[(1, 4)]
step=05 decode=[0] prefill=[(1, 4)]
```

请求 0 已经开始 Decode 时，请求 1 仍在 Prefill，这就是当前版本的 PD 混合轮。

#### 3. 为什么还要预留完整 Prompt 的 Block

当前 BlockManager 在接纳请求时：

```text
物理容量：按完整 Prompt 预留
有效 token：从 0 开始
```

`BlockTable.num_blocks` 可以大于 `ceil(num_tokens / block_size)`。每完成一个 chunk，
`reserve()` 只推进有效 token 数；Paged Attention 永远只读取 `num_tokens` 范围，因此
不会看到尚未写入的槽位。

预留完整容量不是最终最省显存的方案，但它避免一个教学版调度死锁：

```text
多个半完成 Prompt 占满运行槽
所有空闲 Block 又被它们的局部缓存耗尽
没有请求完成，因此没有 Block 可以释放
```

以后加入抢占和换出后，可以改成真正按 chunk 增量分配。

#### 4. 不同历史长度如何组成一个 Prefill Batch

当前 Triton Paged Attention 只支持单 token Decode，多 token Prefill 暂时使用可靠的
PyTorch SDPA 路径：

1. 按 BlockTable 从物理池读取每条请求已有历史。
2. 历史 KV 左补齐到 `max_history`。
3. 当前 Prompt chunk 左补齐到 `max_chunk`。
4. 拼接 history mask 与 current mask。
5. 使用带偏移因果 mask 的模型前向。
6. 只取返回 Cache 最后的 `chunk_size` 个 K/V 写回新物理槽位。

示意：

```text
请求 A：history=3, chunk=2
请求 B：history=1, chunk=4

History KV: [pad H H H]  [pad pad pad H]
Current   : [pad pad C C] [C C C C]
Mask      : [0 1 1 1 | 0 0 1 1]
            [0 0 0 1 | 1 1 1 1]
```

RoPE 位置由完整 mask 的累积和计算，Padding 不占逻辑位置。58 个测试已经覆盖不同历史、
不同 chunk 长度混合 Batch，并与非分块生成逐 token 对齐。

#### 5. Decode 优先与 mixed budget

当前混合轮执行顺序是：

```text
先执行一个 Decode Batch
再执行一个 Prefill Chunk Batch
```

它还不是 vLLM 更深度融合的单次 token batch。为了控制第二次 Prefill 调用对下一轮
Decode 的阻塞，可以设置：

```text
--max-mixed-prefill-tokens 128
```

该参数只在本轮已有 Decode 时限制 Prefill 总量。不设置时优先追求吞吐和 TTFT；设置较小
值时可以降低单次干扰，但可能显著拖慢剩余长 Prompt。

#### 6. RTX 4060 Ti 实测

工作负载仍使用 10D 的 8 请求：

```text
Prompt：[128, 256, 512, 768] × 2
输出：[4, 8] 交替
BS=8
```

| 配置 | 总吞吐 tok/s | TTFT P50/P95 ms | TPOT P95 ms | E2E P95 ms | 显存 MiB |
|---|---:|---:|---:|---:|---:|
| 完整 Prefill | 3460.5 | 777 / 778 | 29.5 | 983 | 5340.5 |
| budget=512, chunk=128 | 3640.7 | 473 / 754 | 84.7 | 937 | 4819.9 |
| budget=1024, chunk=128 | 4205.1 | 336 / 629 | 92.1 | 824 | 4753.4 |
| 1024/128，mixed=128 | 2288.7 | 756 / 1314 | 77.3 | 1520 | 4796.9 |

推荐当前 RTX 4060 Ti 教学配置：

```text
max_num_batched_tokens = 1024
max_prefill_chunk_size = 128
max_mixed_prefill_tokens = 不限制
```

相对完整 Prefill：

- 总吞吐提高约 `21.5%`
- TTFT P50 降低约 `56.7%`
- TTFT P95 降低约 `19.1%`
- E2E P95 降低约 `16.2%`
- 峰值显存减少约 `587 MiB`

TPOT P95 从 `29.5 ms` 增加到 `92.1 ms`，不能隐藏。完整 Prefill 的 TPOT 只在所有 Prompt
结束后开始计时，所以不会包含 Prefill 干扰；分块后早完成的请求更早开始输出，也会真实
承受后续长 Prompt 的干扰。虽然 TPOT 变差，TTFT 和 E2E 都明显改善。

#### 7. 当前实现与完整 vLLM 的差距

- Prefill 历史暂时从 Paged KV 池 gather 成连续 Tensor
- Decode 和 Prefill 还是同一轮中的两次模型调用
- Prompt Block 容量在 admission 时完整预留
- 没有抢占、换出和按优先级重排

下一步最有价值的是多 token Paged Prefill Kernel：直接按 BlockTable 读取历史，避免每个
chunk gather 连续 KV；之后再考虑把 Decode token 与 Prefill token 打包成统一执行计划。

### 2026-06-12：第 10G 步，根据 GPU 显存自动规划 KV Block

此前所有 Paged Engine 都要求手动指定：

```text
--num-kv-blocks 256
```

这个数字过小时，模型虽然能运行，但大量显存没有转化为可服务的上下文容量；数字过大时，
KV 池会在 Engine 初始化时一次性申请失败，或者只剩很少显存给 Prefill 激活，首次请求
才 OOM。

#### 1. 为什么必须在模型加载后规划

不能只读取显卡标称 8 GB，然后减去配置文件里的权重大小。真实占用还包括：

- CUDA Context 和库工作区
- PyTorch allocator 当前保留的显存
- RoPE buffer 等非权重 Tensor
- Windows 桌面和其他 GPU 进程
- 当前模型实际 dtype 与权重共享方式

因此规划发生在真实权重加载完成后。Engine 先调用：

```python
torch.cuda.empty_cache()
free_bytes, total_bytes = torch.cuda.mem_get_info(device)
```

`empty_cache()` 只释放 allocator 中没有活跃 Tensor 引用的缓存，不会释放模型权重。
随后得到的是当前时刻整块 GPU 的真实 free/total 快照，其他程序占用也会被计算进去。

#### 2. 预算公式

```text
target_used = total_memory × gpu_memory_utilization
current_used = total_memory - free_memory
cache_budget = target_used - current_used - runtime_reserve
num_blocks = floor(cache_budget / bytes_per_physical_block)
```

其中：

- `gpu_memory_utilization`：允许模型、KV Cache 等占用的 GPU 总显存比例
- `runtime_reserve`：目标比例内部，专门留给激活、SDPA workspace 和临时 Tensor
- 未进入 utilization 的剩余显存：最后一道 OOM 缓冲

这两层余量不能混为一谈。若只设置 85% 利用率却把其中所有剩余空间都分给 KV Cache，
Prefill 时的临时 Tensor 会把实际峰值继续推高。

Qwen3-1.7B、BF16、Block Size 16 的单 Block 大小仍是：

```text
28 layers × 2(K/V) × 16 tokens × 8 KV heads × 128 dim × 2 bytes
= 1.75 MiB
```

规划器还会计入常驻 GPU BlockTable 每个物理块增加的 `max_num_seqs × 4 bytes`。这个值
很小，但公式应保持完整。

#### 3. 自动与手动模式

默认自动：

```powershell
--gpu-memory-utilization 0.85
--kv-cache-runtime-reserve-mib 1024
```

手动覆盖：

```powershell
--num-kv-blocks 256
```

只要显式传入 `num_kv_blocks`，Engine 就不执行自动规划。这样历史 benchmark 可以固定
容量复现，而普通启动不再需要猜 Block 数。

#### 4. RTX 4060 Ti 实测

测试时系统已有桌面和其他 GPU 占用，规划输出为：

```text
利用率目标=85%
当前占用=5028.5 MiB
运行余量=1024.0 MiB
KV预算=906.9 MiB
Blocks=518
```

最终容量：

```text
518 blocks × 16 tokens = 8288 token slots
```

相比之前常用的 256 blocks，缓存容量增加约 `2.02 倍`。真实 Qwen3-1.7B Triton 推理
通过，结束后 518/518 Block 全部回收。BS8 长上下文压力测试中 Gather 路径峰值约
`5838 MiB`，没有 OOM。

规划结果会随桌面程序和其他 GPU 进程变化，这是正确行为，不是结果不稳定。例如另一个
程序占用 1 GB 后，Engine 应自动减少 KV Block，而不是继续按显卡标称容量强行分配。

#### 5. 参数如何调整

- 一般使用默认 `0.85 + 1024 MiB`
- 上下文容量不够且工作负载 Prefill 较短：逐步提高 utilization，例如 0.88
- 长 Prompt、大 Prefill Batch 或 Gather 路径：增加 runtime reserve
- 出现初始化 OOM：降低 utilization
- 初始化成功但首次 Prefill OOM：增加 runtime reserve

当前算法是简单的后加载静态规划，还没有像完整 vLLM 那样运行一次最大形状 profile 来
测量真实峰值激活。因此余量仍是显式参数，不能承诺任意输入形状绝不 OOM。

### 2026-06-12：第 10H 步，Block 不足时的 RECOMPUTE 抢占

Chunked Prefill 第一版在 admission 时为完整 Prompt 预留 Block。这样不会死锁，但会
出现一个明显问题：

```text
请求 A 实际只计算了 16 token，却预留完整 768-token 容量
请求 B 因为空闲 Block 不足停在 WAITING
```

本阶段改为按实际增长分配物理块，并在资源竞争时把低优先级请求移出 RUNNING。

#### 1. 两种“放不下”必须区分

资源竞争：

```text
A 单独需要 3 Blocks
B 单独需要 3 Blocks
物理池共有 4 Blocks
```

两条请求单独都能完成，只是不能同时保存完整 KV。这种情况可以抢占。

物理上不可能：

```text
单个请求最大上下文需要 5 Blocks
整个物理池只有 4 Blocks
```

移出 RUNNING 不会创造更多容量。Scheduler 在 admission 时直接抛出
`OutOfBlocksError`，错误信息明确说明“即使独占 GPU 也无法完成”。

最大 Block 需求按下面计算：

```text
prompt_tokens + max_new_tokens - 1
```

减一是因为最新生成的 output token 尚未写入 KV；它会作为下一轮 Decode 输入。

#### 2. Chunked 模式改为增量分配

新请求进入 RUNNING 时只注册空 BlockTable：

```text
physical_block_ids = ()
num_tokens = 0
```

每个 Prefill chunk 调用 `reserve()`，只为本轮新增 token 分配必要 Block。
`max_reservable_tokens()` 可以回答“把当前全部空闲块给这条请求，它最多还能前进多少
token”，Scheduler 会据此缩小 chunk，而不是直接失败。

#### 3. Decode 为什么也可能需要抢占

Decode 每轮只增加一个 token，但跨过 Block 边界时仍需要新物理块。若多条请求同时处于
边界，`reserve_many()` 可能需要的块数大于当前空闲数。

Scheduler 先运行 `schedule_decode()`：

1. 汇总本轮所有 Decode 请求新增 1 token 所需的新 Block。
2. 若容量足够，整个 Decode Batch 正常运行。
3. 若容量不足，选择受害者并释放其 BlockTable。
4. 重算剩余 Decode 请求的需求，直到可以原子 reserve。

这样 Engine 不会在模型前向完成后才发现 K/V 无处写入。

#### 4. 受害者选择规则

当前策略是：

```text
优先：尚未产生首 token 的 Prefill 请求
其次：最新进入 RUNNING 的 Decode 请求
同类：LIFO，优先保留更早到达的请求
```

优先抢占 Prefill 是因为它尚未向用户流式输出，不会直接打断正在观察的生成。LIFO 让
老请求先完成，避免所有请求都缓慢前进但没有任何一个释放资源。

Scheduler 不会抢占本轮已经固定 slots 的 Prefill chunk，否则执行计划会引用已经释放的
物理块。

#### 5. RECOMPUTE 如何保持生成正确

抢占时执行：

```text
释放该请求全部物理 KV Blocks
RUNNING -> WAITING
num_computed_tokens -> 0
重算目标 -> Prompt + 已生成 output tokens
```

已经返回给用户的 output token 不能删除。恢复后，模型重新处理：

```text
Prompt + output[0] + ... + output[n-1]
```

最后一个位置的 logits 正好继续预测 `output[n]`。因此重算会浪费计算，但不会改变逻辑
上下文。

采样随机数生成器也不会重建。重算旧上下文期间不采样，只有恢复到原生成位置后才消费
下一个随机数。单元测试验证了相同 seed 下：

```text
充足 Block，无抢占输出 == 紧张 Block，多次重算输出
```

TTFT 同样使用第一次产生首 token 的时间；Decode 请求重算完成时不会覆盖原 TTFT。

#### 6. 为什么需要重入门槛

第一版抢占后立即把请求放回 WAITING 队首，但下一轮只要有运行槽就重新接纳。4 Block
压力测试出现：

```text
请求 B 被抢占 -> 只拿到少量 Block -> 再次被抢占
平均每次 workload 抢占 10 次
```

这叫 preemption thrashing，大量时间花在重复计算同一段 Prompt。

现在首次请求仍可增量进入，但被抢占请求只有在：

```text
free_blocks >= 完整重算目标所需 Blocks
```

时才能恢复 RUNNING。它会暂时停在 WAITING，让当前高优先级请求尽快完成并释放整个工作
集。

#### 7. RTX 4060 Ti 压力基准

工作负载：

```text
4 requests
Prompt lengths = [32, 48, 32, 48]
Output limits = [4, 8, 4, 8]
BS = 2
Block Size = 16
```

| Triton 配置 | 总吞吐 tok/s | TTFT P95 ms | TPOT P95 ms | E2E P95 ms | 抢占/轮 |
|---|---:|---:|---:|---:|---:|
| 16 Blocks，容量充足 | 222.6 | 599.5 | 51.4 | 828.4 | 0 |
| 4 Blocks，无重入保护 | 119.6 | 1320.0 | 131.8 | 1538.0 | 10 |
| 4 Blocks，有重入保护 | 179.6 | 816.7 | 34.0 | 1028.6 | 3 |

4 Blocks 无法同时容纳两个请求的最大上下文，但每条请求单独可以完成。抢占让原本无法
并发执行的 workload 正确结束；重入保护又把 thrashing 大幅压低。

容量紧张仍有约 19% 的吞吐损失，这是重算成本，不应宣称抢占可以免费扩展显存。它的
价值是从“直接 OutOfBlocks / 死锁”降级为“降低吞吐但保持系统可前进”。

#### 8. 当前还缺少什么

- 没有把 KV Block 换出到 CPU，只有丢弃后重算
- 没有请求优先级、截止时间和最大抢占次数
- 没有按已计算 token 数估算受害者重算成本
- 在线请求持续到达时还需要更严格的防饥饿策略

下一步可以给 Scheduler 增加显式 priority 与 aging，或实现 CPU swap。对 8 GB 显卡的
教学项目，RECOMPUTE 比 swap 更简单，也更容易观察计算与显存之间的交换关系。

### 当前验证结果

```text
66 个测试全部通过
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
Paged Engine Prefill、Decode、Admission 与完整 Block 回收通过
Paged Attention 在线 softmax 与连续 Attention 对齐
Paged Decode 已验证不会调用 read_batch Gather
Triton 与 PyTorch Paged Attention 在不同长度 Batch 上对齐
真实 Qwen3-1.7B BF16 Triton 推理通过
向量化 Decode K/V 写回与逐项参考实现对齐
常驻 BlockTable Workspace 多轮更新地址稳定
Triton autotune 四组候选、缓存和基线保护通过
多种上下文长度的 benchmark 已写入 benchmarks/results.jsonl
长上下文 Serving 基准已记录 TTFT、TPOT、E2E、吞吐和显存
Engine 已迁移到独立子包，公共导入和真实 CUDA 推理保持兼容
Chunked Prefill、PD 混合轮、token budget 和不同历史 Batch 已验证
KV Block 可按真实 GPU 显存、利用率目标和运行余量自动规划
Prefill/Decode Block 竞争可触发 RECOMPUTE 抢占并避免重算抖动
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
- [x] 第 9 步：Paged KV Cache
  - [x] 第 9A 步：Block Manager 与物理 KV Block 池
  - [x] 第 9B 步：Engine 接入分页缓存
  - [x] 第 9C 步：Paged Attention 直接按 Block Table 读取
- [ ] 第 10 步：性能测量与优化
  - [x] 第 10A 步：Triton 单 token Paged Attention
  - [x] 第 10B 步：GQA 实验与向量化 Decode K/V 写回
  - [x] 第 10C 步：BlockTable 常驻 GPU 与 Kernel autotune
  - [x] 第 10D 步：长上下文 Serving 吞吐与时延总基线
  - [x] 第 10E 步：Engine 子包重构
  - [x] 第 10F 步：Chunked Prefill 与 PD 混合调度
  - [x] 第 10G 步：GPU 显存利用率与 KV Block 自动规划
  - [x] 第 10H 步：Block 不足时的 RECOMPUTE 抢占
  - [ ] 第 10I 步：固定 Batch Decode 的 CUDA Graph 实验
- [ ] 第 11 步：HTTP 服务（可选）
