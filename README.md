# Toy vLLM

这是一个面向初学者、从零实现大模型推理引擎核心机制的教学项目。
当前固定使用本地 `Qwen3-1.7B` 和 RTX 4060 Ti。

完整的演进路线和每一步原理见 [TUTORIAL.md](TUTORIAL.md)。

## 当前运行环境

PowerShell 中先指定已经安装 CUDA PyTorch 的 Python 3.10：

```powershell
$PYTHON = "C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
```

检查环境：

```powershell
& $PYTHON -m toyvllm --model Qwen3-1.7B env
```

查看聊天模板和 token：

```powershell
& $PYTHON -m toyvllm --model Qwen3-1.7B tokenize "你好，请介绍一下自己。"
```

运行使用 KV Cache 的 greedy 推理：

```powershell
& $PYTHON -m toyvllm --model Qwen3-1.7B generate --max-new-tokens 16 "你好"
```

需要运行旧的重算基线时增加 `--backend naive`。

使用 Qwen3 推荐的采样参数：

```powershell
& $PYTHON -m toyvllm --model Qwen3-1.7B generate --sample --seed 123 `
    --max-new-tokens 32 "写一句关于夜空的短句。"
```

也可以通过 `--temperature`、`--top-k` 和 `--top-p` 覆盖模型默认值。

所有采样超参数都能独立指定；只要传入任意一个就会启用采样：

```powershell
& $PYTHON -m toyvllm generate --temperature 0.8 --top-k 40 `
    --top-p 0.9 --seed 123 "写一句关于夜空的短句。"
```

静态批处理不同长度的请求：

```powershell
& $PYTHON -m toyvllm batch --max-new-tokens 16 `
    "你好" "用一句话解释 KV Cache。" "请用三个词描述夏天。"
```

运行带 Scheduler 的连续批处理并查看每轮轨迹：

```powershell
& $PYTHON -m toyvllm continuous --max-num-seqs 2 --show-schedule `
    "你好" "解释 KV Cache" "描述夏天" "什么是 GPU"
```

运行测试：

```powershell
& $PYTHON -m unittest discover -s tests -v
```

运行当前 benchmark：

```powershell
& $PYTHON bench.py
```

需要保留一轮结果时：

```powershell
& $PYTHON bench.py --save benchmarks/results.jsonl --label stage-01-tokenizer
```

运行并保存朴素推理基线：

```powershell
& $PYTHON bench.py --backend naive --warmup 1 --iterations 3 `
    --max-new-tokens 16 --save benchmarks/results.jsonl --label stage-04-naive
```

运行相同配置的 KV Cache 基准：

```powershell
& $PYTHON bench.py --backend cached --warmup 1 --iterations 3 `
    --max-new-tokens 16 --save benchmarks/results.jsonl --label stage-05-kv-cache
```

观察长上下文下 KV Cache 的收益：

```powershell
& $PYTHON bench.py --backend cached --prompt-repeat 168 `
    --warmup 1 --iterations 3 --max-new-tokens 16
```

测量采样开销：

```powershell
& $PYTHON bench.py --backend cached --sample --seed 123 `
    --warmup 1 --iterations 3 --max-new-tokens 16
```

测量静态 batch：

```powershell
& $PYTHON bench.py --backend static --batch-size 4 `
    --warmup 1 --iterations 3 --max-new-tokens 16
```

对比静态与连续调度：

```powershell
& $PYTHON bench.py --backend continuous --batch-size 4 `
    --num-requests 8 --short-new-tokens 4 --max-new-tokens 16 `
    --warmup 1 --iterations 3
```

Paged KV Cache 已完成 Block Manager、Engine 接入、PyTorch 参考版和 Triton Kernel。
Windows 环境安装本项目验证过的 Triton：

```powershell
& $PYTHON -m pip install triton-windows==3.1.0.post17
```

运行分页后端：

```powershell
& $PYTHON -m toyvllm continuous --cache-backend paged `
    --paged-attention auto `
    --num-kv-blocks 64 --block-size 16 --max-num-seqs 4 `
    "你好" "解释 KV Cache" "描述夏天"
```

`auto` 在 CUDA 和 Triton 可用时选择 `triton`，否则回退到 `paged`。也可以显式使用：

- `--paged-attention gather`：9B Gather + SDPA 对照路径
- `--paged-attention paged`：9C 纯 PyTorch 在线 softmax
- `--paged-attention triton`：融合 Block 扫描和在线 softmax 的 Triton Kernel

Decode 写回也已向量化：所有请求、全部层的新 token K/V 被整理为
`[layers, batch, kv_heads, head_dim]`，Key/Value 各一次写入物理池，避免逐请求逐层
发射大量小写 Kernel。
