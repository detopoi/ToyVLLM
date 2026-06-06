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

运行当前最朴素的 greedy 推理：

```powershell
& $PYTHON -m toyvllm --model Qwen3-1.7B generate --max-new-tokens 16 "你好"
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
