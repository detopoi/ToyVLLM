from __future__ import annotations

"""连续批处理执行器。

Scheduler 决定本轮运行谁；本模块负责把这些 Sequence 转成模型需要的规则 Tensor，
执行 Prefill/Decode，并管理每条请求对应的 GPU KV Cache。
"""

import time
from dataclasses import dataclass

import torch

from toyvllm.block_manager import BlockManager, OutOfBlocksError
from toyvllm.kv_cache import PagedKVCache
from toyvllm.layers.attention import KVCache
from toyvllm.models.qwen3 import Qwen3ForCausalLM
from toyvllm.sampling import SamplingParams, sample_next_token
from toyvllm.scheduler import Scheduler
from toyvllm.sequence import Sequence


@dataclass(frozen=True)
class EngineIteration:
    """一轮调度轨迹，区分旧请求 Decode 与新请求 Prefill。"""

    step: int
    decode_request_ids: tuple[int, ...]
    prefill_request_ids: tuple[int, ...]
    decode_seconds: float
    prefill_seconds: float


@dataclass(frozen=True)
class ContinuousBatchResult:
    """一次请求流执行完成后的结果和性能信息。"""

    sequences: tuple[Sequence, ...]
    completion_order: tuple[int, ...]
    iterations: tuple[EngineIteration, ...]
    total_seconds: float
    peak_memory_mib: float
    request_finish_seconds: tuple[float, ...]

    @property
    def total_output_tokens(self) -> int:
        return sum(len(sequence.output_token_ids) for sequence in self.sequences)

    @property
    def output_tokens_per_second(self) -> float:
        return self.total_output_tokens / self.total_seconds

    @property
    def requests_per_second(self) -> float:
        return len(self.sequences) / self.total_seconds


class ContinuousBatchEngine:
    """动态重组 batch 的教学版连续批处理引擎。

    核心循环不是“一批请求从头跑到尾”，而是：

    1. 让当前 RUNNING 请求各生成一个 token；
    2. 回收本轮完成请求的槽位和 KV Cache；
    3. 从 WAITING 队列接纳新请求；
    4. 对新请求执行 Prefill，让它们进入后续 Decode 流。

    当前版本为了教学清晰，每条请求持有紧凑连续 KV Cache。动态组成 batch 时需要
    pack/unpack，这也是现阶段连续模式性能较差、下一步引入 Paged KV Cache 的原因。
    """

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        *,
        max_num_seqs: int,
        pad_token_id: int,
    ) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.pad_token_id = pad_token_id
        self.scheduler = Scheduler(max_num_seqs)

        # 资源表和状态表都使用 request_id 关联，但所有权不同：
        # Scheduler 管请求状态；Engine 管 GPU Tensor 和随机数生成器。
        #
        # _caches[request_id][layer] = (key, value)
        # 单请求每层形状为 [1, kv_heads, sequence_length, head_dim]。
        self._caches: dict[int, list[KVCache]] = {}

        # 每条请求独立随机流，避免其他请求加入/退出 batch 后改变它的采样结果。
        self._generators: dict[int, torch.Generator | None] = {}

        # Scheduler.finished 侧重状态查询；_all_sequences 保留稳定的提交顺序，
        # 最终返回结果时按 request_id 对齐用户提交的 prompts。
        self._all_sequences: list[Sequence] = []

        # 以下字段记录整个 Engine 执行生命周期。step() 和 run() 共用它们，
        # 所以既支持一次跑完，也支持服务端在轮次之间插入在线请求。
        self._run_started = 0.0
        self._finish_seconds: dict[int, float] = {}
        self._iterations: list[EngineIteration] = []
        self._step_index = 0
        self._execution_started = False

    def add_request(
        self,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int,
        eos_token_ids: set[int],
        sampling_params: SamplingParams | None = None,
    ) -> int:
        """把请求交给 Scheduler，并准备请求级采样随机流。

        此时请求仍是 WAITING，不会创建 KV Cache。只有真正被接纳并完成 Prefill 后，
        _caches 中才会出现对应条目，避免排队请求提前占用显存。
        """

        sequence = self.scheduler.add_request(
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            sampling_params=sampling_params,
        )
        self._all_sequences.append(sequence)
        self._generators[sequence.request_id] = self._create_request_generator(sequence)
        return sequence.request_id

    @torch.inference_mode()
    def run(self) -> ContinuousBatchResult:
        """持续调用 step，直到 waiting 和 running 都为空。"""

        if not self._all_sequences:
            raise ValueError("引擎中没有请求")

        while not self.scheduler.is_done:
            self.step()

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        total_seconds = time.perf_counter() - self._run_started
        peak_memory_mib = 0.0
        if self.device.type == "cuda":
            peak_memory_mib = (
                torch.cuda.max_memory_allocated(self.device) / 1024**2
            )

        sequences = tuple(
            sorted(self._all_sequences, key=lambda sequence: sequence.request_id)
        )
        return ContinuousBatchResult(
            sequences=sequences,
            completion_order=self.scheduler.completion_order,
            iterations=tuple(self._iterations),
            total_seconds=total_seconds,
            peak_memory_mib=peak_memory_mib,
            request_finish_seconds=tuple(
                self._finish_seconds[sequence.request_id]
                for sequence in sequences
            ),
        )

    @torch.inference_mode()
    def step(self) -> EngineIteration:
        """执行一个调度轮次，调用方可在轮次之间继续添加请求。

        为什么先 Decode 再 Admit：

        - Decode 可能让短请求在本轮结束；
        - Scheduler 立即删除这些 RUNNING 请求；
        - 随后的 Admit 可以在同一轮复用刚释放的槽位。

        如果先 Admit，轮次开始时看不到这些即将释放的槽位，新请求会无谓多等一轮。
        """

        if self.scheduler.is_done:
            raise RuntimeError("当前没有可调度请求")
        self._ensure_execution_started()
        step = self._step_index

        # running 是进入本轮前的快照。执行期间 Scheduler 可能删除其中的完成请求，
        # 但 tuple 本身稳定，不会出现遍历字典时修改集合的问题。
        decode_sequences = self.scheduler.running
        decode_seconds = self._execute_decode(decode_sequences, step=step)

        # 这里只返回新接纳请求。旧 running 已经在上面 Decode 过，不能再 Prefill。
        prefill_sequences = self._admit_waiting(step=step)
        prefill_seconds = self._execute_prefill(prefill_sequences, step=step)

        iteration = EngineIteration(
            step=step,
            decode_request_ids=tuple(
                sequence.request_id for sequence in decode_sequences
            ),
            prefill_request_ids=tuple(
                sequence.request_id for sequence in prefill_sequences
            ),
            decode_seconds=decode_seconds,
            prefill_seconds=prefill_seconds,
        )
        self._iterations.append(iteration)
        self._step_index += 1
        return iteration

    def _admit_waiting(self, *, step: int) -> tuple[Sequence, ...]:
        """连续 Tensor 后端只受 max_num_seqs 限制。分页后端会覆盖此资源钩子。"""

        return self.scheduler.admit_waiting(step=step)

    def _ensure_execution_started(self) -> None:
        """第一次 step 时初始化计时，后续在线请求不会重置全局时间线。"""

        if self._execution_started:
            return
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        self._run_started = time.perf_counter()
        self._finish_seconds.clear()
        self._execution_started = True

    def _execute_prefill(
        self,
        sequences: tuple[Sequence, ...],
        *,
        step: int,
    ) -> float:
        """批量处理新请求的完整 Prompt，并建立各自第一份 KV Cache。

        不同长度 Prompt 只在本次模型调用中左 padding。模型返回后会立刻按 mask
        拆回每条请求自己的紧凑缓存，padding 不会永久占据请求缓存。
        """

        if not sequences:
            return 0.0

        # 左 padding 保证每行最后一个位置都是真实 Prompt 末 token，
        # 因而 last_token_only=True 可以统一取得首个生成 token 的 logits。
        max_length = max(len(sequence.prompt_token_ids) for sequence in sequences)
        input_rows = []
        mask_rows = []
        for sequence in sequences:
            padding = max_length - len(sequence.prompt_token_ids)
            input_rows.append(
                [self.pad_token_id] * padding + sequence.prompt_token_ids
            )
            mask_rows.append([0] * padding + [1] * len(sequence.prompt_token_ids))

        input_ids = torch.tensor(input_rows, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(
            mask_rows,
            dtype=torch.long,
            device=self.device,
        )

        started = self._start_timing()
        logits, packed_cache = self.model(
            input_ids,
            last_token_only=True,
            attention_mask=attention_mask,
            use_cache=True,
        )
        self._stop_timing()
        elapsed = time.perf_counter() - started

        # 模型返回的是规则 batch cache。Scheduler 不应知道 Tensor 细节，
        # Engine 在这里拆成 request_id -> per-layer cache。
        caches = self._unpack_cache(packed_cache, attention_mask)
        next_tokens = self._sample_rows(logits[:, -1], sequences)
        for index, sequence in enumerate(sequences):
            # 必须先保存本轮缓存，再提交 token。未完成请求下轮 Decode 要使用它；
            # 若 token 让请求完成，下面会马上 release，不会造成长期泄漏。
            self._caches[sequence.request_id] = caches[index]
            finish_reason = self.scheduler.append_token(
                sequence,
                int(next_tokens[index].item()),
                step=step,
            )
            if finish_reason is not None:
                self._finish_seconds[sequence.request_id] = (
                    time.perf_counter() - self._run_started
                )
                self._release(sequence.request_id)
        return elapsed

    def _execute_decode(
        self,
        sequences: tuple[Sequence, ...],
        *,
        step: int,
    ) -> float:
        """让所有旧 RUNNING 请求各向前生成一个 token。

        输入只有每条请求上轮生成的最后一个 token；历史上下文全部来自 KV Cache。
        """

        if not sequences:
            return 0.0

        # 每条请求的缓存长度可能不同，先临时补齐为规则 batch。
        packed_cache, attention_mask = self._pack_cache(sequences)
        input_ids = torch.tensor(
            [[sequence.last_token_id] for sequence in sequences],
            dtype=torch.long,
            device=self.device,
        )
        # packed cache 的 mask 只描述历史 token。模型还会把 current_input 产生的
        # 新 K/V 追加进去，因此 mask 也必须追加一列 1 与总 key 长度对齐。
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (len(sequences), 1),
                    dtype=attention_mask.dtype,
                    device=self.device,
                ),
            ),
            dim=1,
        )

        started = self._start_timing()
        logits, packed_present = self.model(
            input_ids,
            last_token_only=True,
            attention_mask=attention_mask,
            past_key_values=packed_cache,
            use_cache=True,
        )
        self._stop_timing()
        elapsed = time.perf_counter() - started

        # present 已包含“历史缓存 + 当前 decode token”，再拆回各请求紧凑形式。
        caches = self._unpack_cache(packed_present, attention_mask)
        next_tokens = self._sample_rows(logits[:, -1], sequences)
        for index, sequence in enumerate(sequences):
            self._caches[sequence.request_id] = caches[index]
            finish_reason = self.scheduler.append_token(
                sequence,
                int(next_tokens[index].item()),
                step=step,
            )
            if finish_reason is not None:
                self._finish_seconds[sequence.request_id] = (
                    time.perf_counter() - self._run_started
                )
                self._release(sequence.request_id)
        return elapsed

    def _pack_cache(
        self,
        sequences: tuple[Sequence, ...],
    ) -> tuple[list[KVCache], torch.Tensor]:
        """把不同长度的请求缓存临时拼成模型可接受的规则 batch。

        输入示意：

            request A: [valid valid]
            request B: [valid valid valid valid]

        左补齐后：

            request A: [  pad   pad valid valid]
            request B: [valid valid valid valid]

        返回的 attention_mask 标出哪些位置真实有效。这个过程在 28 层上都会发生，
        包含分配、补零和 cat，是当前连续引擎的主要性能瓶颈。
        """

        # 所有层的序列长度相同，读取第 0 层 Key 即可得到每条请求缓存长度。
        cache_lengths = [
            self._caches[sequence.request_id][0][0].shape[2]
            for sequence in sequences
        ]
        max_length = max(cache_lengths)
        mask_rows = [
            [0] * (max_length - length) + [1] * length
            for length in cache_lengths
        ]
        attention_mask = torch.tensor(
            mask_rows,
            dtype=torch.long,
            device=self.device,
        )

        packed_layers: list[KVCache] = []
        num_layers = len(self._caches[sequences[0].request_id])
        for layer_index in range(num_layers):
            keys = []
            values = []
            for sequence, length in zip(sequences, cache_lengths):
                key, value = self._caches[sequence.request_id][layer_index]
                padding = max_length - length
                if padding:
                    # KV Cache 在 sequence 维（dim=2）左侧补零，与 attention mask 对齐。
                    key_padding = key.new_zeros(
                        (1, key.shape[1], padding, key.shape[3])
                    )
                    value_padding = value.new_zeros(
                        (1, value.shape[1], padding, value.shape[3])
                    )
                    key = torch.cat((key_padding, key), dim=2)
                    value = torch.cat((value_padding, value), dim=2)
                keys.append(key)
                values.append(value)
            packed_layers.append(
                (torch.cat(keys, dim=0), torch.cat(values, dim=0))
            )
        return packed_layers, attention_mask

    @staticmethod
    def _unpack_cache(
        packed_cache: list[KVCache],
        attention_mask: torch.Tensor,
    ) -> list[list[KVCache]]:
        """按 attention_mask 去除临时 padding，恢复每条请求的紧凑缓存。

        返回布局是：

            unpacked[batch_index][layer_index] = (key, value)

        这样请求退出或重新组成不同 batch 时，都不依赖上一轮的 batch 行号。
        """

        batch_size = attention_mask.shape[0]
        unpacked: list[list[KVCache]] = [[] for _ in range(batch_size)]
        valid_masks = attention_mask.to(torch.bool)
        for key, value in packed_cache:
            for batch_index in range(batch_size):
                valid = valid_masks[batch_index]
                unpacked[batch_index].append(
                    (
                        key[batch_index : batch_index + 1, :, valid, :],
                        value[batch_index : batch_index + 1, :, valid, :],
                    )
                )
        return unpacked

    def _sample_rows(
        self,
        logits: torch.Tensor,
        sequences: tuple[Sequence, ...],
    ) -> torch.Tensor:
        """按 batch 行与 Sequence 一一对应，使用请求自己的采样参数和随机流。"""

        return torch.stack(
            [
                sample_next_token(
                    logits[index : index + 1],
                    sequence.sampling_params,
                    generator=self._generators[sequence.request_id],
                )[0]
                for index, sequence in enumerate(sequences)
            ]
        )

    def _create_request_generator(
        self,
        sequence: Sequence,
    ) -> torch.Generator | None:
        """为请求创建稳定随机流；request_id 使相同基础 seed 的请求彼此独立。"""

        params = sequence.sampling_params
        if params.is_greedy:
            return None
        generator = torch.Generator(device=self.device)
        if params.seed is None:
            generator.seed()
        else:
            generator.manual_seed(params.seed + sequence.request_id)
        return generator

    def _release(self, request_id: int) -> None:
        """释放请求级执行资源。

        Scheduler 已在 append_token 中释放逻辑运行槽；这里释放 GPU Cache 和随机流。
        两部分都完成后，这条请求才真正不再占用引擎运行资源。
        """

        self._caches.pop(request_id, None)
        self._generators.pop(request_id, None)

    def _start_timing(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _stop_timing(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


class PagedContinuousBatchEngine(ContinuousBatchEngine):
    """使用 BlockManager 和共享物理 KV Block 池的连续批处理引擎。

    这是 9C 教学实现：

    - 请求长期状态已经存放在分页物理池；
    - Prefill 和 Decode 只写本轮新增的 K/V；
    - 请求结束后按 Block Table 回收物理块；
    - Decode Attention 按 Block Table 直接扫描物理块；
    - 不再通过 read_batch gather 连续历史 Cache。

    当前使用纯 PyTorch 在线 softmax 来解释算法，尚未融合成 Triton/CUDA Kernel。
    """

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        *,
        max_num_seqs: int,
        pad_token_id: int,
        num_blocks: int,
        block_size: int = 16,
        attention_backend: str = "paged",
        vectorized_decode_write: bool = True,
        resident_block_tables: bool = True,
    ) -> None:
        super().__init__(
            model,
            max_num_seqs=max_num_seqs,
            pad_token_id=pad_token_id,
        )
        config = model.config
        parameter = next(model.parameters())
        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=block_size,
        )
        self.paged_cache = PagedKVCache(
            num_layers=config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=parameter.dtype,
            device=parameter.device,
            max_num_seqs=max_num_seqs,
        )
        if attention_backend not in {
            "auto",
            "gather",
            "paged",
            "triton",
            "triton-fixed",
            "triton-grouped",
        }:
            raise ValueError(
                "attention_backend 必须是 auto、gather、paged、triton、"
                "triton-fixed 或 triton-grouped"
            )
        if attention_backend == "auto":
            from toyvllm.kernels.paged_attention import is_triton_available

            attention_backend = (
                "triton"
                if parameter.is_cuda and is_triton_available()
                else "paged"
            )
        self.attention_backend = attention_backend
        self.vectorized_decode_write = vectorized_decode_write
        self.resident_block_tables = resident_block_tables

        # 父类的 _caches 属于连续 Tensor 后端。分页后端不使用它，
        # 保留空字典仅是为了维持父类初始化契约。
        self._caches.clear()

    def _admit_waiting(self, *, step: int) -> tuple[Sequence, ...]:
        """同时受运行槽位和空闲 KV Block 约束的 FIFO admission。"""

        admitted: list[Sequence] = []
        while (
            self.scheduler.waiting
            and len(self.scheduler.running) < self.scheduler.max_num_seqs
        ):
            candidate = self.scheduler.waiting[0]
            required = self.block_manager.blocks_required_for_tokens(
                len(candidate.prompt_token_ids)
            )
            if required > self.block_manager.stats.num_free_blocks:
                # 严格 FIFO：队首暂时放不下时，不绕过它接纳后面的短请求。
                # 如果当前没有 RUNNING 请求可释放块，则系统已无法取得进展。
                if not self.scheduler.running:
                    raise OutOfBlocksError(
                        f"队首请求 {candidate.request_id} 的 Prompt 需要 "
                        f"{required} 个 Block，但仅剩 "
                        f"{self.block_manager.stats.num_free_blocks} 个"
                    )
                break

            sequence = self.scheduler.admit_waiting(
                step=step,
                max_sequences=1,
            )[0]
            # allocate 在模型 Prefill 前建立 Block Table。只有真正被接纳的请求
            # 才占物理块，WAITING 请求不提前占用 GPU Cache。
            self.block_manager.allocate(
                sequence.request_id,
                num_tokens=len(sequence.prompt_token_ids),
            )
            admitted.append(sequence)
        return tuple(admitted)

    def _execute_prefill(
        self,
        sequences: tuple[Sequence, ...],
        *,
        step: int,
    ) -> float:
        if not sequences:
            return 0.0

        max_length = max(len(sequence.prompt_token_ids) for sequence in sequences)
        input_rows = []
        mask_rows = []
        for sequence in sequences:
            padding = max_length - len(sequence.prompt_token_ids)
            input_rows.append(
                [self.pad_token_id] * padding + sequence.prompt_token_ids
            )
            mask_rows.append([0] * padding + [1] * len(sequence.prompt_token_ids))

        input_ids = torch.tensor(input_rows, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(
            mask_rows,
            dtype=torch.long,
            device=self.device,
        )

        started = self._start_timing()
        logits, packed_cache = self.model(
            input_ids,
            last_token_only=True,
            attention_mask=attention_mask,
            use_cache=True,
        )
        self._stop_timing()
        elapsed = time.perf_counter() - started

        tables = tuple(
            self.block_manager.get_block_table(sequence.request_id)
            for sequence in sequences
        )
        self.paged_cache.write_prefill_batch(
            tables,
            packed_cache,
            attention_mask,
        )

        next_tokens = self._sample_rows(logits[:, -1], sequences)
        for index, sequence in enumerate(sequences):
            finish_reason = self.scheduler.append_token(
                sequence,
                int(next_tokens[index].item()),
                step=step,
            )
            if finish_reason is not None:
                self._finish_seconds[sequence.request_id] = (
                    time.perf_counter() - self._run_started
                )
                self._release(sequence.request_id)
        return elapsed

    def _execute_decode(
        self,
        sequences: tuple[Sequence, ...],
        *,
        step: int,
    ) -> float:
        if not sequences:
            return 0.0
        if self.attention_backend == "gather":
            return self._execute_decode_gather(sequences, step=step)

        history_tables = tuple(
            self.block_manager.get_block_table(sequence.request_id)
            for sequence in sequences
        )
        position_ids = torch.tensor(
            [[table.num_tokens] for table in history_tables],
            dtype=torch.long,
            device=self.device,
        )
        paged_attention = self.paged_cache.attention_metadata(
            history_tables,
            backend=self.attention_backend,
            use_workspace=self.resident_block_tables,
            # Decode 位置恰好等于历史长度。RoPE 和 Paged Attention 共享这份 GPU 元数据，
            # 避免为同一组整数分别上传 position_ids 与 context_lengths。
            context_lengths=position_ids[:, 0],
        )

        # 先原子预留写入槽位，避免模型算完后才发现容量不足。Attention 持有的是 reserve
        # 之前的不可变 BlockTable 快照，因此不会错误读取尚未写入的新槽位。
        slots = self.block_manager.reserve_many(
            tuple((sequence.request_id, 1) for sequence in sequences)
        )

        input_ids = torch.tensor(
            [[sequence.last_token_id] for sequence in sequences],
            dtype=torch.long,
            device=self.device,
        )
        started = self._start_timing()
        logits, packed_present = self.model(
            input_ids,
            last_token_only=True,
            position_ids=position_ids,
            paged_attention=paged_attention,
            use_cache=True,
        )
        self._stop_timing()
        elapsed = time.perf_counter() - started

        self.paged_cache.write_decode_batch(
            tuple(slots[sequence.request_id] for sequence in sequences),
            packed_present,
            vectorized=self.vectorized_decode_write,
        )
        next_tokens = self._sample_rows(logits[:, -1], sequences)
        for index, sequence in enumerate(sequences):
            finish_reason = self.scheduler.append_token(
                sequence,
                int(next_tokens[index].item()),
                step=step,
            )
            if finish_reason is not None:
                self._finish_seconds[sequence.request_id] = (
                    time.perf_counter() - self._run_started
                )
                self._release(sequence.request_id)
        return elapsed

    def _execute_decode_gather(
        self,
        sequences: tuple[Sequence, ...],
        *,
        step: int,
    ) -> float:
        """保留 9B 的 Gather 路径，仅用于和 9C 做可重复的 A/B Benchmark。"""

        old_tables = tuple(
            self.block_manager.get_block_table(sequence.request_id)
            for sequence in sequences
        )
        packed_cache, attention_mask = self.paged_cache.read_batch(old_tables)
        slots = self.block_manager.reserve_many(
            tuple((sequence.request_id, 1) for sequence in sequences)
        )
        input_ids = torch.tensor(
            [[sequence.last_token_id] for sequence in sequences],
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (len(sequences), 1),
                    dtype=attention_mask.dtype,
                    device=self.device,
                ),
            ),
            dim=1,
        )

        started = self._start_timing()
        logits, packed_present = self.model(
            input_ids,
            last_token_only=True,
            attention_mask=attention_mask,
            past_key_values=packed_cache,
            use_cache=True,
        )
        self._stop_timing()
        elapsed = time.perf_counter() - started

        self.paged_cache.write_decode_batch(
            tuple(slots[sequence.request_id] for sequence in sequences),
            packed_present,
            vectorized=self.vectorized_decode_write,
        )
        next_tokens = self._sample_rows(logits[:, -1], sequences)
        for index, sequence in enumerate(sequences):
            finish_reason = self.scheduler.append_token(
                sequence,
                int(next_tokens[index].item()),
                step=step,
            )
            if finish_reason is not None:
                self._finish_seconds[sequence.request_id] = (
                    time.perf_counter() - self._run_started
                )
                self._release(sequence.request_id)
        return elapsed

    def _release(self, request_id: int) -> None:
        """回收 Block Table 中的全部物理块，并释放请求随机流。"""

        self.block_manager.free(request_id)
        self._generators.pop(request_id, None)
