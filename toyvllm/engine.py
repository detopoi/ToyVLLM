from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from toyvllm.layers.attention import KVCache
from toyvllm.models.qwen3 import Qwen3ForCausalLM
from toyvllm.sampling import SamplingParams, sample_next_token
from toyvllm.scheduler import Scheduler
from toyvllm.sequence import Sequence


@dataclass(frozen=True)
class EngineIteration:
    step: int
    decode_request_ids: tuple[int, ...]
    prefill_request_ids: tuple[int, ...]
    decode_seconds: float
    prefill_seconds: float


@dataclass(frozen=True)
class ContinuousBatchResult:
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
    """动态重组 batch 的教学版连续批处理引擎。"""

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
        self._caches: dict[int, list[KVCache]] = {}
        self._generators: dict[int, torch.Generator | None] = {}
        self._all_sequences: list[Sequence] = []
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
        """执行一个调度轮次，调用方可在轮次之间继续添加请求。"""

        if self.scheduler.is_done:
            raise RuntimeError("当前没有可调度请求")
        self._ensure_execution_started()
        step = self._step_index

        # 先推进已经在运行的请求。它们完成后立即释放槽位。
        decode_sequences = self.scheduler.running
        decode_seconds = self._execute_decode(decode_sequences, step=step)

        # 同一轮内马上用 FIFO 等待队列补满空槽，并执行新请求 prefill。
        prefill_sequences = self.scheduler.admit_waiting(step=step)
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

    def _ensure_execution_started(self) -> None:
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

        caches = self._unpack_cache(packed_cache, attention_mask)
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

    def _execute_decode(
        self,
        sequences: tuple[Sequence, ...],
        *,
        step: int,
    ) -> float:
        if not sequences:
            return 0.0

        packed_cache, attention_mask = self._pack_cache(sequences)
        input_ids = torch.tensor(
            [[sequence.last_token_id] for sequence in sequences],
            dtype=torch.long,
            device=self.device,
        )
        # 当前 decode token 是有效 key，所以在历史 mask 后追加一列 1。
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
        self._caches.pop(request_id, None)
        self._generators.pop(request_id, None)

    def _start_timing(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _stop_timing(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
