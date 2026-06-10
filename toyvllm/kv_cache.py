from __future__ import annotations

"""Paged KV Cache 的数据面：预分配物理块池并按 Block Table 读写。"""

from dataclasses import dataclass

import torch

from toyvllm.engine.block_manager import BlockTable, PhysicalTokenSlot
from toyvllm.layers.attention import KVCache, PagedAttentionMetadata


@dataclass(frozen=True)
class PagedKVCacheShape:
    num_layers: int
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int


@dataclass
class PagedAttentionWorkspace:
    """常驻 CPU/GPU 的 BlockTable 元数据缓冲区。

    GPU Tensor 只在初始化时分配一次。每轮 Decode 更新前 `batch_size` 行和实际使用的
    Block 列，返回的是 Workspace 的切片视图，不再反复创建新的 CUDA Tensor。
    """

    cpu_block_tables: torch.Tensor
    cpu_context_lengths: torch.Tensor
    gpu_block_tables: torch.Tensor
    gpu_context_lengths: torch.Tensor
    row_block_ids: list[tuple[int, ...] | None]
    context_signature: tuple[int, ...] | None = None

    @classmethod
    def allocate(
        cls,
        *,
        max_num_seqs: int,
        max_num_blocks: int,
        device: torch.device,
    ) -> PagedAttentionWorkspace:
        if max_num_seqs <= 0:
            raise ValueError("max_num_seqs 必须大于 0")
        shape = (max_num_seqs, max_num_blocks)
        pin_memory = device.type == "cuda"
        return cls(
            cpu_block_tables=torch.empty(
                shape,
                dtype=torch.int32,
                pin_memory=pin_memory,
            ),
            cpu_context_lengths=torch.empty(
                max_num_seqs,
                dtype=torch.int32,
                pin_memory=pin_memory,
            ),
            gpu_block_tables=torch.empty(shape, dtype=torch.int32, device=device),
            gpu_context_lengths=torch.empty(
                max_num_seqs,
                dtype=torch.int32,
                device=device,
            ),
            row_block_ids=[None] * max_num_seqs,
        )

    def update(
        self,
        tables: tuple[BlockTable, ...],
        *,
        context_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(tables)
        max_blocks = max(table.num_blocks for table in tables)
        if batch_size > self.gpu_block_tables.shape[0]:
            raise ValueError("Decode batch 超过 BlockTable Workspace 容量")
        if max_blocks > self.gpu_block_tables.shape[1]:
            raise ValueError("请求 BlockTable 超过 Workspace 容量")

        block_table_changed = False
        for row, table in enumerate(tables):
            block_ids = table.physical_block_ids
            if self.row_block_ids[row] != block_ids:
                cpu_row = self.cpu_block_tables[row]
                cpu_row.fill_(-1)
                cpu_row[: len(block_ids)].copy_(
                    torch.tensor(block_ids, dtype=torch.int32)
                )
                self.row_block_ids[row] = block_ids
                block_table_changed = True

        gpu_tables = self.gpu_block_tables[:batch_size, :max_blocks]
        gpu_lengths = self.gpu_context_lengths[:batch_size]
        if block_table_changed:
            gpu_tables.copy_(
                self.cpu_block_tables[:batch_size, :max_blocks]
            )

        if context_lengths is not None:
            if context_lengths.shape != (batch_size,):
                raise ValueError("外部 context_lengths 形状必须是 [batch]")
            if context_lengths.device != gpu_lengths.device:
                raise ValueError("外部 context_lengths 必须和 Workspace 位于同一设备")
            gpu_lengths = context_lengths
        else:
            context_signature = tuple(table.num_tokens for table in tables)
            if self.context_signature != context_signature:
                self.cpu_context_lengths[:batch_size].copy_(
                    torch.tensor(context_signature, dtype=torch.int32)
                )
                gpu_lengths.copy_(self.cpu_context_lengths[:batch_size])
                self.context_signature = context_signature
        return gpu_tables, gpu_lengths


class PagedKVCache:
    """固定物理块池。

    每个 K/V Tensor 的布局为：

        [num_layers, num_blocks, block_size, num_kv_heads, head_dim]

    `num_blocks` 这一维是 BlockManager 分配的物理块号。请求的逻辑 token 不要求在
    物理显存中连续，只需通过 BlockTable 找到对应 block id 和 block offset。

    这里的物理块位于 CPU 或 GPU 全局内存，生命周期跨越多个 decode kernel。
    它不是 CUDA shared memory。Shared memory 是一次 kernel 内由同一 thread block
    临时共享的片上空间，kernel 结束后内容就不存在。
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
        max_num_seqs: int | None = None,
    ) -> None:
        dimensions = {
            "num_layers": num_layers,
            "num_blocks": num_blocks,
            "block_size": block_size,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")

        self.shape = PagedKVCacheShape(**dimensions)
        cache_shape = (
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        )

        # 一次性预分配整个物理池。后续请求增长只分配整数 block id，
        # 不再为每条请求反复申请越来越长的 Tensor。
        self.key_cache = torch.empty(cache_shape, dtype=dtype, device=device)
        self.value_cache = torch.empty(cache_shape, dtype=dtype, device=device)
        self.attention_workspace = (
            None
            if max_num_seqs is None
            else PagedAttentionWorkspace.allocate(
                max_num_seqs=max_num_seqs,
                max_num_blocks=num_blocks,
                device=self.key_cache.device,
            )
        )

    @property
    def device(self) -> torch.device:
        return self.key_cache.device

    @property
    def dtype(self) -> torch.dtype:
        return self.key_cache.dtype

    @property
    def bytes_per_block(self) -> int:
        elements = (
            self.shape.num_layers
            * self.shape.block_size
            * self.shape.num_kv_heads
            * self.shape.head_dim
        )
        # 一个物理块同时包含 Key 和 Value，所以乘 2。
        return elements * self.key_cache.element_size() * 2

    @property
    def allocated_bytes(self) -> int:
        return self.bytes_per_block * self.shape.num_blocks

    def write(
        self,
        slots: tuple[PhysicalTokenSlot, ...],
        layer_key_values: list[KVCache],
    ) -> None:
        """把一段新 token 的各层 K/V 写入已分配物理槽位。

        `layer_key_values[layer]` 的形状沿用模型输出：

            Key/Value [1, num_kv_heads, num_new_tokens, head_dim]
        """

        num_tokens = len(slots)
        if num_tokens == 0:
            raise ValueError("slots 不能为空")
        self._validate_layer_key_values(layer_key_values, num_tokens)

        block_ids = torch.tensor(
            [slot.physical_block_id for slot in slots],
            dtype=torch.long,
            device=self.device,
        )
        offsets = torch.tensor(
            [slot.block_offset for slot in slots],
            dtype=torch.long,
            device=self.device,
        )

        for layer_index, (key, value) in enumerate(layer_key_values):
            # 模型布局 [1, heads, tokens, dim] 转成物理池写入布局 [tokens, heads, dim]。
            key_rows = key[0].transpose(0, 1).to(
                device=self.device,
                dtype=self.dtype,
            )
            value_rows = value[0].transpose(0, 1).to(
                device=self.device,
                dtype=self.dtype,
            )
            self.key_cache[layer_index, block_ids, offsets] = key_rows
            self.value_cache[layer_index, block_ids, offsets] = value_rows

    def read(self, table: BlockTable) -> list[KVCache]:
        """按逻辑 token 顺序从可能不连续的物理块中还原紧凑 KV Cache。"""

        if table.block_size != self.shape.block_size:
            raise ValueError("BlockTable 与物理池的 block_size 不一致")
        if table.num_tokens == 0:
            # Chunked Prefill 的首个 chunk 尚无历史。返回形状正确的零长度 Cache，
            # 使 read_batch 可以把“无历史”和“有历史”请求放进同一个 Batch。
            return [
                (
                    self.key_cache.new_empty(
                        (1, self.shape.num_kv_heads, 0, self.shape.head_dim)
                    ),
                    self.value_cache.new_empty(
                        (1, self.shape.num_kv_heads, 0, self.shape.head_dim)
                    ),
                )
                for _ in range(self.shape.num_layers)
            ]

        slots = table.slots()
        block_ids = torch.tensor(
            [slot.physical_block_id for slot in slots],
            dtype=torch.long,
            device=self.device,
        )
        offsets = torch.tensor(
            [slot.block_offset for slot in slots],
            dtype=torch.long,
            device=self.device,
        )

        layers: list[KVCache] = []
        for layer_index in range(self.shape.num_layers):
            # Gather 结果为 [tokens, heads, dim]，再恢复模型使用的
            # [1, heads, tokens, dim]。
            key_rows = self.key_cache[layer_index, block_ids, offsets]
            value_rows = self.value_cache[layer_index, block_ids, offsets]
            key = key_rows.transpose(0, 1).unsqueeze(0).contiguous()
            value = value_rows.transpose(0, 1).unsqueeze(0).contiguous()
            layers.append((key, value))
        return layers

    def read_batch(
        self,
        tables: tuple[BlockTable, ...],
    ) -> tuple[list[KVCache], torch.Tensor]:
        """把多条分页请求临时 gather 成模型当前接口需要的连续 Batch。

        这是 9B 的过渡接口。它仍会产生 padding 和复制；9C Paged Attention 将直接接收
        物理块池与 Block Table，届时不再构造这份连续历史 Cache。
        """

        if not tables:
            raise ValueError("tables 不能为空")
        compact = [self.read(table) for table in tables]
        lengths = [table.num_tokens for table in tables]
        max_length = max(lengths)
        attention_mask = torch.tensor(
            [
                [0] * (max_length - length) + [1] * length
                for length in lengths
            ],
            dtype=torch.long,
            device=self.device,
        )

        packed: list[KVCache] = []
        for layer_index in range(self.shape.num_layers):
            keys = []
            values = []
            for request_layers, length in zip(compact, lengths):
                key, value = request_layers[layer_index]
                padding = max_length - length
                if padding:
                    key_padding = key.new_zeros(
                        (1, self.shape.num_kv_heads, padding, self.shape.head_dim)
                    )
                    value_padding = value.new_zeros(
                        (1, self.shape.num_kv_heads, padding, self.shape.head_dim)
                    )
                    key = torch.cat((key_padding, key), dim=2)
                    value = torch.cat((value_padding, value), dim=2)
                keys.append(key)
                values.append(value)
            packed.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        return packed, attention_mask

    def attention_metadata(
        self,
        tables: tuple[BlockTable, ...],
        *,
        backend: str = "paged",
        use_workspace: bool = True,
        context_lengths: torch.Tensor | None = None,
    ) -> PagedAttentionMetadata:
        """创建不复制 K/V 数据的 Paged Attention 只读视图。

        Tensor 只是引用同一份物理池；真正随请求变化的只有很小的 BlockTable 元组。
        """

        if not tables:
            raise ValueError("tables 不能为空")
        for table in tables:
            if table.block_size != self.shape.block_size:
                raise ValueError("BlockTable 与物理池的 block_size 不一致")
        if backend not in {
            "paged",
            "triton",
            "triton-fixed",
            "triton-grouped",
        }:
            raise ValueError(
                "Paged Attention backend 必须是 paged、triton、"
                "triton-fixed 或 triton-grouped"
            )

        block_table_tensor = None
        context_lengths = None
        if backend in {"triton", "triton-fixed", "triton-grouped"}:
            max_blocks = max(table.num_blocks for table in tables)
            if use_workspace and self.attention_workspace is not None:
                block_table_tensor, context_lengths = (
                    self.attention_workspace.update(
                        tables,
                        context_lengths=context_lengths,
                    )
                )
            else:
                # 10A 的瞬时路径保留用于 A/B：每轮新建 CPU 数据并上传为 CUDA Tensor。
                rows = [
                    list(table.physical_block_ids)
                    + [-1] * (max_blocks - table.num_blocks)
                    for table in tables
                ]
                block_table_tensor = torch.tensor(
                    rows,
                    dtype=torch.int32,
                    device=self.device,
                )
                if context_lengths is None:
                    context_lengths = torch.tensor(
                        [table.num_tokens for table in tables],
                        dtype=torch.int32,
                        device=self.device,
                    )
        return PagedAttentionMetadata(
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            block_tables=tables,
            backend=backend,
            block_table_tensor=block_table_tensor,
            context_lengths=context_lengths,
        )

    def write_prefill_batch(
        self,
        tables: tuple[BlockTable, ...],
        packed_cache: list[KVCache],
        attention_mask: torch.Tensor,
    ) -> None:
        """把左 Padding 的 Prefill Batch 拆成请求，并写入各自物理块。"""

        if len(tables) != attention_mask.shape[0]:
            raise ValueError("BlockTable 数量与 batch size 不一致")
        valid_masks = attention_mask.to(torch.bool)
        for batch_index, table in enumerate(tables):
            valid = valid_masks[batch_index]
            layer_values: list[KVCache] = []
            for key, value in packed_cache:
                layer_values.append(
                    (
                        key[batch_index : batch_index + 1, :, valid, :],
                        value[batch_index : batch_index + 1, :, valid, :],
                    )
                )
            self.write(table.slots(), layer_values)

    def write_decode_batch(
        self,
        slots_by_request: tuple[tuple[PhysicalTokenSlot, ...], ...],
        packed_present: list[KVCache],
        *,
        vectorized: bool = True,
    ) -> None:
        """只把本轮 Decode 新增的最后一个 K/V 写入物理池。

        packed_present 包含“临时连续历史 + 当前 token”。当前 token 始终位于最后一列，
        因此无需把整段历史拆回并重写。
        """

        batch_size = len(slots_by_request)
        if any(len(slots) != 1 for slots in slots_by_request):
            raise ValueError("当前 Decode 每条请求必须恰好新增一个 token")
        if not packed_present or packed_present[0][0].shape[0] != batch_size:
            raise ValueError("packed_present 的 batch size 不一致")

        if vectorized:
            self._write_decode_batch_vectorized(
                slots_by_request,
                packed_present,
            )
            return

        # 10A 及之前的参考写法：按请求拆分，再由 write() 按层写入。代码直观，但会产生
        # batch_size * num_layers * 2 次小 Tensor 写操作，仅保留用于 benchmark 对照。
        for batch_index, slots in enumerate(slots_by_request):
            layer_values: list[KVCache] = []
            for key, value in packed_present:
                layer_values.append(
                    (
                        key[batch_index : batch_index + 1, :, -1:, :],
                        value[batch_index : batch_index + 1, :, -1:, :],
                    )
                )
            self.write(slots, layer_values)

    def _write_decode_batch_vectorized(
        self,
        slots_by_request: tuple[tuple[PhysicalTokenSlot, ...], ...],
        packed_present: list[KVCache],
    ) -> None:
        """把整批、全部层的新 K/V 合并为两次物理池写入。

        每层模型返回 `[batch, kv_heads, sequence, head_dim]`。只取最后一个 token 后，
        沿 layer 维 stack 成：

            [num_layers, batch, kv_heads, head_dim]

        物理池使用同样的 layer/batch/head/dim 顺序做高级索引，因此 Key 和 Value 各需
        一次赋值，不再为每个请求、每一层单独发射小写 Kernel。
        """

        block_ids = torch.tensor(
            [slots[0].physical_block_id for slots in slots_by_request],
            dtype=torch.long,
            device=self.device,
        )
        offsets = torch.tensor(
            [slots[0].block_offset for slots in slots_by_request],
            dtype=torch.long,
            device=self.device,
        )
        key_rows = torch.stack(
            [key[:, :, -1, :] for key, _ in packed_present],
            dim=0,
        ).to(device=self.device, dtype=self.dtype)
        value_rows = torch.stack(
            [value[:, :, -1, :] for _, value in packed_present],
            dim=0,
        ).to(device=self.device, dtype=self.dtype)

        self.key_cache[:, block_ids, offsets] = key_rows
        self.value_cache[:, block_ids, offsets] = value_rows

    def clear_blocks(self, block_ids: tuple[int, ...] | list[int]) -> None:
        """可选地清零释放块，便于测试或有数据隔离要求的场景。

        正确性并不依赖清零：BlockTable.num_tokens 保证未写入槽位不会被读取。
        生产引擎通常更关注性能，会在块重新分配后直接覆盖有效位置。
        """

        if not block_ids:
            return
        ids = torch.tensor(block_ids, dtype=torch.long, device=self.device)
        # 不能写 self.key_cache[:, ids].zero_()：LongTensor 高级索引会返回副本，
        # zero_ 只清零副本，不会修改原物理池。index_fill_ 明确在 dim=1 原地写回。
        self.key_cache.index_fill_(1, ids, 0)
        self.value_cache.index_fill_(1, ids, 0)

    def _validate_layer_key_values(
        self,
        layer_key_values: list[KVCache],
        num_tokens: int,
    ) -> None:
        if len(layer_key_values) != self.shape.num_layers:
            raise ValueError("K/V 层数与物理池不一致")

        expected = (
            1,
            self.shape.num_kv_heads,
            num_tokens,
            self.shape.head_dim,
        )
        for layer_index, (key, value) in enumerate(layer_key_values):
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError(
                    f"第 {layer_index} 层 K/V 形状应为 {expected}，"
                    f"实际为 {tuple(key.shape)} / {tuple(value.shape)}"
                )
