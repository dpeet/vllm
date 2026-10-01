# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk-backed Qwen4Exp PLE n-gram table (``QWEN4EXP_PLE_MMAP=1``).

The n-gram table is 47.7 GiB in FP8 (NVIDIA checkpoint) or ~95 GiB in BF16
(ukisai Swift), yet one token only touches ``ngram_heads`` rows of it. On a
unified-memory box (DGX Spark / GB10) neither the device table nor the pinned
host table fits next to the other weights, so with this path the table stays
in the checkpoint's safetensors files and only the rows a step needs are read
(positional reads on a thread pool; the page cache keeps the hot rows).

Two consumers:

* ``Qwen4ExpNGramEmbedding`` builds :class:`Qwen4ExpPLEDiskEmbedding` instead
  of the device or pinned-host table, so the full-table allocation never
  happens, whatever ``--engram-config`` says.
* ``Qwen4ExpModelState.prepare_inputs`` (``QWEN4EXP_PLE_STAGED=1``) gathers the
  step's rows into a fixed GPU staging buffer before the forward or the CUDA
  graph replay; the forward then only takes a view of that buffer, so FULL
  decode graphs stay capturable.

Provenance: the preadv reader and the staged gather are Kai / 2Wild's
single-Spark vLLM recipe (github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark,
Apache-2.0; staging idea after Trosfy's vllm#54129). The placeholder-embedding
shape against v0.30's ``ngram_embedding.py`` interface, the per-shard row
validation and the FP8-scale / 16-bit table handling follow blazux's v0.30
disk-backed PLE (github.com/blazux/qwen3.8-Flash-DGX ``src/vllm_ple_mmap.py``,
Apache-2.0).
"""

import contextlib
import json
import math
import os
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

_DTYPE_BYTES = {"F8_E4M3": 1, "F8_E5M2": 1, "BF16": 2, "F16": 2, "F32": 4}
_TORCH_DTYPE = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def ple_disk_enabled() -> bool:
    """Serve the PLE table from the checkpoint files instead of memory."""
    return os.environ.get("QWEN4EXP_PLE_MMAP", "0") == "1"


def ple_staged_enabled() -> bool:
    """Gather PLE rows in the model state, before the forward/graph replay."""
    return ple_disk_enabled() and os.environ.get("QWEN4EXP_PLE_STAGED", "0") == "1"


def _read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


def layer_index_from_prefix(prefix: str) -> int:
    """Return the decoder layer index encoded in a module prefix."""
    match = re.search(r"layers\.(\d+)\.", prefix)
    if match is None:
        raise ValueError(f"PLE disk: no layer index in prefix {prefix!r}")
    return int(match.group(1))


class PLEDiskTable:
    """Row reader over one PLE layer's checkpoint shards.

    Shard ``i`` holds global rows ``[i * shard_rows, i * shard_rows + rows_i)``,
    the layout ``Qwen4ExpNGramEmbedding.load_weights`` assumes. Construction
    fails closed on a missing, duplicate or mis-sized shard, mixed dtypes, a
    width other than ``head_dim``, an FP8 table without its global
    ``weight_scale`` or a 16-bit table with one.
    """

    def __init__(
        self,
        model_dir: str,
        layer_idx: int,
        *,
        num_rows: int,
        head_dim: int,
        split_parts: int,
        threads: int | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.layer_idx = int(layer_idx)
        shard_re = re.compile(
            rf"(?:^|\.)layers\.{self.layer_idx}\.ple\.ple_embedding\."
            r"ngram_embedding\.shard_(\d+)\.weight$"
        )
        scale_re = re.compile(
            rf"(?:^|\.)layers\.{self.layer_idx}\.ple\.ple_embedding\."
            r"ngram_embedding\.weight_scale$"
        )
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map: dict[str, str] = json.load(f)["weight_map"]
            files = sorted(
                {
                    fn
                    for name, fn in weight_map.items()
                    if shard_re.search(name) or scale_re.search(name)
                }
            )
        else:
            files = sorted(
                fn for fn in os.listdir(model_dir) if fn.endswith(".safetensors")
            )

        # shard index -> (path, absolute byte offset of row 0, rows)
        shards: dict[int, tuple[str, int, int]] = {}
        dtype_names: set[str] = set()
        widths: set[int] = set()
        self.scale_entry: tuple[str, int, int, str] | None = None
        for fn in files:
            path = os.path.join(model_dir, fn)
            if not os.path.exists(path):
                raise FileNotFoundError(f"PLE disk: shard file missing: {path}")
            header, data_start = _read_header(path)
            file_size = os.path.getsize(path)
            for name, meta in header.items():
                begin, end = meta["data_offsets"]
                if (scale_re.search(name) or shard_re.search(name)) and (
                    data_start + end > file_size
                ):
                    # A partial download otherwise boots fine and the first request that
                    # touches a missing row kills the engine with a short read.
                    raise ValueError(
                        f"PLE disk: {path} is truncated: {name} ends at byte "
                        f"{data_start + end} but the file has {file_size}"
                    )
                if scale_re.search(name):
                    self.scale_entry = (
                        path,
                        data_start + begin,
                        end - begin,
                        meta["dtype"],
                    )
                    continue
                match = shard_re.search(name)
                if match is None:
                    continue
                if meta["dtype"] not in _TORCH_DTYPE:
                    raise ValueError(f"PLE disk: unsupported dtype {meta['dtype']}")
                rows, width = meta["shape"]
                if end - begin != rows * width * _DTYPE_BYTES[meta["dtype"]]:
                    raise ValueError(f"PLE disk: {name} byte size != shape")
                index = int(match.group(1))
                if index in shards:
                    raise ValueError(f"PLE disk: shard {index} appears twice")
                shards[index] = (path, data_start + begin, rows)
                dtype_names.add(meta["dtype"])
                widths.add(width)
        if not shards:
            raise ValueError(
                f"PLE disk: no n-gram shards for layer {self.layer_idx} in {model_dir}"
            )
        if len(dtype_names) != 1:
            raise ValueError(f"PLE disk: shards have mixed dtypes {dtype_names}")
        if widths != {head_dim}:
            raise ValueError(f"PLE disk: shard width {widths} != head_dim {head_dim}")

        # Same shard arithmetic as Qwen4ExpNGramEmbedding.load_weights.
        self.shard_rows = math.ceil(num_rows / split_parts)
        for index in range(split_parts):
            expected = max(0, min(self.shard_rows, num_rows - index * self.shard_rows))
            if expected == 0:
                if index in shards and shards[index][2] != 0:
                    raise ValueError(f"PLE disk: unexpected rows in shard {index}")
                continue
            if index not in shards:
                raise ValueError(f"PLE disk: shard {index} is missing")
            if shards[index][2] != expected:
                raise ValueError(
                    f"PLE disk: shard {index} has {shards[index][2]} rows, "
                    f"expected {expected}"
                )
        extra = sorted(i for i in shards if i >= split_parts)
        if extra:
            raise ValueError(f"PLE disk: shards beyond split_ngram_parts: {extra}")

        self.dtype_name = dtype_names.pop()
        self.dtype = _TORCH_DTYPE[self.dtype_name]
        self.row_bytes = head_dim * _DTYPE_BYTES[self.dtype_name]
        self.rows_total = num_rows
        if self.dtype in _FP8_DTYPES and self.scale_entry is None:
            raise ValueError("PLE disk: FP8 table without ngram_embedding.weight_scale")
        if self.dtype not in _FP8_DTYPES and self.scale_entry is not None:
            raise ValueError(f"PLE disk: {self.dtype_name} table has a weight_scale")

        # Positional reads: one fd per file, rows fetched with preadv straight
        # into the staging buffer. An mmap backend was measured first: page
        # faults serialize on the process mmap_lock (~10-20K rows/s cold at any
        # thread count), while preadv scales with the NVMe queue depth.
        self.threads = threads or max(8, min(48, 2 * (os.cpu_count() or 8)))
        self.chunk_rows = int(os.environ.get("QWEN4EXP_PLE_MMAP_CHUNK", "8"))
        if self.chunk_rows < 1:
            raise ValueError(
                f"QWEN4EXP_PLE_MMAP_CHUNK must be >= 1, got {self.chunk_rows}"
            )
        self._fds: dict[str, int] = {}
        self.shard_base: dict[int, tuple[int, int]] = {
            index: (self._open(path), offset)
            for index, (path, offset, _rows) in shards.items()
        }
        self._pool = ThreadPoolExecutor(
            max_workers=self.threads, thread_name_prefix="ple-disk"
        )
        self._staging: torch.Tensor | None = None

    def _open(self, path: str) -> int:
        if path not in self._fds:
            fd = os.open(path, os.O_RDONLY)
            with contextlib.suppress(AttributeError, OSError):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            self._fds[path] = fd
        return self._fds[path]

    def read_scale(self) -> torch.Tensor:
        """Read the FP8 table's global scale as a float32 scalar tensor."""
        if self.scale_entry is None:
            raise ValueError("PLE disk: table has no weight_scale")
        path, offset, nbytes, dtype_name = self.scale_entry
        if dtype_name not in _TORCH_DTYPE or nbytes != _DTYPE_BYTES[dtype_name]:
            raise ValueError(f"PLE disk: weight_scale is not one {dtype_name} value")
        with open(path, "rb") as f:
            f.seek(offset)
            raw = bytearray(f.read(nbytes))
        return torch.frombuffer(raw, dtype=_TORCH_DTYPE[dtype_name]).float()[0]

    def _staging_for(self, num_rows: int) -> torch.Tensor:
        if self._staging is None or self._staging.shape[0] < num_rows:
            capacity = max(num_rows, 4096)
            try:
                self._staging = torch.empty(
                    (capacity, self.row_bytes),
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                )
            except RuntimeError:
                # No CUDA context (CPU test): pageable staging still works,
                # just without async H2D overlap.
                self._staging = torch.empty(
                    (capacity, self.row_bytes), dtype=torch.uint8, device="cpu"
                )
        return self._staging[:num_rows]

    def _gather_chunk(self, ids: list[int], view: memoryview, start: int) -> None:
        row_bytes = self.row_bytes
        for k, row in enumerate(ids):
            shard, local = divmod(row, self.shard_rows)
            fd, offset = self.shard_base[shard]
            dst = (start + k) * row_bytes
            got = os.preadv(
                fd, [view[dst : dst + row_bytes]], offset + local * row_bytes
            )
            if got != row_bytes:
                raise OSError(f"PLE disk: short read {got}/{row_bytes} (row {row})")

    def gather_cpu(self, ids: torch.Tensor) -> torch.Tensor:
        """Gather flat int64 CPU row ids into a reused uint8 [n, row_bytes].

        The returned tensor is overwritten by the next call; callers that copy
        it asynchronously to the GPU rely on the next call's device-to-host id
        copy synchronizing the stream first.
        """
        if ids.device.type != "cpu":
            raise ValueError("PLE disk: gather_cpu takes CPU ids")
        num = ids.numel()
        out = self._staging_for(num)
        if num == 0:
            return out
        id_list = ids.reshape(-1).tolist()
        low, high = min(id_list), max(id_list)
        if low < 0 or high >= self.rows_total:
            raise IndexError(
                f"PLE disk: row id range [{low}, {high}] outside {self.rows_total} rows"
            )
        view = memoryview(out.numpy()).cast("B")
        chunk = self.chunk_rows
        if num <= chunk:
            self._gather_chunk(id_list, view, 0)
            return out
        futures = [
            self._pool.submit(self._gather_chunk, id_list[a : a + chunk], view, a)
            for a in range(0, num, chunk)
        ]
        for future in futures:
            future.result()
        return out

    def describe(self) -> str:
        return (
            f"PLEDiskTable(layer={self.layer_idx}, rows={self.rows_total:,}, "
            f"dtype={self.dtype_name}, row_bytes={self.row_bytes}, "
            f"files={len(self._fds)}, threads={self.threads})"
        )


class Qwen4ExpPLEDiskEmbedding(nn.Module):
    """Stands in for the device/pinned PLE table; rows come from disk.

    It exposes what ``Qwen4ExpNGramEmbedding`` and ``Qwen4ExpPLELayer`` read
    from their embedding (``supports_prefetch``, ``start_prefetch``,
    ``dequantize``, ``weight``, ``org_vocab_size``, ``embedding_dim``) and
    allocates no table storage: ``weight`` is a zero-row tensor that only
    carries the table's dtype for logging and introspection.
    """

    supports_prefetch: ClassVar[bool] = False
    weight_scale: torch.Tensor

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        prefix: str,
        model_dir: str,
        split_parts: int,
        expected_dtype: torch.dtype,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        if not reduce_results:
            # Every rank reads whole rows of the full table, so a caller-side
            # reduce-scatter would sum TP identical copies.
            raise ValueError(
                f"PLE disk: {prefix} serves full rows and cannot defer the "
                "vocabulary reduction (reduce_results=False)"
            )
        self.reduce_results = reduce_results
        self.num_embeddings = int(num_embeddings)
        self.org_vocab_size = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.prefix = prefix
        threads = os.environ.get("QWEN4EXP_PLE_MMAP_THREADS")
        self.table = PLEDiskTable(
            model_dir,
            layer_index_from_prefix(prefix),
            num_rows=self.org_vocab_size,
            head_dim=self.embedding_dim,
            split_parts=split_parts,
            threads=int(threads) if threads else None,
        )
        if self.table.dtype != expected_dtype:
            raise ValueError(
                f"PLE disk: checkpoint table is {self.table.dtype} but the "
                f"quantization config selects {expected_dtype} for {prefix}"
            )
        self.is_fp8 = self.table.dtype in _FP8_DTYPES
        self.weight = torch.empty((0, self.embedding_dim), dtype=self.table.dtype)
        # NaN until the checkpoint's global scale is loaded (FP8 tables only).
        self.register_buffer(
            "weight_scale", torch.full((), float("nan")), persistent=False
        )
        self._staging: torch.Tensor | None = None
        logger.info("PLE disk: %s -> %s", prefix, self.table.describe())

    def load_scale(self, loaded_weight: torch.Tensor) -> None:
        """Take the checkpoint's global FP8 scale (a single value)."""
        if not self.is_fp8:
            raise ValueError(f"PLE disk: {self.table.dtype} table got a weight_scale")
        if loaded_weight.numel() != 1:
            raise ValueError(
                f"PLE disk: weight_scale must hold one value, got "
                f"{tuple(loaded_weight.shape)}"
            )
        self.weight_scale.copy_(loaded_weight.reshape(()).float())

    def verify_loaded(self) -> None:
        """Fail closed when an FP8 table's global scale never arrived."""
        if not self.is_fp8:
            return
        if bool(torch.isnan(self.weight_scale)):
            raise ValueError("FP8 PLE checkpoint is missing its global scale")
        # A zero, negative or infinite scale would silently zero or blow up
        # every PLE embedding.
        scale = float(self.weight_scale)
        if not (math.isfinite(scale) and scale > 0.0):
            raise ValueError(
                "FP8 PLE checkpoint global scale must be finite and positive, "
                f"got {scale}"
            )

    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        """Rows are staged by the model state or gathered in ``forward``."""
        return None

    def dequantize(
        self,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.is_fp8:
            return embeddings.to(output_dtype)
        scale = self.weight_scale
        if scale.device != embeddings.device:
            raise RuntimeError("PLE disk: FP8 scale must be on the output device")
        return embeddings.to(output_dtype) * scale.to(output_dtype)

    # Staged mode: a fixed GPU buffer the captured graph reads.
    def init_staging(self, max_rows: int, device: torch.device) -> None:
        self._staging = torch.zeros(
            (max_rows, self.table.row_bytes), dtype=torch.uint8, device=device
        )
        logger.info(
            "PLE disk staged gather: %d rows x %d bytes on %s",
            max_rows,
            self.table.row_bytes,
            device,
        )

    @property
    def is_staged(self) -> bool:
        return self._staging is not None

    def stage_zero(self, num_rows: int) -> None:
        assert self._staging is not None
        self._staging[: max(num_rows, 0)].zero_()

    def stage_rows(self, ngram_ids: torch.Tensor, padded_rows: int) -> None:
        """Gather ``ngram_ids`` rows into staging and zero up to ``padded_rows``."""
        staging = self._staging
        assert staging is not None
        num_rows = ngram_ids.numel()
        if padded_rows > staging.shape[0] or num_rows > padded_rows:
            raise ValueError(
                f"PLE disk: {num_rows}/{padded_rows} rows exceed the staging "
                f"buffer ({staging.shape[0]} rows)"
            )
        rows = self.table.gather_cpu(ngram_ids.reshape(-1).to("cpu", torch.int64))
        staging[:num_rows].copy_(rows, non_blocking=True)
        if padded_rows > num_rows:
            staging[num_rows:padded_rows].zero_()

    def staged_rows(self, num_rows: int) -> torch.Tensor:
        """View of the first ``num_rows`` staged rows in the table dtype."""
        assert self._staging is not None
        return self._staging[:num_rows].view(self.table.dtype)

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Synchronous gather (eager only: the id copy syncs the stream)."""
        if self.is_staged:
            return self.staged_rows(ngram_ids.numel()).view(
                *ngram_ids.shape, self.embedding_dim
            )
        rows = self.table.gather_cpu(ngram_ids.reshape(-1).to("cpu", torch.int64))
        device_rows = rows.to(ngram_ids.device, non_blocking=True)
        return device_rows.view(self.table.dtype).view(
            *ngram_ids.shape, self.embedding_dim
        )


__all__ = [
    "PLEDiskTable",
    "Qwen4ExpPLEDiskEmbedding",
    "layer_index_from_prefix",
    "ple_disk_enabled",
    "ple_staged_enabled",
]
