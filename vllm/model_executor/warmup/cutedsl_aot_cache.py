# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[GB10] Opt-in on-disk cache of exported CuTeDSL kernels.

``cute.compile`` always compiles with ``no_cache=True``, so every boot pays
the full CuTeDSL compile for each warmup unit. When
``VLLM_CUTEDSL_AOT_CACHE_DIR`` is set, ``cached_cute_compile`` exports each
compiled kernel as an object file (``export_to_c``) and later boots load it
with ``cute.runtime.load_module(..., enable_tvm_ffi=True)``. Unset, it only
calls ``compile_fn()``, so lanes that don't set it behave as before.

The key covers every input baked into the kernel (the caller's ``key_parts``,
which must include the compile options), the target arch, the versions of the
packages that generate the code, the fork commit and the source of the
modules that define the kernel, because a key that missed one of them would
load a kernel built for something else with no error. Any load or export
failure falls back to compiling, so the cache can only cost time, never a
boot.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import os
import uuid
from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

CACHE_DIR_ENV = "VLLM_CUTEDSL_AOT_CACHE_DIR"
FORK_COMMIT_FILE = "/opt/gb10-flash-next/FORK_COMMIT"
_VERSIONED_PACKAGES = ("vllm", "torch", "tvm_ffi", "nvidia-cutlass-dsl", "quack")

_counters = {"hits": 0, "compiled": 0, "fallbacks": 0}


def snapshot_counters() -> dict[str, int]:
    """Copy of the process-wide hit/compiled/fallback counters."""
    return dict(_counters)


def reset_counters() -> None:
    for name in _counters:
        _counters[name] = 0


@lru_cache(maxsize=1)
def _package_versions() -> tuple[tuple[str, str], ...]:
    versions = []
    for package in _VERSIONED_PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        versions.append((package, version))
    return tuple(versions)


def _fork_commit() -> str:
    try:
        commit = Path(FORK_COMMIT_FILE).read_text().strip()
        if commit:
            return commit
    except OSError:
        pass
    return os.environ.get("VLLM_FORK_COMMIT") or "dev"


def _source_file(module: ModuleType | str) -> str:
    if isinstance(module, ModuleType):
        path = module.__file__
    else:
        spec = importlib.util.find_spec(module)
        path = spec.origin if spec is not None else None
    if not path:
        raise FileNotFoundError(f"no source file for module {module!r}")
    return path


def _source_hash(source_modules: Iterable[ModuleType | str]) -> str:
    digest = hashlib.sha256()
    for module in source_modules:
        digest.update(Path(_source_file(module)).read_bytes())
    return digest.hexdigest()


def cache_symbol_name(
    kind: str, key_parts: Any, source_modules: Iterable[ModuleType | str]
) -> str:
    """The exported symbol (and file stem) for one kernel, a valid C identifier."""
    material = repr(
        (
            kind,
            repr(key_parts),
            os.environ.get("CUTE_DSL_ARCH", ""),
            _package_versions(),
            _fork_commit(),
            _source_hash(source_modules),
        )
    )
    return "cutedsl_" + hashlib.sha256(material.encode()).hexdigest()[:16]


def _load_function(path: str, name: str) -> Any:
    import cutlass.cute as cute

    return getattr(cute.runtime.load_module(path, enable_tvm_ffi=True), name)


def _export(compiled: Any, path: str, name: str) -> None:
    compiled.export_to_c(path, function_name=name)


def _compile(compile_fn: Callable[[], Any]) -> Any:
    _counters["compiled"] += 1
    return compile_fn()


def cached_cute_compile(
    kind: str,
    key_parts: Any,
    compile_fn: Callable[[], Any],
    source_modules: Iterable[ModuleType | str],
) -> Any:
    """Return ``compile_fn()``'s compiled kernel, loading it from disk when cached.

    The returned object is either the compiled function or a loaded
    ``tvm_ffi.Function``; both serve positional calls on the same FFI path.
    """
    cache_dir = os.environ.get(CACHE_DIR_ENV)
    if not cache_dir:
        return _compile(compile_fn)

    try:
        name = cache_symbol_name(kind, key_parts, source_modules)
        os.makedirs(cache_dir, exist_ok=True)
    except Exception as exc:
        _counters["fallbacks"] += 1
        logger.debug("CuTeDSL AOT cache key failed for %s: %r", kind, exc)
        return _compile(compile_fn)

    path = os.path.join(cache_dir, name + ".o")
    if os.path.exists(path):
        try:
            loaded = _load_function(path, name)
            _counters["hits"] += 1
            logger.debug("CuTeDSL AOT cache hit %s (%s)", path, kind)
            return loaded
        except Exception as exc:
            # Recompile below and overwrite the unloadable file.
            _counters["fallbacks"] += 1
            logger.debug("CuTeDSL AOT cache load failed for %s: %r", path, exc)

    compiled = _compile(compile_fn)
    # Same dir + os.replace: a crash or a concurrent boot never leaves a
    # half-written file under the final name. The ".o" suffix matters because
    # export_to_c picks the output format from it.
    tmp = os.path.join(cache_dir, f"{name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}.o")
    try:
        _export(compiled, tmp, name)
        os.replace(tmp, path)
        logger.debug("CuTeDSL AOT cache stored %s (%s)", path, kind)
    except Exception as exc:
        _counters["fallbacks"] += 1
        logger.debug("CuTeDSL AOT cache export failed for %s: %r", path, exc)
        with contextlib.suppress(OSError):
            os.unlink(tmp)
    return compiled
