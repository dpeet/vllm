# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the [GB10] CuTeDSL AOT cache (cutedsl_aot_cache.py).

Fake compile, export and load callables stand in for CuTeDSL, so no GPU is
needed; the skinny GEMM and HC down+SiLU tests drive the real wrapped
``_compile``/``compile`` methods with only the CuTeDSL compile faked.

Usage (in the image under test):
    python3 -m pytest tests/gb10/test_cutedsl_aot_cache_cpu.py
"""

import os
from types import ModuleType

import pytest

from vllm.model_executor.warmup import cutedsl_aot_cache as aot

KIND = "test-kernel"


class FakeCompiled:
    def __init__(self, tag: str = "compiled"):
        self.tag = tag


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    path = tmp_path / "aot"
    monkeypatch.setenv(aot.CACHE_DIR_ENV, str(path))
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_121a")
    monkeypatch.setattr(aot, "FORK_COMMIT_FILE", str(tmp_path / "no-fork-commit"))
    monkeypatch.setenv("VLLM_FORK_COMMIT", "abc123")
    aot.reset_counters()
    return path


@pytest.fixture
def fake_io(monkeypatch):
    """Fake export (writes the symbol name) and load (reads it back)."""
    calls = {"export": [], "load": []}

    def export(compiled, path, name):
        calls["export"].append(path)
        with open(path, "w") as f:
            f.write(name)

    def load(path, name):
        calls["load"].append(path)
        with open(path) as f:
            if f.read() != name:
                raise RuntimeError("corrupt object file")
        return ("loaded", name)

    monkeypatch.setattr(aot, "_export", export)
    monkeypatch.setattr(aot, "_load_function", load)
    return calls


@pytest.fixture
def source_module(tmp_path):
    src = tmp_path / "kernel_src.py"
    src.write_text("X = 1\n")
    module = ModuleType("kernel_src")
    module.__file__ = str(src)
    return module


def counting_compile(tag="compiled"):
    calls = []

    def compile_fn():
        calls.append(1)
        return FakeCompiled(tag)

    return compile_fn, calls


def files(path):
    return sorted(os.listdir(path)) if path.exists() else []


def test_disabled_when_env_unset(tmp_path, monkeypatch, fake_io, source_module):
    monkeypatch.delenv(aot.CACHE_DIR_ENV, raising=False)
    compile_fn, calls = counting_compile()
    result = aot.cached_cute_compile(KIND, ("k",), compile_fn, [source_module])
    assert isinstance(result, FakeCompiled)
    assert calls == [1]
    assert fake_io == {"export": [], "load": []}
    assert files(tmp_path) == ["kernel_src.py"]


def test_miss_exports_then_hit_loads(cache_dir, fake_io, source_module):
    compile_fn, calls = counting_compile()
    first = aot.cached_cute_compile(KIND, ("k",), compile_fn, [source_module])
    assert isinstance(first, FakeCompiled) and calls == [1]
    name = aot.cache_symbol_name(KIND, ("k",), [source_module])
    assert name.startswith("cutedsl_") and len(name) == len("cutedsl_") + 16
    assert files(cache_dir) == [f"{name}.o"]

    second = aot.cached_cute_compile(KIND, ("k",), compile_fn, [source_module])
    assert second == ("loaded", name)
    assert calls == [1]
    assert aot.snapshot_counters() == {"hits": 1, "compiled": 1, "fallbacks": 0}


def test_export_goes_to_temp_file_then_rename(cache_dir, fake_io, source_module):
    compile_fn, _ = counting_compile()
    aot.cached_cute_compile(KIND, ("k",), compile_fn, [source_module])
    name = aot.cache_symbol_name(KIND, ("k",), [source_module])
    (tmp,) = fake_io["export"]
    assert os.path.dirname(tmp) == str(cache_dir)
    assert tmp != str(cache_dir / f"{name}.o")
    assert tmp.endswith(".o")  # export_to_c picks the format from the suffix
    assert not os.path.exists(tmp)


def test_export_error_leaves_no_file(cache_dir, monkeypatch, source_module):
    def failing_export(compiled, path, name):
        with open(path, "w") as f:
            f.write("partial")
        raise RuntimeError("export blew up")

    monkeypatch.setattr(aot, "_export", failing_export)
    compile_fn, calls = counting_compile()
    result = aot.cached_cute_compile(KIND, ("k",), compile_fn, [source_module])
    assert isinstance(result, FakeCompiled) and calls == [1]
    assert files(cache_dir) == []
    assert aot.snapshot_counters() == {"hits": 0, "compiled": 1, "fallbacks": 1}


def test_corrupt_file_falls_back_and_is_replaced(cache_dir, fake_io, source_module):
    name = aot.cache_symbol_name(KIND, ("k",), [source_module])
    cache_dir.mkdir()
    (cache_dir / f"{name}.o").write_text("garbage")
    compile_fn, calls = counting_compile()
    result = aot.cached_cute_compile(KIND, ("k",), compile_fn, [source_module])
    assert isinstance(result, FakeCompiled) and calls == [1]
    assert aot.snapshot_counters() == {"hits": 0, "compiled": 1, "fallbacks": 1}
    assert (cache_dir / f"{name}.o").read_text() == name

    again = aot.cached_cute_compile(KIND, ("k",), compile_fn, [source_module])
    assert again == ("loaded", name) and calls == [1]


def test_unloadable_file_without_fake_load(cache_dir, monkeypatch, source_module):
    """The real loader on a garbage file raises; the cache must still compile."""
    monkeypatch.setattr(aot, "_export", lambda compiled, path, name: None)
    name = aot.cache_symbol_name(KIND, ("k",), [source_module])
    cache_dir.mkdir()
    (cache_dir / f"{name}.o").write_bytes(b"\x00not an object file")
    compile_fn, calls = counting_compile()
    result = aot.cached_cute_compile(KIND, ("k",), compile_fn, [source_module])
    assert isinstance(result, FakeCompiled) and calls == [1]
    assert aot.snapshot_counters()["fallbacks"] >= 1


def test_name_changes_with_every_key_input(cache_dir, monkeypatch, source_module):
    base = ("bf16", "cfg", False, True, "--opts")
    names = {aot.cache_symbol_name(KIND, base, [source_module])}

    def add(name):
        assert name not in names
        names.add(name)

    for i in range(len(base)):
        changed = list(base)
        changed[i] = f"other-{i}"
        add(aot.cache_symbol_name(KIND, tuple(changed), [source_module]))
    add(aot.cache_symbol_name("other-kind", base, [source_module]))

    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")
    add(aot.cache_symbol_name(KIND, base, [source_module]))
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_121a")

    monkeypatch.setenv("VLLM_FORK_COMMIT", "def456")
    add(aot.cache_symbol_name(KIND, base, [source_module]))
    fork_file = cache_dir.parent / "FORK_COMMIT"
    fork_file.write_text("filecommit\n")
    monkeypatch.setattr(aot, "FORK_COMMIT_FILE", str(fork_file))
    add(aot.cache_symbol_name(KIND, base, [source_module]))

    versions = aot._package_versions()
    monkeypatch.setattr(
        aot, "_package_versions", lambda: versions + (("quack", "9.9.9"),)
    )
    add(aot.cache_symbol_name(KIND, base, [source_module]))

    with open(source_module.__file__, "a") as f:
        f.write("Y = 2\n")
    add(aot.cache_symbol_name(KIND, base, [source_module]))


def test_hc_instances_differing_in_rank_or_prefetch(cache_dir, fake_io, monkeypatch):
    hc_mod = pytest.importorskip(
        "vllm.models.qwen4_exp.nvidia.ops.cute_dsl.hc_down_silu"
    )
    pytest.importorskip("vllm.models.qwen4_exp.nvidia.ops.cute_dsl._hc_down_silu_fma")
    pytest.importorskip("vllm.models.qwen4_exp.nvidia.ops.cute_dsl._hc_down_silu_mma")
    compiled_options = []

    def fake_uncached(self, compile_key, options):
        compiled_options.append(options)
        return FakeCompiled(repr(compile_key))

    monkeypatch.setattr(hc_mod.HcDownSiluGemm, "_compile_uncached", fake_uncached)
    instances = [
        hc_mod.HcDownSiluGemm(316, 8, 10240),
        hc_mod.HcDownSiluGemm(317, 8, 10240),
        hc_mod.HcDownSiluGemm(316, 8, 10240, prefetch_pdl_weights=True),
        hc_mod.HcDownSiluGemm(316, 9, 10240),
        hc_mod.HcDownSiluGemm(316, 8, 10240 + 8),
    ]
    for g in instances:
        for m in (2, 8):
            g.compile(g.dispatch(m))
    # FMA and MMA keys per instance: every one a distinct file.
    assert len(files(cache_dir)) == 2 * len(instances)
    assert set(compiled_options) == {
        "--enable-tvm-ffi",
        "--enable-tvm-ffi --ptxas-options -maxrregcount=64",
    }

    # A fresh instance with the same inputs hits instead of compiling.
    again = hc_mod.HcDownSiluGemm(316, 8, 10240)
    again.compile(again.dispatch(8))
    assert again._compiled[again.dispatch(8)][0] == "loaded"
    assert aot.snapshot_counters()["hits"] == 1


def test_skinny_gemm_key(cache_dir, fake_io, monkeypatch):
    torch = pytest.importorskip("torch")
    sg = pytest.importorskip("vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm")
    pytest.importorskip("vllm.model_executor.kernels.linear.cute_dsl._skinny_gemm")
    use_pdl = {"value": True}
    monkeypatch.setattr(
        sg.ShapeDynamicSkinnyGemm, "_use_pdl", staticmethod(lambda: use_pdl["value"])
    )

    def fake_uncached(self, dtype, config, has_residual, pdl, options):
        assert pdl is use_pdl["value"]
        return FakeCompiled()

    monkeypatch.setattr(sg.ShapeDynamicSkinnyGemm, "_compile_uncached", fake_uncached)
    g = sg.ShapeDynamicSkinnyGemm()
    cfg = g._config(4, 4096, 4096)
    g._compile(torch.bfloat16, cfg, False)
    g._compile(torch.float16, cfg, False)
    g._compile(torch.bfloat16, cfg, True)
    g._compile(torch.bfloat16, g._config(4, 2304, 4096), False)
    use_pdl["value"] = False
    g._compile(torch.bfloat16, cfg, False)
    assert len(files(cache_dir)) == 5

    use_pdl["value"] = True
    g2 = sg.ShapeDynamicSkinnyGemm()
    g2._compile(torch.bfloat16, cfg, False)
    assert g2._compiled[(torch.bfloat16, cfg, False)][0] == "loaded"
