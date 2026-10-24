# SPDX-License-Identifier: Apache-2.0
"""[GB10] GPU round trip of the CuTeDSL AOT cache (cutedsl_aot_cache.py).

Phase 1 (child process, empty cache dir): every unit compiles through the
patched path and is exported (a miss). Phase 2 (a fresh child process, same
dir): every unit loads from the cache (a hit). In each phase the cached
kernel's output must equal an eager compile's (cache off) bit for bit, both
called directly and captured in a ``torch.cuda.graph`` on a side stream and
replayed. Units: shape-dynamic skinny GEMM (m=4, k=4096, compiled for n=4096
and also run at a second n) and Qwen4Exp HC down+SiLU (FMA m=2 and MMA m=8;
rank=316, hc=8, k=10240, N=324).

Run in a throwaway container from the image under test, with the changed
files bind-mounted over site-packages:
    docker run --rm --gpus all -m 6g --tmpfs /w:size=1g -e CUTE_DSL_ARCH=sm_121a \
        ... python3 tests/gb10/check_cutedsl_aot_gpu.py [cache_dir]
Prints PASS/FAIL lines; exits nonzero on any failure.
"""

import os
import subprocess
import sys
from contextlib import contextmanager

CACHE_ENV = "VLLM_CUTEDSL_AOT_CACHE_DIR"
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {label}{' ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


@contextmanager
def cache_off():
    saved = os.environ.pop(CACHE_ENV)
    try:
        yield
    finally:
        os.environ[CACHE_ENV] = saved


def graph_replay(fn):
    """Capture fn() in a CUDA graph on a side stream, replay, return the output."""
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()  # warm the allocator outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        out = fn()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    return out


def run_phase(phase: str) -> None:
    import torch

    from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
        ShapeDynamicSkinnyGemm,
    )
    from vllm.model_executor.warmup import cutedsl_aot_cache as aot
    from vllm.models.qwen4_exp.nvidia.ops.cute_dsl.hc_down_silu import (
        HcDownSiluGemm,
    )

    aot.reset_counters()
    torch.manual_seed(0)
    dtype = torch.bfloat16

    # Skinny GEMM: cached instance vs an eager-compiled one, same config.
    m, k = 4, 4096
    cached = ShapeDynamicSkinnyGemm()
    cfg = cached._config(m, 4096, k)
    cached._compile(dtype, cfg, False)
    with cache_off():
        eager = ShapeDynamicSkinnyGemm()
        eager._compile(dtype, cfg, False)
    a = torch.randn(m, k, dtype=dtype, device="cuda")
    for n in (4096, 2048):
        b = torch.randn(n, k, dtype=dtype, device="cuda")
        ref = eager(a, b, cfg)
        out = cached(a, b, cfg)
        torch.cuda.synchronize()
        check(f"{phase} skinny n={n} eager-equal", torch.equal(out, ref))
        replayed = graph_replay(lambda b=b: cached(a, b, cfg))
        check(f"{phase} skinny n={n} graph-replay-equal", torch.equal(replayed, ref))

    # HC down+SiLU: FMA (m=2) and MMA (m=8) keys.
    rank, hc, hk, big_n = 316, 8, 10240, 324
    cached_hc = HcDownSiluGemm(rank, hc, hk)
    with cache_off():
        eager_hc = HcDownSiluGemm(rank, hc, hk)
    w = torch.randn(big_n, hk, dtype=dtype, device="cuda") * 0.02
    cols = rank + hc
    for hm in (2, 8):
        key = cached_hc.dispatch(hm)
        cached_hc.compile(key)
        with cache_off():
            eager_hc.compile(key)
        x = torch.randn(hm, hk, dtype=dtype, device="cuda")
        ref = eager_hc(x, w)[:, :cols]
        out = cached_hc(x, w)[:, :cols]
        torch.cuda.synchronize()
        check(f"{phase} hc {key[0]} m={hm} eager-equal", torch.equal(out, ref))
        replayed = graph_replay(lambda x=x: cached_hc(x, w))[:, :cols]
        check(
            f"{phase} hc {key[0]} m={hm} graph-replay-equal",
            torch.equal(replayed, ref),
        )

    counters = aot.snapshot_counters()
    # The eager compiles run with the cache off and count as compiled too.
    units = 3
    if phase == "miss":
        expected = {"hits": 0, "compiled": 2 * units, "fallbacks": 0}
    else:
        expected = {"hits": units, "compiled": units, "fallbacks": 0}
    check(f"{phase} counters", counters == expected, f"{counters}")
    loaded = type(cached._compiled[(dtype, cfg, False)]).__name__
    print(f"INFO {phase} skinny compiled-object type: {loaded}")
    sys.exit(1 if failures else 0)


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--phase":
        run_phase(sys.argv[2])
        return
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else "/w/cutedsl-aot"
    os.makedirs(cache_dir, exist_ok=True)
    check("cache dir starts empty", not os.listdir(cache_dir), cache_dir)
    env = {**os.environ, CACHE_ENV: cache_dir}
    for phase in ("miss", "hit"):
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--phase", phase], env=env
        )
        check(f"{phase} phase process", result.returncode == 0)
        if phase == "miss":
            stored = sorted(f for f in os.listdir(cache_dir) if f.endswith(".o"))
            leftovers = [f for f in os.listdir(cache_dir) if ".tmp-" in f]
            check("miss exported 3 files", len(stored) == 3, f"{stored}")
            check("no temp files left", not leftovers, f"{leftovers}")
    print("ALL PASS" if not failures else f"FAILED: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
