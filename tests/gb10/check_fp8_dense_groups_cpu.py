# SPDX-License-Identifier: Apache-2.0
"""CPU check of QWEN38_FP8_DENSE (the block-FP8 cast of BF16 dense linears).

Needs an importable vLLM carrying this fork (run it in the image under test, or with
the fork's Python sources on PYTHONPATH over a vLLM install); reads only the
checkpoint's index and hf_quant_config.json, never a weight. Runs against a
MIXED_PRECISION checkpoint (ModelOptMixedPrecisionConfig) or a plain NVFP4 one
(ModelOptNvFp4Config, whose BF16 linears are its exclude_modules). Asserts:
  1. each group (gdn, attn, shared) matches at least one vLLM linear prefix derived
     from the checkpoint index (packed names applied), in the target, and attn and
     shared also in the MTP drafter;
  2. the real config built from hf_quant_config.json returns
     Fp8PerBlockOnlineLinearMethod for every one of those prefixes (counted per
     group) at its unquantized exits (excluded layers, and for MIXED_PRECISION the
     MTP layers not in quantized_layers), and never for routers, gates, in_proj_ba,
     the indexer and lm_head;
  3. mtp gates the drafter, unset/empty changes nothing, lm_head and unknown groups
     raise ValueError, and a fused QSA+indexer linear is refused.

Usage: python3 tests/gb10/check_fp8_dense_groups_cpu.py [ckpt_dir]
"""

import inspect
import json
import os
import re
import sys
import types
from pathlib import Path
from unittest import mock

import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import modelopt as mo
from vllm.model_executor.layers.quantization.online import fp8 as online_fp8
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.utils import configure_quant_config

# model.py imports qsa.py, which now imports modelopt's group parser (qsa.py alone
# hits the pre-existing qsa <-> model import cycle, so import model first).
from vllm.models.qwen4_exp.nvidia import model as qwen4_model  # isort: skip
from vllm.models.qwen4_exp.nvidia import mtp as mtp_mod
from vllm.models.qwen4_exp.nvidia import qsa as qwen4_qsa

Qwen4ExpForConditionalGeneration = qwen4_model.Qwen4ExpForConditionalGeneration

CKPT = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/opt/ai/artisanal-inference/models/Qwen3.8-Flash-Next-NVFP4-nvidia"
)
failures: list[str] = []


def check(ok: bool, label: str) -> None:
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        failures.append(label)


def raises(fn, label: str) -> None:
    try:
        fn()
    except ValueError as e:
        check(True, f"{label}: raised ValueError ({e})")
        return
    check(False, f"{label}: did not raise")


class FakeLinear(LinearBase):
    """Only isinstance and the type name matter to the cast's selection."""

    def __init__(self):  # skip LinearBase.__init__ (no weights, no TP groups)
        pass


class Qwen4ExpQSAQKVIndexerLinear(FakeLinear):  # name-matched, like the fused layer
    pass


class FakeHead(ParallelLMHead):
    def __init__(self):
        pass


def with_env(value: str | None):
    env = {} if value is None else {"QWEN38_FP8_DENSE": value}
    patch = mock.patch.dict(os.environ, env)
    if value is None:
        os.environ.pop("QWEN38_FP8_DENSE", None)
    return patch


# Checkpoint names -> the vLLM linear prefixes they load into (packed leaves).
weight_map = json.loads((CKPT / "model.safetensors.index.json").read_text())[
    "weight_map"
]
packed = Qwen4ExpForConditionalGeneration.packed_modules_mapping
leaf_to_fused = {
    shard: fused
    for fused, shards in packed.items()
    for shard in shards
    if "." not in shard  # indexer.index_qk_proj stays its own linear when unfused
}
prefixes: set[str] = set()
for name in weight_map:
    if not name.endswith(".weight") or ".experts." in name:
        continue
    module = name.removesuffix(".weight")
    parent, dot, leaf = module.rpartition(".")
    prefixes.add(f"{parent}{dot}{leaf_to_fused.get(leaf, leaf)}")
target = {p for p in prefixes if not p.startswith("mtp.")}
drafter = prefixes - target

matched: dict[str, set[str]] = {}
for group, pattern in mo._QWEN38_FP8_DENSE_PATTERNS.items():
    matched[group] = {p for p in prefixes if pattern.search(p)}
    t = len(matched[group] & target)
    d = len(matched[group] & drafter)
    check(t >= 1, f"group {group} matches {t} target prefixes, {d} drafter prefixes")
    if group in ("attn", "shared"):
        check(d >= 1, f"group {group} matches the MTP drafter")
all_cast = set().union(*matched.values())
print(f"all groups: {len(all_cast)} linears ({len(all_cast & target)} target)")
never = [
    p
    for p in prefixes
    if re.search(r"\.(mlp\.gate|shared_expert_gate|in_proj_ba|index_qk_proj)$", p)
    or "hyper_connection" in p
    or p == "lm_head"
    or "embed_tokens" in p
    or ".ple." in p
]
check(len(never) > 0 and not (set(never) & all_cast), "no router/gate/HC/PLE match")

# The online method reads the model dtype from the current vLLM config at init.
fake_vllm_config = types.SimpleNamespace(
    model_config=types.SimpleNamespace(dtype=torch.bfloat16)
)
mock.patch.object(
    online_fp8, "get_current_vllm_config", return_value=fake_vllm_config
).start()

# The real config: its unquantized exits now return the block-FP8 method.
quant_cfg = json.loads((CKPT / "hf_quant_config.json").read_text())
quant_algo = quant_cfg["quantization"]["quant_algo"]
config_cls = {
    "MIXED_PRECISION": mo.ModelOptMixedPrecisionConfig,
    "NVFP4": mo.ModelOptNvFp4Config,
}[quant_algo]
config = config_cls.from_config(quant_cfg)
print(f"quant_algo {quant_algo} -> {config_cls.__name__}")
config.packed_modules_mapping = dict(packed)
# As model init does: the config's checkpoint names move to vLLM's.
mapper = Qwen4ExpForConditionalGeneration.hf_to_vllm_mapper
config.apply_vllm_mapper(mapper)

# The drafter gets its own config, as _make_draft_vllm_config builds it: the MTP
# class's mappings, then exclude_modules and quantized_layers renumbered from the
# checkpoint's mtp.layers.0 to the drafter's mtp.layers.<num_hidden_layers>.
hf_config = json.loads((CKPT / "config.json").read_text())
mtp_start = hf_config.get("text_config", hf_config)["num_hidden_layers"]
draft_config = config_cls.from_config(quant_cfg)
configure_quant_config(draft_config, mtp_mod.Qwen4ExpMTP)
draft_config.exclude_modules = mtp_mod._remap_ignored_layers(
    draft_config.exclude_modules, mtp_start
)
if getattr(draft_config, "quantized_layers", None):
    draft_config.quantized_layers = mtp_mod._remap_quantized_layers(
        draft_config.quantized_layers, mtp_start
    )


def method_for(prefix: str, layer=None):
    if prefix.startswith("mtp."):
        prefix = mtp_mod._remap_ignored_layers([prefix], mtp_start)[0]
        return draft_config.get_quant_method(layer or FakeLinear(), prefix)
    return config.get_quant_method(layer or FakeLinear(), prefix)


def is_loaded(p: str) -> bool:
    """False for a name the mapper drops (never loaded)."""
    return bool(mapper.apply_list([p]))


def vllm_prefix(p: str) -> str:
    if p.startswith("mtp."):  # the drafter's names; method_for renumbers them
        return p
    (mapped,) = mapper.apply_list([p])
    return mapped


sample_target = {g: sorted(matched[g] & target)[0] for g in matched}
sample_drafter = {g: sorted(matched[g] & drafter)[0] for g in ("attn", "shared")}
with with_env(None):
    for g, p in sample_target.items():
        m = method_for(vllm_prefix(p))
        check(type(m) is UnquantizedLinearMethod, f"unset: {p} stays unquantized")
    check(mo.qwen38_fp8_dense_groups() == frozenset(), "unset: no groups")
with with_env(""):
    check(mo.qwen38_fp8_dense_groups() == frozenset(), "empty: no groups")
with with_env(" , "):
    check(mo.qwen38_fp8_dense_groups() == frozenset(), "blank items: no groups")

with with_env("gdn,attn,shared"):
    for g, p in sample_target.items():
        vp = vllm_prefix(p)
        excluded = config.is_layer_excluded(vp)
        m = method_for(vp)
        check(
            isinstance(m, Fp8PerBlockOnlineLinearMethod),
            f"{g}: {vp} -> {type(m).__name__} (excluded={excluded})",
        )
    for g, p in sample_drafter.items():
        m = method_for(p)
        check(type(m) is UnquantizedLinearMethod, f"no mtp group: {p} stays BF16")
    cast_never = [
        p
        for p in sorted(never)
        if is_loaded(p)
        and isinstance(
            method_for(vllm_prefix(p), FakeHead() if p == "lm_head" else None),
            Fp8PerBlockOnlineLinearMethod,
        )
    ]
    kinds = sorted({re.sub(r"\d+", "N", p) for p in never})
    print(f"    {len(never)} router/gate/in_proj_ba/indexer/HC/PLE/embed/head names:")
    for kind in kinds:
        print(f"      {kind}")
    check(not cast_never, f"none of them cast at get_quant_method ({cast_never[:3]})")
    check(
        any(p.endswith(".in_proj_ba") for p in never)
        and any(p.endswith(".index_qk_proj") for p in never)
        and any(p.endswith(".mlp.gate") for p in never)
        and "lm_head" in never,
        "the never-cast set covers in_proj_ba, the indexer, the router and lm_head",
    )
    raises(
        lambda: mo._qwen38_fp8_dense_method(
            Qwen4ExpQSAQKVIndexerLinear(), "x.layers.3.self_attn.qkv_proj"
        ),
        "fused QSA+indexer linear refused",
    )
with with_env("attn,shared,mtp"):
    for g, p in sample_drafter.items():
        excluded = draft_config.is_layer_excluded(
            mtp_mod._remap_ignored_layers([p], mtp_start)[0]
        )
        m = method_for(p)
        check(
            isinstance(m, Fp8PerBlockOnlineLinearMethod),
            f"mtp+{g}: {p} -> {type(m).__name__} (excluded={excluded})",
        )
with with_env("gdn,attn,shared,mtp"):
    # Every matched prefix, not a sample: on plain NVFP4 a prefix the exclude list
    # missed would get the NVFP4 method and fail to load its BF16 weight.
    total = 0
    for g in sorted(matched):
        loaded = sorted(p for p in matched[g] if p in drafter or is_loaded(p))
        cast = [
            p
            for p in loaded
            if isinstance(method_for(vllm_prefix(p)), Fp8PerBlockOnlineLinearMethod)
        ]
        total += len(cast)
        t = sum(not p.startswith("mtp.") for p in cast)
        check(
            len(cast) == len(loaded),
            f"all groups+mtp: {g} casts {len(cast)}/{len(loaded)} "
            f"({t} target, {len(cast) - t} drafter)",
        )
    print(f"all groups+mtp: {total} linears cast at get_quant_method")
if config.get_name() == "modelopt_fp4":
    # qsa.py builds qkv_proj without a modelopt_fp4 config unless attn is set,
    # because without the config it never reaches get_quant_method.
    check(
        qwen4_model.without_modelopt_fp4(config) is None,
        "modelopt_fp4: without_modelopt_fp4 drops the config",
    )
    qsa_src = inspect.getsource(qwen4_qsa)
    check(
        'quant_config\n                if "attn" in qwen38_fp8_dense_groups()\n'
        "                else model.without_modelopt_fp4(quant_config)" in qsa_src,
        "modelopt_fp4: QSA qkv_proj keeps the config when attn is set",
    )
with with_env("mtp"):
    m = method_for(sample_drafter["attn"])
    check(type(m) is UnquantizedLinearMethod, "mtp alone casts nothing")

for bad in ("lm_head", "gdn,lm_head", "bogus", "gdn,attnx"):
    with with_env(bad):
        raises(mo.qwen38_fp8_dense_groups, f"QWEN38_FP8_DENSE={bad!r}")
        raises(
            lambda: method_for(vllm_prefix(sample_target["gdn"])),
            f"QWEN38_FP8_DENSE={bad!r} at get_quant_method",
        )

print("ALL PASS" if not failures else f"FAIL ({len(failures)})")
sys.exit(1 if failures else 0)
