# SPDX-License-Identifier: Apache-2.0
"""CPU-only check of the Qwen4Exp load-time weight-name filters on a real checkpoint.

Extracts the committed functions with ast (no torch/vLLM import), runs them against
the checkpoint's model.safetensors.index.json and shard headers, and asserts that
every name the MTP draft head or the target model consumes stays in the files read,
and that the draft resolves to the lazy safetensors iterator when the target's load
format is a whole-file loader (instanttensor, fastsafetensors).

Usage: uv run --no-project python tests/gb10/check_weight_name_filter_cpu.py [ckpt_dir]
"""

import ast
import json
import logging
import os
import struct
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CKPT = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/opt/ai/artisanal-inference/models/Qwen3.8-Flash-Next-NVFP4-nvidia"
)
GIB = 1024**3


def load_functions(path: Path, names: set[str], namespace: dict) -> dict:
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names, f"missing in {path}: {names}"
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class Qwen4ExpPLEDiskEmbedding:  # stand-in for the isinstance check
    pass


class FakeModel:
    def __init__(self, ple_on_disk: bool):
        self._mods = (
            [object(), Qwen4ExpPLEDiskEmbedding()] if ple_on_disk else [object()]
        )

    def modules(self):
        return iter(self._mods)


ns = {
    "os": os,
    "json": json,
    "Callable": Callable,
    "logger": logging.getLogger("check"),
    "SAFE_WEIGHTS_INDEX_NAME": "model.safetensors.index.json",
    "nn": type("nn", (), {"Module": object}),
    "Qwen4ExpPLEDiskEmbedding": Qwen4ExpPLEDiskEmbedding,
}
load_functions(
    REPO / "vllm/model_executor/model_loader/weight_utils.py",
    {"filter_safetensors_files_by_weight_names"},
    ns,
)
load_functions(
    REPO / "vllm/models/qwen4_exp/nvidia/mtp.py", {"_remap_mtp_weight_name"}, ns
)
load_functions(
    REPO / "vllm/models/qwen4_exp/nvidia/model.py", {"qwen4_exp_main_weight_kept"}, ns
)
filter_files = ns["filter_safetensors_files_by_weight_names"]
remap_mtp = ns["_remap_mtp_weight_name"]
main_kept = ns["qwen4_exp_main_weight_kept"]


def header(path: str) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


folder = str(CKPT)
all_files = sorted(str(p) for p in CKPT.glob("*.safetensors"))
with open(CKPT / "model.safetensors.index.json") as index_file:
    weight_map = json.load(index_file)["weight_map"]
headers = {f: header(f) for f in all_files}
for f, h in headers.items():  # the index must agree with the shard headers
    assert all(weight_map[n] == os.path.basename(f) for n in h), f
assert sum(len(h) for h in headers.values()) == len(weight_map)


def nbytes(names_by_file: dict[str, list[str]]) -> int:
    total = 0
    for f, names in names_by_file.items():
        for n in names:
            a, b = headers[f][n]["data_offsets"]
            total += b - a
    return total


def report(label: str, keep: Callable[[str], bool]) -> list[str]:
    kept = filter_files(all_files, folder, keep)
    for name, file_name in weight_map.items():
        if keep(name):
            assert os.path.join(folder, file_name) in kept, (label, name)
    file_bytes = sum(os.path.getsize(f) for f in kept)
    read = {f: [n for n in headers[f] if keep(n)] for f in kept}
    print(f"{label}: {len(kept)}/{len(all_files)} files, {file_bytes / GIB:.2f} GiB")
    for f in kept:
        print(f"    {os.path.basename(f)}  {os.path.getsize(f) / GIB:.2f} GiB")
    print(
        f"    kept tensors {sum(map(len, read.values()))} = "
        f"{nbytes(read) / GIB:.2f} GiB (bytes the lazy mmap path reads)"
    )
    return kept


total = sum(os.path.getsize(f) for f in all_files)
print(
    f"checkpoint: {len(all_files)} files, {total / GIB:.2f} GiB, "
    f"{len(weight_map)} tensors"
)

mtp_files = report("MTP draft", lambda n: remap_mtp(n) is not None)
assert {os.path.basename(f) for f in mtp_files} == {
    "model-00009-of-00010.safetensors",
    "model-00010-of-00010.safetensors",
    "model-fp8-mtp-ple.safetensors",
}
assert remap_mtp("lm_head.weight") and remap_mtp(
    "model.language_model.embed_tokens.weight"
)

disk = FakeModel(ple_on_disk=True)
main_disk = report("target, PLE on disk", lambda n: main_kept(disk, n))
assert all(os.path.basename(f) != "model-fp8-mtp-ple.safetensors" for f in main_disk)
rejected = [n for n in weight_map if not main_kept(disk, n)]
assert all(
    n.startswith("mtp.")
    or ".ngram_embedding.shard_" in n
    or n.endswith(".ngram_embedding.weight_scale")
    for n in rejected
)
assert sum(".ngram_embedding.shard_" in n for n in rejected) == 128
scale = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
assert not main_kept(disk, scale)
resident = FakeModel(ple_on_disk=False)
assert main_kept(resident, scale)
assert all(main_kept(resident, n) for n in weight_map if not n.startswith("mtp."))

# The draft resolves to the lazy safetensors iterator under a whole-file target
# load format; the target (no load_filtered_weights_lazily) keeps its format.
loader_py = REPO / "vllm/model_executor/model_loader/default_loader.py"
loader_tree = ast.parse(loader_py.read_text())
whole_file = next(
    n
    for n in loader_tree.body
    if isinstance(n, ast.Assign) and n.targets[0].id == "WHOLE_FILE_LOAD_FORMATS"
)
ns["WHOLE_FILE_LOAD_FORMATS"] = ast.literal_eval(whole_file.value)
load_functions(loader_py, {"resolve_filtered_load_format"}, ns)
resolve = ns["resolve_filtered_load_format"]
mtp_filter = lambda n: remap_mtp(n) is not None  # noqa: E731
for fmt in ("instanttensor", "fastsafetensors"):
    assert resolve(fmt, mtp_filter, True) == "safetensors", fmt
    assert resolve(fmt, mtp_filter, False) == fmt, fmt  # target with its filter
    assert resolve(fmt, None, True) == fmt, fmt
for fmt in ("auto", "safetensors", "npcache", "pt"):
    assert resolve(fmt, mtp_filter, True) == fmt, fmt


def class_flag(path: Path, cls: str, flag: str = "load_filtered_weights_lazily"):
    tree = ast.parse(path.read_text())
    body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    for n in body.body:
        if isinstance(n, ast.Assign) and n.targets[0].id == flag:
            return ast.literal_eval(n.value)
    return None


assert class_flag(REPO / "vllm/models/qwen4_exp/nvidia/mtp.py", "Qwen4ExpMTP") is True
for cls in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"):
    assert class_flag(REPO / "vllm/models/qwen4_exp/nvidia/model.py", cls) is None

# _get_weights_iterator must dispatch on the resolved format, not the LoadConfig's,
# and give the lazy branch the "lazy" strategy and the filter.
loader_cls = next(
    n
    for n in loader_tree.body
    if isinstance(n, ast.ClassDef) and n.name == "DefaultModelLoader"
)
get_iter = next(
    n
    for n in loader_cls.body
    if isinstance(n, ast.FunctionDef) and n.name == "_get_weights_iterator"
)
format_names = ("npcache", *ns["WHOLE_FILE_LOAD_FORMATS"])
for node in ast.walk(get_iter):
    if (
        isinstance(node, ast.Compare)
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value in format_names
    ):
        assert isinstance(node.left, ast.Name), ast.unparse(node)
        assert node.left.id == "load_format", ast.unparse(node)
lazy_branch = next(
    n
    for n in ast.walk(get_iter)
    if isinstance(n, ast.If) and ast.unparse(n.test) == "lazy_filtered"
)
lazy_call = lazy_branch.body[0].value
assert ast.unparse(lazy_call.func) == "safetensors_weights_iterator"
assert ast.literal_eval(lazy_call.args[2]) == "lazy"
assert {k.arg for k in lazy_call.keywords} >= {"weight_name_filter"}
print("draft load format: instanttensor/fastsafetensors -> safetensors (lazy) OK")

with tempfile.TemporaryDirectory() as d:  # no index: unchanged
    files = [os.path.join(d, "a.safetensors"), os.path.join(d, "b.safetensors")]
    assert filter_files(files, d, lambda n: False) == files
print("OK")
