"""Independent weight-level check of a converted Nemotron-H artifact against the BF16 source.

This decodes the artifact's own bytes from its registered layouts and compares the result
element-wise against the checkpoint tensor the logical parameter came from; it does not use the
converter's quantization code or its logical sources.

Usage:  .venv/bin/python -m tools.verify.check_nemotron_artifact [artifact] [source-dir]
"""

from __future__ import annotations

import collections
import dataclasses
import json
import struct
import sys

import torch

from tools.artifact.codecs.nvfp4 import unswizzle_nvfp4_scales
from tools.artifact.layouts import block_scale_geometry, row_scale_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts
from tools.convert.sources.safetensors import SafetensorsSource

ARTIFACT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/storage/models/nemotron-74b/nemotron_h_74b_nvfp4.ninfer"
SOURCE = sys.argv[2] if len(sys.argv) > 2 else "/mnt/storage/models/nemo/74B"

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_DIRECT = {"bf16": torch.bfloat16, "fp32": torch.float32, "int32": torch.int32}


def object_rows(art: Artifact, obj, row_begin: int, row_end: int) -> torch.Tensor:
    """Decode rows [row_begin, row_end) of one stored object into an FP32 matrix."""
    n, k = obj.shape
    rows = row_end - row_begin
    if obj.format in _DIRECT:
        dtype = _DIRECT[obj.format]
        word = dtype.itemsize
        raw = art.read_range(obj.offset + row_begin * k * word, rows * k * word)
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(rows, k).to(torch.float32)
    if obj.format == "fp8_e4m3fn_row_bf16":
        geometry = row_scale_geometry(obj.format, obj.shape)
        codes = torch.frombuffer(
            bytearray(art.read_range(obj.offset + row_begin * k, rows * k)), dtype=torch.uint8
        ).reshape(rows, k)
        scales = torch.frombuffer(
            bytearray(
                art.read_range(obj.offset + geometry.scale_plane_offset + row_begin * 2, rows * 2)
            ),
            dtype=torch.bfloat16,
        ).float()
        return codes.view(torch.float8_e4m3fn).float() * scales[:, None]
    if obj.format == "nvfp4":
        geometry = block_scale_geometry(obj.format, obj.shape)
        codes = torch.frombuffer(
            bytearray(art.read_range(obj.offset + row_begin * (k // 2), rows * (k // 2))),
            dtype=torch.uint8,
        ).reshape(rows, k // 2)
        stored = torch.frombuffer(
            bytearray(
                art.read_range(obj.offset + geometry.scale_plane_offset, geometry.scale_plane_bytes)
            ),
            dtype=torch.uint8,
        )
        scales = unswizzle_nvfp4_scales(stored, obj.shape)[row_begin:row_end]
        divisor = struct.unpack(
            "<f", art.read_range(obj.offset + geometry.weight_divisor_offset, 4)
        )[0]
        # Two E2M1 codes per byte, the low nibble holding the even element.
        words = torch.stack((codes & 0x0F, codes >> 4), dim=-1).reshape(rows, k)
        magnitude = _E2M1[(words & 0x7).to(torch.int64)]
        signed = torch.where((words & 0x8) != 0, -magnitude, magnitude)
        block = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
        return signed * block / divisor
    raise ValueError(f"{obj.id}: unsupported format {obj.format}")


def parameter_rows(art: Artifact, name: str, first: int, last: int) -> torch.Tensor:
    """Decode rows [first, last) of one bound 2-D parameter, walking its binding parts."""
    pieces = []
    cursor = 0
    for object_id, begin, end in binding_parts(art.directory.bindings[name], art.by_id, name):
        obj = art.object(object_id)
        if len(obj.shape) != 2:
            flat = object_rows(art, dataclasses.replace(obj, shape=(1, obj.shape[0])), 0, 1)
            pieces.append(flat.reshape(-1)[begin:end])
            cursor += end - begin
            continue
        k = obj.shape[1]
        if begin % k or end % k:
            raise ValueError(f"{name}: part [{begin},{end}) is not row aligned in {obj.id}")
        base_row, count = begin // k, (end - begin) // k
        low, high = max(first, cursor), min(last, cursor + count)
        if low < high:
            pieces.append(object_rows(art, obj, base_row + low - cursor, base_row + high - cursor))
        cursor += count
    return torch.cat(pieces)


def parameter(art: Artifact, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    values = parameter_rows(art, name, 0, shape[0] if len(shape) == 2 else 1)
    return values.reshape(shape)


def compare(label: str, stored: torch.Tensor, source: torch.Tensor) -> dict:
    if stored.shape != source.shape:
        raise ValueError(f"{label}: shape {tuple(stored.shape)} vs source {tuple(source.shape)}")
    reference = source.to(torch.float32).reshape(-1)
    value = stored.reshape(-1).float()
    difference = (value - reference).abs()
    norm = float(reference.norm())
    return {
        "element": label,
        "shape": tuple(source.shape),
        "relative_l2": float(difference.norm()) / norm if norm else 0.0,
        "max_abs": float(difference.max()),
        "rms_source": float(reference.pow(2).mean().sqrt()),
        "exact": bool(torch.equal(value, reference)),
    }


def report_row(result: dict) -> None:
    print(
        f"  {result['element']}: rel_l2 {result['relative_l2']:.5f} "
        f"max_abs {result['max_abs']:.4g} rms_source {result['rms_source']:.4g} "
        f"exact {result['exact']}"
    )


def main() -> None:
    art = Artifact.open(ARTIFACT)
    base = SafetensorsSource(SOURCE)
    stem = "language_model.backbone.layers."
    report = json.load(open(ARTIFACT + ".conversion.json"))
    print(
        f"artifact {ARTIFACT}: {report['objects']} objects, "
        f"payload {report['payload_bytes'] / 2**30:.2f} GiB, {report['seconds']:.0f}s"
    )
    print(f"formats: {report['formats']}")

    rows: list[dict] = []

    # 1. Routed experts: every distinct (experts, intermediate) pair the checkpoint has, sampled at
    #    three experts per pair, plus one `down` matrix per pair.
    banks = collections.defaultdict(list)
    for index, block in enumerate(report["components"]["text"]["config"]["block_configs"]):
        if block["block_type"] == "moe":
            banks[(block["n_routed_experts"], block["moe_intermediate_size"])].append(index)
    print(f"\n[1] routed experts: {len(banks)} distinct (experts, width) pairs")
    for (experts, width), layers in sorted(banks.items()):
        layer = layers[0]
        prefix = f"text/layers/{layer}/moe/experts/"
        for expert in sorted({0, experts // 2, experts - 1}):
            stored = parameter(art, f"{prefix}{expert}/up", (width, 1024))
            source = base.read_flat(f"{stem}{layer}.mixer.experts.{expert}.up_proj.weight")
            result = compare(
                f"L{layer} expert {expert}/{experts} up [{width},1024]",
                stored,
                source.to(torch.float32).reshape(width, 1024),
            )
            rows.append(result)
            report_row(result)
        name = f"{prefix}{experts - 1}/down"
        stored = parameter(art, name, (1024, width))
        source = base.read_flat(f"{stem}{layer}.mixer.experts.{experts - 1}.down_proj.weight")
        result = compare(
            f"L{layer} expert {experts - 1}/{experts} down [1024,{width}]",
            stored,
            source.to(torch.float32).reshape(1024, width),
        )
        rows.append(result)
        report_row(result)

    # 2. The shared expert: an NVFP4 projection rather than a bank, at three layers.
    print("\n[2] shared experts (NVFP4 projections)")
    for layer in (1, 31, 87):
        for role, shape, source_name in (
            ("up", (5376, 4096), "shared_experts.up_proj.weight"),
            ("down", (4096, 5376), "shared_experts.down_proj.weight"),
        ):
            name = f"text/layers/{layer}/moe/shared/{role}"
            if name not in art.directory.bindings:
                continue
            stored = parameter(art, name, shape)
            source = base.read_flat(f"{stem}{layer}.mixer.{source_name}").to(torch.float32)
            result = compare(f"L{layer} shared {role} {shape}", stored, source.reshape(shape))
            rows.append(result)
            report_row(result)

    # 3. FP8 projections: the Mamba mixer, attention q/k/o and the head (a row window of the head).
    print("\n[3] FP8 projections")
    fp8 = [
        ("text/layers/0/mamba/in_projection", (18560, 4096), stem + "0.mixer.in_proj.weight", 4096),
        ("text/layers/0/mamba/out_projection", (4096, 8192), stem + "0.mixer.out_proj.weight", 4096),
        ("text/layers/7/attention/query", (4096, 4096), stem + "7.mixer.q_proj.weight", 4096),
        ("text/layers/7/attention/key", (256, 4096), stem + "7.mixer.k_proj.weight", 256),
        ("text/layers/7/attention/output", (4096, 4096), stem + "7.mixer.o_proj.weight", 4096),
        ("text/output_head", (131072, 4096), "language_model.lm_head.weight", 2048),
    ]
    for name, shape, path, window in fp8:
        stored = parameter_rows(art, name, 0, window)
        source = base.read_flat(path).to(torch.float32).reshape(shape)[:window]
        result = compare(f"{name} rows[0:{window})", stored, source)
        rows.append(result)
        report_row(result)

    # 4. BF16 and FP32 parameters, which must be exact: norms and the router stay BF16, and the four
    #    scalars the scan and the router read are the checkpoint's BF16 upcast to FP32.
    print("\n[4] BF16 / FP32 parameters (expected exact)")
    exact = [
        ("text/final_norm", (4096,), "norm_f.weight", None),
        ("text/token_embedding", (131072, 4096), "embeddings.weight", 1024),
        ("text/layers/0/block_norm", (4096,), "layers.0.norm.weight", None),
        ("text/layers/1/moe/router", (240, 4096), "layers.1.mixer.gate.weight", 240),
        ("text/layers/0/mamba/norm", (8192,), "layers.0.mixer.norm.weight", None),
        ("text/layers/0/mamba/convolution_bias", (10240,), "layers.0.mixer.conv1d.bias", None),
        ("text/layers/0/mamba/a_log", (128,), "layers.0.mixer.A_log", None),
        ("text/layers/0/mamba/d_skip", (128,), "layers.0.mixer.D", None),
        ("text/layers/0/mamba/dt_bias", (128,), "layers.0.mixer.dt_bias", None),
        ("text/layers/1/moe/correction_bias", (240,), "layers.1.mixer.gate.e_score_correction_bias", None),
        ("text/layers/1/moe/latent_in", (1024, 4096), "layers.1.mixer.fc1_latent_proj.weight", 1024),
        ("text/layers/1/moe/latent_out", (4096, 1024), "layers.1.mixer.fc2_latent_proj.weight", 4096),
    ]
    for name, shape, source_name, window in exact:
        source = base.read_flat(f"language_model.backbone.{source_name}")
        source = source.to(torch.float32).reshape(shape)
        stored = (
            parameter(art, name, shape)
            if window is None
            else parameter_rows(art, name, 0, window)
        )
        if window is not None:
            source = source[:window]
        result = compare(name, stored, source)
        rows.append(result)
        report_row(result)

    # 5. The depthwise convolution's stored order: the artifact must carry `j * C + c`, the
    #    checkpoint's `[C, 1, K]` transposed.
    print("\n[5] depthwise convolution order")
    stored = parameter(art, "text/layers/0/mamba/convolution", (10240, 4))
    source = base.read_flat(f"{stem}0.mixer.conv1d.weight").to(torch.float32).reshape(10240, 4)
    # The payload is tap-major, so the stored flat order is the checkpoint's transpose.
    result = compare(
        "L0 conv1d payload vs transposed source",
        stored.reshape(-1),
        source.transpose(0, 1).reshape(-1),
    )
    rows.append(result)
    report_row(result)

    # 6. Artifact-side byte accounting, summed from the objects themselves.
    sizes = collections.defaultdict(int)
    bank_bytes = 0
    # A routed-expert bank is the one parent a job with several parameters produced.
    bank_objects = set()
    for job in report["methods"]:
        if len(job["parameters"]) > 1 and "/moe/experts/" in job["parameters"][0]:
            for name in job["parameters"]:
                for part in art.directory.bindings.get(name, {}).get("parts", ()):
                    bank_objects.add(part["object"])
    for obj in art.objects:
        if obj.kind != "tensor":
            continue
        sizes[obj.format] += obj.bytes
        if obj.id in bank_objects:
            bank_bytes += obj.bytes
    print("\n[6] artifact bytes by format")
    for fmt, count in sorted(sizes.items()):
        print(f"  {fmt:24s} {count / 2**30:8.3f} GiB")
    print(f"  {'payload total':24s} {art.payload_bytes / 2**30:8.3f} GiB")
    print(
        f"  routed expert banks ({len(bank_objects)} parents): "
        f"{bank_bytes / 2**30:.3f} GiB = {bank_bytes / 1e9:.2f} GB"
    )

    expert_rows = [r for r in rows if " expert " in r["element"]]
    shared_rows = [r for r in rows if " shared " in r["element"]]
    rel = sorted(r["relative_l2"] for r in expert_rows)
    print(f"\n[summary] {len(rows)} comparisons")
    if rel:
        print(
            f"  routed experts ({len(expert_rows)}): relative L2 min {rel[0]:.5f} "
            f"median {rel[len(rel) // 2]:.5f} max {rel[-1]:.5f}; max_abs up to "
            f"{max(r['max_abs'] for r in expert_rows):.4g}"
        )
    if shared_rows:
        print(
            "  shared experts: relative L2 "
            + ", ".join(f"{r['relative_l2']:.5f}" for r in shared_rows)
        )
    print(f"  bit-exact comparisons: {sum(1 for r in rows if r['exact'])} of {len(rows)}")
    json.dump(rows, open("/tmp/nemotron_check/results.json", "w"), indent=2)
    art.close()


if __name__ == "__main__":
    main()
