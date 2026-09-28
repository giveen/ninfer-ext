"""Compile a per-tensor EXL3 bit allocation from a ``ninfer-sensitivity`` measurement.

Each tensor's end-to-end KL contribution at rate ``K`` (half bits) is modelled as

    kld(t, K) = S_t * rfn_t(K) ** alpha

where ``S_t`` is the measured sensitivity (``kld / rfn ** alpha`` at the injected noise level) and
``rfn_t(K)`` follows the EXL3 trellis error curve: amplitude halves per bit, anchored to the
quantizer's own proxy error at the rate it was run at. Because ``kld`` is convex and decreasing in
``K`` while storage is linear, the budget allocation is a separable convex problem -- greedy
assignment by marginal KL reduction per bit is exact up to one tensor of granularity.

Output: ``{name: half_bits}`` for ``ninfer-quantize --rates``.
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True, help="ninfer-sensitivity JSON")
    parser.add_argument("--report", type=Path, required=True, help="ninfer-quantize report.json")
    parser.add_argument("--bitrate", type=float, required=True, help="target body rate in half bits")
    parser.add_argument("--min-k", type=float, default=2.0)
    parser.add_argument("--max-k", type=float, default=16.0)
    parser.add_argument("--alpha", type=float, default=2.0, help="KL-vs-error exponent")
    parser.add_argument("--bit-ratio", type=float, default=0.5, help="error amplitude ratio per bit")
    parser.add_argument("--head-k", type=int, default=12, help="output-head half bits")
    parser.add_argument(
        "--group-by-layer",
        action="store_true",
        help="pool the measurement per Text layer and give its projections one rate. The measured "
        "signal is dominated by the layer axis (about 84x) rather than the projection type (1.3x), "
        "so per-tensor allocation fits mostly noise.",
    )
    parser.add_argument(
        "--rate-step",
        type=int,
        default=1,
        help="candidate base rates advance by this many half bits. Use 2 to keep every tensor on the "
        "tensor-core fast path: the contraction special-cases even half bits and falls back to the "
        "bit-by-bit reference decode for odd ones, which is about 4x slower.",
    )
    parser.add_argument(
        "--offset",
        action="append",
        default=[],
        metavar="SUBSTRING:DELTA",
        help="add DELTA half bits to every tensor whose name contains SUBSTRING, e.g. "
        "attention:2 to keep the uniform recipe's attention promotion. The offset is inside the "
        "budget, so the base rates absorb it.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    measurement = json.load(open(args.measurement))
    report = json.load(open(args.report))
    rfn_injected = float(measurement["rfn"])

    # The quantizer's proxy error is the anchored rfn at the rate it ran at.
    anchor = {}
    for tensor in report["tensors"]:
        anchor[tensor["name"]] = (float(tensor["half_bits"]), float(tensor["proxy_error"]))

    def layer_key(name):
        match = re.search(r"layers/(\d+)/", name)
        return match.group(1) if match else None

    offsets = []
    for spec in args.offset:
        substring, _, delta = spec.rpartition(":")
        if not substring or not delta:
            raise ValueError(f"bad --offset {spec!r}, expected SUBSTRING:DELTA")
        offsets.append((substring, int(delta)))

    def rate_offset(name):
        return sum(delta for substring, delta in offsets if substring in name)

    tensors = []
    for measured in measurement["targets"]:
        name = measured["name"]
        if name not in anchor:
            raise ValueError(f"{name} is not in the quantizer report")
        k0, rfn0 = anchor[name]
        if rfn0 <= 0.0:
            raise ValueError(f"{name} has a zero proxy error")
        # The measurement's sign is noise (adjacent projections of one layer flip); the sensitivity
        # is the magnitude.
        sensitivity = abs(float(measured["kld"])) / rfn_injected**args.alpha
        tensors.append(
            {
                "name": name,
                "numel": measured["numel"],
                "k0": k0,
                "rfn0": rfn0,
                "S": sensitivity,
                "layer": layer_key(name),
                "off": rate_offset(name),
            }
        )

    # An allocation unit is one Text layer (its projections share a rate) or one tensor when
    # --group-by-layer is off. A unit's KL at rate k sums its members' contributions, each member
    # keeping the quantizer anchor it was run at -- --hq promoted attention and GDN, so their k0
    # differs from the MLP's and the curve must be evaluated per member:
    #     kld(k) = sum_m S_m * (rfn0_m * bit_ratio ** (k - k0_m)) ** alpha
    units = []
    if args.group_by_layer:
        by_layer = {}
        for t in tensors:
            by_layer.setdefault(t["layer"], []).append(t)
        for layer, members in sorted(by_layer.items(), key=lambda kv: (kv[0] is None, kv[0])):
            units.append(
                {
                    "name": f"layer{layer}" if layer is not None else members[0]["name"],
                    "numel": sum(m["numel"] for m in members),
                    "members": members,
                }
            )
    else:
        units = [{"name": t["name"], "numel": t["numel"], "members": [t]} for t in tensors]

    def kld_at(unit, base):
        # Each member is quantized at its base plus its own type offset.
        return sum(
            m["S"] * (m["rfn0"] * args.bit_ratio ** (base + m["off"] - m["k0"])) ** args.alpha
            for m in unit["members"]
        )

    # Greedy: every unit starts at min_k, then bit-per-weight increments go to the largest marginal
    # KL reduction per storage bit until the budget is spent. The type offsets are part of the
    # budget, so the base rates absorb them:
    #     sum_t numel_t * rate_t = sum_j unit_numel_j * base_j + sum_t numel_t * off_t
    total = sum(t["numel"] for t in units)
    fixed = sum(t["numel"] * t["off"] for t in tensors)
    budget = args.bitrate * total - fixed
    k = [args.min_k] * len(units)
    spent = args.min_k * total
    if spent > budget:
        raise ValueError(f"target {args.bitrate} is below min_k = {args.min_k} plus the offsets")

    step = float(args.rate_step)
    heap = []
    for i, t in enumerate(units):
        gain = kld_at(t, args.min_k) - kld_at(t, args.min_k + step)
        heapq.heappush(heap, (-gain / t["numel"], i, args.min_k + step))
    while heap:
        _, i, next_k = heapq.heappop(heap)
        if next_k > args.max_k:
            continue
        t = units[i]
        if spent + t["numel"] * step > budget:
            continue
        k[i] = next_k
        spent += t["numel"] * step
        if next_k + step <= args.max_k:
            gain = kld_at(t, next_k) - kld_at(t, next_k + step)
            heapq.heappush(heap, (-gain / t["numel"], i, next_k + step))

    predicted = sum(kld_at(t, k[i]) for i, t in enumerate(units))
    rates = {}
    for i, unit in enumerate(units):
        for member in unit["members"]:
            rates[member["name"]] = int(round(k[i])) + member["off"]
    rates["text/output_head"] = args.head_k
    json.dump(rates, open(args.out, "w"), indent=2, sort_keys=True)

    mean_k = (spent + fixed) / total
    print(f" -- {len(units)} allocation units, mean {mean_k:.3f} half bits ({mean_k / 2:.3f} bpw)")
    print(f" -- predicted KL {predicted:.6g} (clean NLL {measurement['clean_nll']:.6f})")
    histogram = {}
    for value in rates.values():
        histogram[value] = histogram.get(value, 0) + 1
    for value in sorted(histogram):
        print(f"    {value:2d} half bits: {histogram[value]:3d} tensors")
    print(f" -- wrote {args.out}")


if __name__ == "__main__":
    main()
