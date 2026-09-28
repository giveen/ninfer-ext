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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    measurement = json.load(open(args.measurement))
    report = json.load(open(args.report))
    rfn_injected = float(measurement["rfn"])

    # The quantizer's proxy error is the anchored rfn at the rate it ran at.
    anchor = {}
    for tensor in report["tensors"]:
        anchor[tensor["name"]] = (float(tensor["half_bits"]), float(tensor["proxy_error"]))

    tensors = []
    for measured in measurement["targets"]:
        name = measured["name"]
        if name not in anchor:
            raise ValueError(f"{name} is not in the quantizer report")
        k0, rfn0 = anchor[name]
        if rfn0 <= 0.0:
            raise ValueError(f"{name} has a zero proxy error")
        sensitivity = float(measured["kld"]) / rfn_injected**args.alpha
        tensors.append(
            {"name": name, "numel": measured["numel"], "k0": k0, "rfn0": rfn0, "S": sensitivity}
        )

    def rfn_at(t, k):
        return t["rfn0"] * args.bit_ratio ** (t["k0"] - k)

    def kld_at(t, k):
        return t["S"] * rfn_at(t, k) ** args.alpha

    # Greedy: every tensor starts at min_k, then one bit-per-weight increments go to the largest
    # marginal KL reduction per storage bit until the budget is spent.
    total = sum(t["numel"] for t in tensors)
    budget = args.bitrate * total
    k = [args.min_k] * len(tensors)
    spent = args.min_k * total
    if spent > budget:
        raise ValueError(f"target {args.bitrate} is below min_k = {args.min_k}")

    step = 1.0
    heap = []
    for i, t in enumerate(tensors):
        gain = kld_at(t, args.min_k) - kld_at(t, args.min_k + step)
        heapq.heappush(heap, (-gain / t["numel"], i, args.min_k + step))
    while heap:
        _, i, next_k = heapq.heappop(heap)
        if next_k > args.max_k:
            continue
        t = tensors[i]
        if spent + t["numel"] * step > budget:
            continue
        k[i] = next_k
        spent += t["numel"] * step
        if next_k + step <= args.max_k:
            gain = kld_at(t, next_k) - kld_at(t, next_k + step)
            heapq.heappush(heap, (-gain / t["numel"], i, next_k + step))

    predicted = sum(kld_at(t, k[i]) for i, t in enumerate(tensors))
    rates = {t["name"]: int(round(k[i])) for i, t in enumerate(tensors)}
    rates["text/output_head"] = args.head_k
    json.dump(rates, open(args.out, "w"), indent=2, sort_keys=True)

    mean_k = spent / total
    print(f" -- {len(tensors)} tensors, mean {mean_k:.3f} half bits ({mean_k / 2:.3f} bpw)")
    print(f" -- predicted KL {predicted:.6g} (clean NLL {measurement['clean_nll']:.6f})")
    histogram = {}
    for value in rates.values():
        histogram[value] = histogram.get(value, 0) + 1
    for value in sorted(histogram):
        print(f"    {value:2d} half bits: {histogram[value]:3d} tensors")
    print(f" -- wrote {args.out}")


if __name__ == "__main__":
    main()
