"""Capacity planner: batch size vs throughput, accuracy and availability under the published limits.

    python capacity.py                                  # defaults = this repo's measured 3-call pipeline
    python capacity.py --calls 1 --tau 3000             # single-call speculative fan-out design (estimate)
    python capacity.py --acc "4:1,9:.988,16:.959"       # your own accuracy-vs-batch-size points

Model (see METRICS.md):
    burst throughput      p(B)     = min(R*B/c, T/tau)                  products/min
    both limits bind at   B*       = c*T / (R*tau)
    one-batch reserve     p_res(B) = min((R-c)*B/c, T/tau - B, 60*(TPS/tau - B))   (per-second token refresh optional)
    harmonic mean         HM       = 2*a*p_hat / (a + p_hat),  p_hat = p / ref  (ref defaults to T/tau; --ref for a common scale)
"""

import argparse

DEFAULT_ACC = "2:1,4:1,6:1,9:.988,12:.967,18:.955,25:.918,33:.815"  # end-to-end RIGHT from the 300-product sweep


def interp(points: dict[float, float], b: float) -> float:
    ks = sorted(points)
    if b <= ks[0]:
        return points[ks[0]]
    for lo, hi in zip(ks, ks[1:]):
        if lo <= b <= hi:
            return points[lo] + (points[hi] - points[lo]) * (b - lo) / (hi - lo)
    return points[ks[-1]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calls", type=float, default=3, help="Jev requests per batch (c)")
    ap.add_argument("--tau", type=float, default=2340, help="input tokens per product across all calls")
    ap.add_argument("--rpm", type=float, default=1200, help="requests per minute limit (R)")
    ap.add_argument("--tps", type=float, default=250_000, help="input tokens per second limit")
    ap.add_argument("--acc", default=DEFAULT_ACC, help="B:accuracy points, comma separated (linear interpolation)")
    ap.add_argument("--ref", type=float, default=0, help="products/min used to normalize throughput in HM (default: this design's own token ceiling; pass 6410 to compare designs on a common scale)")
    ap.add_argument("--sizes", default="6,9,12,14,16,18,25,33", help="batch sizes to tabulate")
    a = ap.parse_args()

    R, c, tau, TPS = a.rpm, a.calls, a.tau, a.tps
    T = TPS * 60
    acc = {float(k): float(v) for k, v in (kv.split(":") for kv in a.acc.split(","))}
    ceiling = T / tau  # token ceiling: products/min if requests were unlimited
    ref = a.ref or ceiling
    print(f"c={c:g} calls/batch, tau={tau:,.0f} tok/product, R={R:,.0f} req/min, T={T / 1e6:.1f}M tok/min")
    print(f"token ceiling {ceiling:,.0f}/min (HM normalized to {ref:,.0f}) | B* (both limits bind) = c*T/(R*tau) = {c * T / (R * tau):.2f}\n")
    print(f"{'B':>5} {'burst p/min':>12} {'acc':>6} {'HM':>6} | {'reserve p/min':>13} {'HM':>6}   (reserve = always able to pull one more batch)")
    best = {"burst": (0, 0), "reserve": (0, 0)}
    for b in (float(x) for x in a.sizes.split(",")):
        ac = interp(acc, b)
        p_burst = min(R * b / c, ceiling)
        p_res = max(0.0, min((R - c) * b / c, ceiling - b, 60 * (TPS / tau - b)))
        hm = lambda p: 2 * ac * (p / ref) / (ac + p / ref)
        print(f"{b:>5g} {p_burst:>12,.0f} {ac:>6.1%} {hm(p_burst):>6.3f} | {p_res:>13,.0f} {hm(p_res):>6.3f}")
        best["burst"] = max(best["burst"], (hm(p_burst), b))
        best["reserve"] = max(best["reserve"], (hm(p_res), b))
    print(f"\nbest HM: burst at B={best['burst'][1]:g} ({best['burst'][0]:.3f}), with per-second reserve at B={best['reserve'][1]:g} ({best['reserve'][0]:.3f})")


if __name__ == "__main__":
    main()
