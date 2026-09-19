"""Risk of a 429 when running above B*, as a function of batch size B and time spent there t_b.

    risk_429(B, t_b) = 1 - a(B) ** (calls_per_s * t_b),      a(B) = E_eps[ min(1, B* / (B * (1 + eps))) ]

with B* = c*T/(R*tau) the batch size where steady operation exactly meets both limits. For eps = 0 and B > B*:

    risk_429 = 1 - (B*/B) ** (calls_per_s * t_b)         (0 for B <= B*)

Premises (see METRICS.md): dispatch is paced at `calls_per_s` (20 = 1,200 requests/min), the window is already full
at B* (no slack), the limiter rejects the excess tokens at per-call granularity, and calls are treated as independent.

eps is the UNKNOWN relative error of the token accounting (actual = planned * (1 + eps)) and is carried through as a
distribution, never estimated here. Pass a float (point value), a list of samples (equal weight), or a {eps: weight}
dict. `mode` says how eps enters:
    "per_call"    (default) independent error on every call: risk = 1 - E[a_eps] ** n
    "systematic"  one unknown bias for the whole burst:       risk = E[1 - a_eps ** n]
Optional bankable-pool slack: give `u_b` (baseline token utilization of the sliding window). The burst then costs nothing
until the pool drains at t_ex = W*(1 - (1+eps)*u_b) / ((1+eps)*(1 + x - u_b)), x = B/B* - 1, and risk is computed on
t_b - t_ex. `u_b=None` means no slack (hard per-second cap, or a window already full at B*).

    python risk.py                                   # grid over B and t_b using this repo's B*
    python risk.py --B 17,18,20 --t 0.1,1,5 --eps 0,0.1 --mode systematic
    python risk.py --u-b 0.7                         # with bankable slack from a 70%-utilized baseline

>>> round(b_star(), 2)
16.03
>>> risk_429(16, 5, b_star=16.03)                    # at or below B*: risk-free when eps = 0
0.0
>>> round(risk_429(18, 1, b_star=16.03), 3)
0.902
>>> round(risk_429(18, 0.1, b_star=16.03), 3)        # 2 calls
0.207
>>> risk_429(14, 60, b_star=16.03, eps=0.1)          # below B*/(1+eps_max) = 14.57: still zero
0.0
>>> round(b_safe(16.03, 0.1), 2)
14.57
>>> round(t_ex(18, b_star=16.03, u_b=0.7), 1)        # seconds above B* before the pool drains
42.6
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable, Mapping

EpsLike = float | Iterable[float] | Mapping[float, float]


def b_star(calls: float = 3, tau: float = 2340, rpm: float = 1200, tps: float = 250_000) -> float:
    """Batch size where requests and tokens bind together: c*T/(R*tau), T = tokens per minute."""
    return calls * (tps * 60) / (rpm * tau)


def b_safe(b_star_: float, eps_max: float) -> float:
    """Largest B that is risk-free when the accounting error is bounded by eps <= eps_max."""
    return b_star_ / (1 + eps_max)


def _eps_points(eps: EpsLike) -> list[tuple[float, float]]:
    if isinstance(eps, (int, float)):
        pts = [(float(eps), 1.0)]
    elif isinstance(eps, Mapping):
        total = sum(eps.values())
        pts = [(float(e), w / total) for e, w in eps.items()]
    else:
        xs = [float(e) for e in eps]
        pts = [(e, 1 / len(xs)) for e in xs]
    if any(1 + e <= 0 for e, _ in pts):
        raise ValueError("eps must be > -1 (actual tokens cannot be <= 0)")
    return pts


def call_accept_prob(B: float, b_star_: float, eps: float = 0.0) -> float:
    """P(a call is admitted) for a single eps value: the limiter admits B*/(B(1+eps)) of the demanded tokens, capped at 1."""
    return min(1.0, b_star_ / (B * (1 + eps)))


def t_ex(B: float, *, b_star: float, u_b: float, eps: float = 0.0, window: float = 60.0) -> float:
    """Seconds above B* before a bankable sliding-window token pool (baseline utilization u_b) drains. inf if it never does."""
    x = B / b_star - 1
    den = (1 + eps) * (1 + x - u_b)
    if den <= 0:
        return math.inf
    return max(0.0, window * (1 - (1 + eps) * u_b) / den)


def risk_429(B: float, t_b: float, *, b_star: float = None, calls_per_s: float = 20.0, eps: EpsLike = 0.0,
             mode: str = "per_call", u_b: float | None = None, window: float = 60.0) -> float:
    """P(at least one 429) after spending t_b seconds at batch size B. See the module docstring for the model."""
    b_star = globals()["b_star"]() if b_star is None else b_star
    pts = _eps_points(eps)
    mean_eps = sum(e * w for e, w in pts)

    def n_calls(e: float) -> float:
        lag = 0.0 if u_b is None else t_ex(B, b_star=b_star, u_b=u_b, eps=e, window=window)
        return calls_per_s * max(0.0, t_b - lag)

    if mode == "per_call":
        a = sum(w * call_accept_prob(B, b_star, e) for e, w in pts)
        return 1.0 - a ** n_calls(mean_eps)
    if mode == "systematic":
        return sum(w * (1.0 - call_accept_prob(B, b_star, e) ** n_calls(e)) for e, w in pts)
    raise ValueError("mode must be 'per_call' or 'systematic'")


def expected_429s(B: float, t_b: float, **kw) -> float:
    """Expected number of rejected calls in t_b seconds at batch size B (same keyword arguments as risk_429)."""
    b_star_ = globals()["b_star"]() if kw.get("b_star") is None else kw["b_star"]
    pts = _eps_points(kw.get("eps", 0.0))
    a = sum(w * call_accept_prob(B, b_star_, e) for e, w in pts)
    u_b, cps = kw.get("u_b"), kw.get("calls_per_s", 20.0)
    lag = 0.0 if u_b is None else t_ex(B, b_star=b_star_, u_b=u_b, eps=sum(e * w for e, w in pts), window=kw.get("window", 60.0))
    return cps * max(0.0, t_b - lag) * (1 - a)


def surface(Bs: Iterable[float], tbs: Iterable[float], **kw) -> list[list[float]]:
    """risk_429 over a grid: rows are B, columns are t_b. Convenient for sims and plots."""
    tbs = list(tbs)
    return [[risk_429(B, t, **kw) for t in tbs] for B in Bs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", default="16,17,18,20,25,33", help="batch sizes, comma separated")
    ap.add_argument("--t", default="0.1,0.5,1,2,5,10", help="seconds spent at B, comma separated")
    ap.add_argument("--eps", default="0", help="accounting error: one value, or comma list of equally weighted samples")
    ap.add_argument("--mode", choices=("per_call", "systematic"), default="per_call")
    ap.add_argument("--u-b", type=float, default=None, help="baseline token utilization of the window (enables bankable slack)")
    ap.add_argument("--calls-per-s", type=float, default=20.0)
    ap.add_argument("--bstar", type=float, default=None, help="override B* (default: derived from --calls/--tau/--rpm/--tps)")
    ap.add_argument("--calls", type=float, default=3)
    ap.add_argument("--tau", type=float, default=2340)
    ap.add_argument("--rpm", type=float, default=1200)
    ap.add_argument("--tps", type=float, default=250_000)
    a = ap.parse_args()

    bs = a.bstar or b_star(a.calls, a.tau, a.rpm, a.tps)
    eps = [float(x) for x in a.eps.split(",")]
    eps = eps[0] if len(eps) == 1 else eps
    Bs, tbs = [float(x) for x in a.B.split(",")], [float(x) for x in a.t.split(",")]
    grid = surface(Bs, tbs, b_star=bs, calls_per_s=a.calls_per_s, eps=eps, mode=a.mode, u_b=a.u_b)
    print(f"risk_429(B, t_b)  B*={bs:.2f}  calls/s={a.calls_per_s:g}  eps={eps}  mode={a.mode}  u_b={a.u_b}")
    print(f"{'B':>6} | " + " ".join(f"{t:>7g}s" for t in tbs))
    for B, row in zip(Bs, grid):
        print(f"{B:>6g} | " + " ".join(f"{r:>8.1%}" for r in row))


if __name__ == "__main__":
    main()
