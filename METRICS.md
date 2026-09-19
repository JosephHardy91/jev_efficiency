# Metrics

Everything here is reproducible with `demo.py` (scorecard), `sweep.py` (budget sweep) and `capacity.py` (limit math).
Data are 10 synthetic emails repeated `--copies` times, so repeats are not independent samples; each setting was run once
or twice. Treat results as directional.

## 1. Per-phase accuracy (each phase against its own ground truth)

A single end-to-end number hides which phase failed, so `demo.py` prints one score per phase. **A null (abstention) is
never counted as wrong**; it lowers coverage, and accuracy is computed over answered fields.

| Phase | Ground truth | Reported as |
|---|---|---|
| 1. Search (code) | Is the gold SKU in the top 10? (only for products that have a stocked SKU) | recall@10, and rank-1 count |
| 2. Pass 1 (Jev) | `TRUTH[email][field]` for qty, qty_uom, category, material, size_unit, rating_system | correct / answered, abstained, wrong, per field |
| 3. Pass 2 (Jev) | size_in, rating, angle. Cascades from pass 1: no rating system means no rating question, which shows as an abstention. Fields that do not apply (angle on a pipe) are skipped | same |
| 4. Rerank (Jev) | Defined by its *input*: the gold SKU if the shortlist contains it, otherwise `none` | top-1 correct, with a breakdown: gold present → picked gold / picked other SKU / said none; gold absent → said none / picked a SKU |

Rerank is scored twice: fed **Jev's extracted specs** (as in the pipeline) and fed the **true specs** (oracle). The
oracle run isolates the rerank from upstream errors.

Ground-truth conventions: `size_unit` truth is IPS (the US default; no email states it, so `null` is an abstention and
DIPS/mm would be wrong); "200 ft coil" is scored as UOM `FT`; email 10 asks for a DR21 fitting that is not stocked, so its
correct rerank answer is `none`.

**End-to-end RIGHT** = fraction of products whose rerank top-1 equals (gold SKU if shortlisted, else `none`). It is the
accuracy `a` used in the harmonic mean below. The search-only baseline is "gold at rank 1" and cannot answer `none`.

## 2. Throughput model

Symbols: `B` products per call, `c` Jev calls per batch (3 today: pass 1, pass 2, rerank), `τ` input tokens per product
(2.34k measured across the three calls: about 1.26k pass 1, 0.49k pass 2, 0.59k rerank), `R` = 1,200 requests/min,
`T` = 15M tokens/min (250k tokens/s × 60). Published limits (Models page): 64k tokens per request (32k for state +
longest question), 250k tokens/s, 1,200 requests/min.

```
burst throughput   p(B)  = min(R·B/c, T/τ)          products/min   (requests bind, then tokens bind)
both bind at       B*    = c·T / (R·τ)               = 3·15M / (1200·2340) = 16.03
```

| Design | c | τ | Token ceiling T/τ | B* | p at B* |
|---|---|---|---|---|---|
| Three calls (measured) | 3 | 2.34k | 6,410/min | 16.0 | 6,400/min (107/s) |
| One call, speculative fan-out (**estimate, unbuilt**) | 1 | ~3.0k | 5,000/min | 4.2 | 5,000/min |
| One call, trimmed to fit the token limit (**target**) | 1 | 1.39k | 10,791/min | 9.0 | 10,800/min |

180 products/s (10,800/min) needs one request per 9-product batch *and* about 1.4k tokens per product.

## 3. Harmonic mean of throughput and accuracy

`HM = 2·a·p̂ / (a + p̂)` with `p̂ = p / ref`, `a` = end-to-end RIGHT at that batch size. Because `p̂` is capped at the
token ceiling, growing B past B* adds nothing to `p̂` and only lowers `a`, so the optimum sits at B*.

Normalization matters. `sweep.py` uses the best throughput *observed in the sweep*. `capacity.py` uses the design's own
token ceiling by default, or `--ref` (e.g. 6410) to compare designs on one scale. Absolute HM values are not comparable
across normalizations; the argmax over B is what to read.

## 4. Availability (three definitions we worked through)

All assume even spending through the minute and B=16, c=3 unless stated. `N = c·p/B` requests/min.

1. **Free seconds after a burst**: `60 − max(N/20, p·τ/250k)`. At p = 3,600/min: B=9 → 0 s, B=12 → 15 s, B≥16 → 26 s
   (the plateau is set by tokens per product).
2. **Seconds with at least N requests still available in the minute** (N = requests for the whole p):
   `min(60, 60·(1200 − N)/N)`. Full minute at N ≤ 600 (p ≤ 3,200/min); p = 3,600 → 46.7 s; 4,000 → 36 s; 4,800 → 20 s;
   5,600 → 8.6 s; 6,400 → 0 s.
3. **Always able to pull one more batch** (N = requests in one batch): keep `c` requests unspent through the minute
   (`p ≤ (R − c)·B/c = 399·B`), and keep one batch of tokens free. Two readings of the token side:
   * per-minute reserve: `p ≤ T/τ − B` → optimum **B=16, p ≈ 6,384/min**;
   * strict per-second reserve (`p ≤ 60·(250k/τ − B)`): optimum **B=14, p ≈ 5,570/min**, HM 0.914.

   If the token cap refreshes each second, a batch (~33k-37k tokens) is available at the start of every second without
   any reserve, and the two optima coincide at B=16. Which reading is true depends on how the limiter is implemented
   (undocumented); test it before relying on either.

## 5. Measured results

**Budget sweep** (300 products, 2 runs each, nulling at 0.6; throughput measured in short bursts; HM normalized to
best observed throughput):

| Budget (est tok/call) | Products/call | In flight | Products/s | tok/s | RIGHT | HM |
|---|---|---|---|---|---|---|
| 3,000 | 2 | 20 | 47.9 | 127k | 100% | 0.536 |
| 6,000 | 4 | 20 | 82.1 | 202k | 100% | 0.771 |
| 9,000 | 6 | 20 | 105.0 | 251k | 100% | 0.890 |
| **12,000** | **9** | 20 | **130.9** | 307k | 98.8% | **0.994** |
| 18,000 | 12 | 13 | 114.3 | 265k | 96.7% | 0.918 |
| 24,000 | 18 | 10 | 103.2 | 238k | 95.5% | 0.864 |
| 36,000 | 25 | 6 | 97.4 | 223k | 91.8% | 0.822 |
| 48,000 | 33 | 5 | 89.4 | 203k | 81.5% | 0.743 |

The 12k row ran above the published request and token rates for its ~2.3 s burst, but the SDK retries 429s by default
(2 retries, honoring `Retry-After`), so any throttling was hidden inside latency. These throughputs are therefore upper
bounds until re-measured with `--max-retries 0`. The sustained rate was **not** measured (a planned 8,000-product,
~1-minute run was not executed).

**Per-phase results, 100 products** (10 emails × 10, nulling at 0.6):

| Products/call | Search gold in top 10 | Pass 1 answered accuracy | Pass 2 answered accuracy | Rerank, extracted specs | Rerank, true specs |
|---|---|---|---|---|---|
| 4 | 70/90 | 100% (9 abstained) | 100% | 100/100 | 100/100 |
| 9 | 70/90 | 100% (3 abstained) | 100% (1 abstained) | 99/100 | 97/100 |
| 18 | 70/90 | 100% (6 abstained) | 100% | 96/100 | 94/100 |
| 36 | 70/90 | 97.7-100% by field; 2 wrong, 55 abstained | 98.8-100%; 2 wrong, 40 abstained | 88/100 | 91/100 |

At 36 products/call: 8 of the 70 products with a shortlisted gold SKU came back `none` and 1 a wrong SKU.
With nulling **off**, extraction alone drops to 91.7% correct at 36/call and rerank to 74/100, so the degradation is
real and not an artifact of nulling; nulling converts wrong values into abstentions and lifted rerank to 83-88.

**Rerank on extracted vs true specs** looked like a real difference (99 vs 97) at 100 products but disappeared at 200
(189 vs 188); it flips in both directions and all misses were one email (see CONSIDERATIONS.md).

## 6. Reproduce

```bash
python capacity.py                                   # the throughput/HM/reserve tables above
python capacity.py --calls 1 --tau 3000 --ref 6410   # the single-call estimate on the same scale
python sweep.py                                      # the budget sweep (~$0.4)
python -u demo.py --copies 10 --budget 12000         # per-phase scorecard for one setting
```

## 7. Burst dynamics and risk of a 429 (model; not validated against the live limiter)

**Sliding window.** With window `W`=60 s, trailing usage `S(t) = ∫[t−W,t] u` and slack `s(t) = C − S(t)`, so
`ds/dt = u(t−W) − u(t)`. Spendable now is what expired from the window plus any slack above the reserve; the past
schedule fixes the present allotment. Idle time is not banked beyond `W` if the limiter is a sliding log (a token bucket
would bank). Guaranteeing headroom `X` at every instant requires `ū ≤ (C − X)/W`. A side simulation (not in the repo)
of bursts that spend a fraction `f` of current slack showed the burst series bounded by `C` (no window exhaustion) and
settling for `f < 1`, and cycling with period `W` for `f = 1` (spend everything = no reserve).

**Burst means changing B.** Running above B\* at the maximum request rate (20 calls/s) demands tokens at `B/B*` times
the cap. If tokens are a hard per-second flow, nothing above B\* adds throughput, so `B_max_burst = B*`. A full-rate
burst of `τ` seconds from a baseline of `d` products/s at batch `B_s` needs
`B_b(τ) = c·q_max·τ / (R − c·d·(60−τ)/B_s)`, which rises to B\* at τ = 60 s. If tokens are instead a bankable minute
pool, a burst above the per-second rate needs `B > B*` (same formula with `q_b > q_max`) while
`(d·(60−τ) + q_b·τ)·τ_tok ≤ T`; accuracy falls past B≈20-25.

**Risk of a 429 as a function of B and time at B (implemented in `risk.py`).** Dispatch is paced, so usage per second is
deterministic; the only randomness is `ε`, the **unknown** relative error of the token accounting
(actual = planned·(1+ε)), which is carried through as a distribution and never estimated. At the maximum request rate,
planned utilization is `B/B*`, so:

```
per-second hazard      h(B) = S_ε(B*/B − 1)                     S_ε = P(ε > ·)
risk-free batch size   B_α = B* / (1 + Q_ε(1 − α))              B_safe = B*/(1+ε_max) for a hard bound ε ≤ ε_max
call admit probability a(B) = E_ε[ min(1, B* / (B·(1+ε))) ]     unit risk per call = 1 − a(B); 1 − B*/B when ε = 0
risk_429(B, t_b)       = 1 − a(B)^(20·t_b)                      ε = 0, B > B*:  1 − (B*/B)^(20·t_b)
expected 429s          = 20·t_b·(1 − a(B));   mean time to first 429 = 1 / (20·(1 − a(B)))
```

"B\* is risk-free" is a statement about `F_ε`, not about B\*: the risk at B\* is `P(ε > 0)`, zero only if the accounting is
a guaranteed upper bound. The steepness of `h` near B\* is the density of `ε` at 0. With bankable slack (baseline token
utilization `u_b`, overshoot `x = B/B* − 1`) risk is 0 until the pool drains at
`t_ex(ε) = W·(1 − (1+ε)·u_b) / ((1+ε)·(1 + x − u_b))` and follows the formula on `t_b − t_ex` afterwards. `mode` in
`risk.py` selects independent per-call error (`1 − E[a]^n`) or one systematic bias per burst (`E[1 − a_ε^n]`).

Premises: paced dispatch at the request cap; a window already full at B\* (no slack) unless `u_b` is given; the limiter
rejects excess tokens at per-call granularity; calls treated as independent.

**Retracted.** An earlier version of this section modeled random (Poisson) call arrivals, which gives a 50% per-second
risk at B\* and 19-35% at B=12-14. That does not apply to a paced dispatcher and was removed, together with two
conclusions that depended on it (expected fulfilled throughput peaking near B=12; retry-feedback "cliffs" at 58-70%
utilization).

**To measure instead of assume:** run `demo.py --max-retries 0 --quiet --no-report --no-oracle --copies N --budget b` at
increasing load and batch size; compare `SUMMARY.rate_limited` with `risk_429`, and inspect `max_retry_after_s` to tell a
sliding window from a fixed one. The `usage.input_tokens` on each response gives actual tokens per call, from which the
distribution of `ε` could be examined.

## 8. Tuning `--min-conf` (null threshold)

Confidence is the concentration of a Choice's distribution, so a threshold trades **accuracy on answered fields** against
**coverage** (share of fields answered). Two tools:

* **Offline curve** (printed at the end of every `demo.py` run): each product's spec is rebuilt at thresholds
  0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95 from the stored per-field confidences, mirroring `run_batch`'s assembly and cascade, with a
  self-check that it reproduces the run's own specs. It reports accuracy/coverage per pass and pooled, `HM(accuracy,
  coverage)`, and a utility `(right − K·wrong)/applicable fields` with `K = --wrong-cost`. It is exact for thresholds at or above
  the run's own `--min-conf` (fields nulled at run time were never asked in pass 2), so run with `--min-conf 0` for the
  full curve; rows below the run's threshold are marked `~` and never picked as best.
* **Sweep** (`sweep.py --min-confs ...`): real runs at each threshold, because nulls also change what the rerank sees, which
  the field-level curve cannot show.

Results (100 products per run, 3 runs per row, `--wrong-cost 3`):

| Budget (products/call) | min-conf | RIGHT (end to end) | Answered accuracy | Coverage | Utility |
|---|---|---|---|---|---|
| 48,000 (33) | 0 | 73.0% | 96.8% | 98.4% | 0.857 |
| 48,000 (33) | 0.5 | 81.7% | 98.6% | 90.8% | 0.858 |
| 48,000 (33) | 0.6 | 86.0% | 99.2% | 88.5% | 0.856 |
| 48,000 (33) | 0.7 | 85.3% | 99.5% | 83.6% | 0.818 |
| 48,000 (33) | 0.8 | 87.3% | 99.8% | 76.5% | 0.760 |
| 48,000 (33) | 0.9 | 85.3% | 100.0% | 61.0% | 0.610 |
| 12,000 (8) | 0 | 98.7% | 100.0% | 99.8% | 0.998 |
| 12,000 (8) | 0.6 | 98.7% | 100.0% | 99.4% | 0.994 |
| 12,000 (8) | 0.8 | 98.7% | 100.0% | 97.1% | 0.971 |

Reading it: at large batches (33 per call) nulling lifts end-to-end accuracy by about 9-14 points, mostly by turning wrong
extractions into abstentions before the rerank, and the gain plateaus from 0.6 upward while coverage keeps falling, so 0.6
is a reasonable knee there. At the default batch size (8-9 per call) extraction is already near-perfect and nulling
changes RIGHT not at all while costing a little coverage. Field-level utility alone (K=3) never prefers nulling at these
settings; only the end-to-end effect on the rerank does. Caveats: 10 emails repeated, three runs per row, so differences
among 0.6-0.9 are within noise; throughput in these short runs is noisy and was not used to choose a threshold.
