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

## 7. Burst dynamics and risk of 429 (model; not validated)

**Sliding window.** With window `W`=60 s, trailing usage `S(t) = ∫[t−W,t] u` and slack `s(t) = C − S(t)`, so
`ds/dt = u(t−W) − u(t)`. Idle time is not banked beyond `W` if the limiter is a sliding log (a token bucket would bank).
A burst at rate `q_b` for `τ_b` seconds uses `X = q_b·τ_b`. Its start stays in the window for 60 s, so for the remaining
`60 − τ_b` seconds only `s = C − S_prior − X` is left and the sustainable rate is `s / (60 − τ_b)`; capacity returns at
`q_b` over the `τ_b` seconds after that mark. Guaranteeing headroom `X` at every instant requires `ū ≤ (C − X)/W`.

**Burst means changing B.** Raising B sends more products per request (tokens/s cap the gain at B\*). Throughput
`p(B) = min(R·B/c, T/τ)` is piecewise linear with a kink at B\*, but the **probability of being fulfilled is not linear
in B**: calls are lumpy (each carries `B·τ` tokens at once) and larger B means a larger quantum and higher variance.
Toy model, per-second hard cap `cap`=250k tokens/s, Poisson call arrivals, `τ`=2.34k: usage per second has mean `q·τ`
and std `τ·√(q·B)`, so `P(429 in a second) ≈ 1 − Φ((cap − q·τ)/(τ·√(q·B)))`.

| Utilization | B=4 | B=9 | B=16 | B=25 |
|---|---|---|---|---|
| 60% | 0.4% | 3.8% | 9.1% | 14.3% |
| 80% | 12.4% | 22.1% | 28.2% | 32.2% |
| 90% | 29.3% | 35.8% | 39.3% | 41.4% |

Expected fulfilled throughput `p(B)·(1 − risk(B))` peaks below B\*: B=9 → 58.7/s, **B=12 → 64.5/s**, B=14 → 60.3/s,
B=16 → 53.5/s (vs 106.7/s nominal at B=16). A risk-aware utility is `HM(a(B), p̂·(1 − risk))`, and its optimum sits below
B\* by a margin set by risk tolerance. Paced (evenly spaced) dispatch would shrink the variance term. The assumptions
(Poisson arrivals, hard per-second cap) are unverified, so read the table for shape only.

**To measure instead of assume:** run `demo.py --max-retries 0 --quiet --no-report --no-oracle --copies N --budget b`
at increasing load; plot `SUMMARY.rate_limited` against utilization and batch size, and inspect `max_retry_after_s`.
