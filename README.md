# Jev product-line pipeline demo

Turns free-text customer emails (ISCO-style pipe and fittings requests) into catalog matches with TypeSafe's **Jev**
System One model. Code owns the workflow; Jev only makes typed judgments (Choice questions) whose **options are built
per call**.

```
email ──► search (code, no Jev) ──► shortlist of 10 SKUs
  │
  ├─► pass 1 (Jev): qty, qty UOM, category, material, size unit, rating system   (independent, one request)
  ├─► pass 2 (Jev): size, rating value, category follow-ups (elbow angle, tee config)
  │                 options are framed by pass 1's answers
  └─► rerank (Jev): one Choice whose options are that email's 10 candidate SKUs + "none"
```

Low-confidence answers become `null` (abstentions). Products are packed into shared calls (batches) sized by an
estimated-token budget, and batches run concurrently. Every phase is scored against ground truth.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # typesafe-sdk from https://pypi.typesafe.ai/ (tested with 0.7.0)
export TYPESAFE_API_KEY=...                         # create one at https://console.typesafe.ai/ ; never commit it
```

## Run

```bash
.venv/bin/python -u demo.py                    # 10 sample emails, live stage tracker, per-phase scorecard, report.html
.venv/bin/python -u demo.py --copies 30        # 300 products (10 emails x 30) to exercise batching
.venv/bin/python -u demo.py --offline          # search only, no API key, no cost
```

| Flag | Default | Meaning |
|---|---|---|
| `--copies N` | 1 | Repeat the 10 sample emails N times (load testing; repeats are not independent samples) |
| `--budget N` | 12000 | Estimated input tokens per call. Sets products per call. Published limit is 64k (32k for state + longest question) |
| `--min-conf X` | 0.6 | Any Choice answer below this confidence becomes `null` (abstention). `0` disables nulling. The scorecard prints a precision-vs-coverage curve so you can tune it; exact for thresholds at or above the run's own, so use `--min-conf 0` for the full curve |
| `--wrong-cost K` | 1 | Utility used by the curve and `sweep.py`: +1 per correct field, −K per wrong field, 0 per abstention |
| `--workers N` | `min(20, 250000 // budget)` | Calls in flight. Derived from the published 250k tok/s and 1,200 req/min limits |
| `--quiet` | off | Suppress per-stage lines; print a 5-second throughput ticker instead |
| `--no-report` | off | Skip `report.html` / `results.json` (use for large runs) |
| `--max-retries N` | 2 | SDK retries for 429/5xx (SDK default 2 silently retries 429s with backoff). Use `0` to make every 429 visible: it is counted and the batch is resent after `Retry-After` |
| `--gen N` / `--gen-seed S` | off / 0 | Generate N distinct emails with known ground truth (about 20% ask for an unstocked item) instead of the 10 fixed ones |
| `--per-call N` | off | Assign exactly N products per call (overrides `--budget` packing) |
| `--shuffle SEED` | off | Shuffle product order before packing: randomizes which products share a call and their position |
| `--no-oracle` | off | Skip the extra rerank run fed the *true* specs (saves one call per batch) |
| `--offline` | off | No Jev calls: print search rank of the gold SKU per email |

## Outputs

* **Terminal**: live per-batch stage lines (`▶` start, `✓` done with seconds and estimated vs actual tokens), then a
  per-phase scorecard (ending with the `--min-conf` curve), then a `SUMMARY {json}` line (used by `sweep.py`).
* **`report.html`**: self-contained x-ray. A stage tracker at the top replays the *recorded* timings (pizza-tracker
  style, speed selectable), followed by one collapsible row per email with: the email, every Choice's option set and
  probability distribution for both passes (NULL badges where confidence fell below the threshold), the values code
  normalized, and the 10-candidate shortlist with rerank probabilities. Gold is marked green.
* **`results.json`**: the same data raw. Both files are git-ignored and overwritten every run.

## Files

| File | Purpose |
|---|---|
| `demo.py` | The pipeline: catalog, option sets, search, two-pass extraction, rerank, batching, scorecard, report |
| `capacity.py` | Capacity planner: batch size vs throughput / accuracy / availability under the published limits |
| `sweep.py` | Runs `demo.py` across token budgets and `--min-conf` values (`--min-confs`) and ranks them: end-to-end accuracy, answered accuracy vs coverage, throughput, utility |
| `collect.py` | Runs `demo.py` over a grid and keeps per-product records in `runs/` (input for `fit.py`). Designed mode: `--gen 300 --sizes 4,8,...,32 --shuffle` assigns batch size, randomizes product order and run order |
| `fit.py` | "Spindle" regression (additive piecewise-linear model, a separate hinge per variable) of end-to-end accuracy on products per call, position, min-conf and confounders. Needs `pip install -r requirements-analysis.txt` |
| `risk.py` | `risk_429(B, t_b)`: risk of a 429 when running above B\*, with the unknown token-accounting error carried through. Importable for simulations |
| `EXTENDING.md` | How to adapt the pipeline to a new workflow |
| `METRICS.md` | Every metric, formula and measured result |
| `CONSIDERATIONS.md` | Findings, pitfalls, caveats and open questions |

## Risk of a 429 above B\* (`risk.py`)

```python
from risk import risk_429, surface, b_star

risk_429(18, 1)                                            # batch size 18 held for 1 s, eps = 0  ->  0.902
risk_429(18, 1, eps=[0, 0.05, 0.1])                        # eps (token-accounting error) carried as samples
risk_429(18, 1, eps={0: .5, .1: .5}, mode="systematic")    # one unknown bias for the whole burst
risk_429(20, 30, u_b=0.7)                                  # with bankable slack from a 70%-utilized baseline
surface([17, 18, 20], [0.1, 1, 5])                         # grid: rows are B, columns are t_b
```

`python risk.py --B 17,18,20 --t 0.1,1,5 --eps 0,0.1 --mode systematic` prints the grid. The model and its premises are in
METRICS.md section 7; `python -m doctest risk.py` checks the closed forms.

## Cost

Jev input tokens are $0.042 per million; output tokens are free. This pipeline uses about 2.3k input tokens per product
(about $0.0001), so 100 products is about a cent.
