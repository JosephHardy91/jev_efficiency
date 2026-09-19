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
| `--min-conf X` | 0.6 | Any Choice answer below this confidence becomes `null` (abstention). `0` disables nulling |
| `--workers N` | `min(20, 250000 // budget)` | Calls in flight. Derived from the published 250k tok/s and 1,200 req/min limits |
| `--quiet` | off | Suppress per-stage lines; print a 5-second throughput ticker instead |
| `--no-report` | off | Skip `report.html` / `results.json` (use for large runs) |
| `--max-retries N` | 2 | SDK retries for 429/5xx (SDK default 2 silently retries 429s with backoff). Use `0` to make every 429 visible: it is counted and the batch is resent after `Retry-After` |
| `--no-oracle` | off | Skip the extra rerank run fed the *true* specs (saves one call per batch) |
| `--offline` | off | No Jev calls: print search rank of the gold SKU per email |

## Outputs

* **Terminal**: live per-batch stage lines (`▶` start, `✓` done with seconds and estimated vs actual tokens), then a
  per-phase scorecard, then a `SUMMARY {json}` line (used by `sweep.py`).
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
| `sweep.py` | Runs `demo.py` across token budgets and ranks them by harmonic mean of accuracy and throughput |
| `EXTENDING.md` | How to adapt the pipeline to a new workflow |
| `METRICS.md` | Every metric, formula and measured result |
| `CONSIDERATIONS.md` | Findings, pitfalls, caveats and open questions |

## Cost

Jev input tokens are $0.042 per million; output tokens are free. This pipeline uses about 2.3k input tokens per product
(about $0.0001), so 100 products is about a cent.
