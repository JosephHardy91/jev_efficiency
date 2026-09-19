# Considerations, findings and open questions

## Design findings

* **Closed option lists must cover the real universe.** A DR list without `25` made Jev return DR26 for "DR 25". The
  rerank still recovered because it also reads the email, which hid the error; only per-phase scoring exposed it.
* **Only qty is a free field.** It is a Choice over every number found by regex. Distractors (PO numbers, phones, dates)
  were never picked. Everything else is a Choice over a real option set with a `none` escape hatch.
* **Two extraction passes.** Pass 2's options depend on pass 1 (size list by unit, rating list by system, follow-ups by
  category), so it cannot be merged without speculating every branch. That merge is what a one-call design would do.
* **Search bounds everything.** 20 of 90 products with a stocked SKU never reach the shortlist (`tees` vs `tee`, `10"`
  tokenization), capping the rerank. The rerank correctly answers `none` there; fix recall in search, not in Jev.
* **`none` is a legitimate answer.** Scoring "gold absent ⇒ none is right" is what makes the rerank number meaningful. A
  search-only baseline can never score on those products.
* **Rerank error to look at:** for "4 tees, 12-inch" with no stocked SKU, the rerank picks the 4-inch tee (likely
  confusing the quantity with the size), even when given the true specs. Every rerank miss in the 200-product run was
  this email.

## Batching and accuracy

* Products share one state when batched, so each question's own email sits among distractors. The Jev 1.13 jaggedness
  page says accuracy falls as state grows with unrelated content ("context rot"), and the sweep matches: end-to-end
  accuracy 100% at 4 per call, ~96% at 12-18, ~82-88% at 33-36.
* Extraction degrades mostly by **abstaining** (confidence falls below threshold), not by answering wrong: at 36/call,
  ~95 abstentions vs 4 wrong answers across the two passes.
* Many questions over **one** state are free (fan-out pattern); many **states** in one call are not.
* Use the smallest budget that meets your throughput target. The 64k limit is a ceiling, not a target.
* The token estimator started 40% low (chars/token 3.0 vs about 1.8 measured) and would have overfilled calls. It now
  starts at 1.8 and recalibrates from real usage. Pass 1 and rerank estimate within ~20%; pass 2 undershoots 20-40%. Keep
  a margin under 64k. A failed call splits its batch in half and retries.

## Nulls (abstentions)

* Any Choice answer with confidence below `--min-conf` (default 0.6) becomes `null`, and so does a `none` answer.
* Confidence measures how concentrated the distribution is, not whether the answer is right. Several acceptable options
  spread probability; do not tune a threshold on one question type and reuse it on another.
* A null size unit still uses the IPS size list in pass 2 as a working default; the unit stays null in the result.
* Judge a phase by accuracy on answered fields **and** coverage. Counting nulls as wrong penalizes correct abstention.

## Limits and rate semantics

* Published (Models page): 64k tokens/request (32k for state + longest question), 250k tokens/s, 1,200 requests/min,
  Choice ≤ 255 options, $0.042/M input tokens (output free). The page says limits adjust dynamically; higher limits exist on
  custom/enterprise plans. The SDK retries `429`/`529` with backoff.
* **429s were hidden in every measurement in METRICS.md §5.** The SDK's default `RetryPolicy` retries 429 twice with
  backoff (0.5-5 s) and honors `Retry-After`, so a throttled call would only have shown up as extra latency. A ~2.3 s
  burst ran at ~44 requests/s and ~307k tokens/s (above both published rates), but we cannot say it was *unthrottled*.
  Re-measure with `--max-retries 0`, which surfaces and counts each 429 (`SUMMARY.rate_limited`).
* **Risk numbers from an earlier Poisson-arrival model were retracted** (50% per-second risk at B\*, retry-feedback cliffs at
  58-70% utilization, fulfilled-throughput peak near B=12). A paced dispatcher has no arrival variance. The current model is
  METRICS.md section 7 / `risk.py`: risk above B\* is `1 − (B*/B)^(20·t_b)` for exact accounting, and B\* is risk-free
  only if the token accounting `ε` is a guaranteed upper bound. `ε` is unknown and carried through as a distribution.
* A 429 is nonfulfillment, not a size problem. `run_batch` waits `Retry-After` and resends the **same** batch; splitting
  it (the response to other errors such as context-too-large) would send more requests and worsen the 429.
* **Open question:** does the limiter refresh per second (fixed window / token bucket) or track a rolling window? It
  decides whether "always able to pull one more batch" costs anything (B=16 vs B=14 in METRICS.md §4). A burst test with `--max-retries 0`
  would settle it: read `Retry-After` (time until the oldest usage expires means a sliding window; time to the next
  second boundary means a fixed window) and compare 429 rate against utilization.
* The one-call speculative-fan-out design (all branches asked up front, rerank from email + shortlist) is an
  **unbuilt estimate**. It trades requests for tokens and needs about 1.4k tokens/product to reach 180 products/s.
  Rerank-from-email-only accuracy is untested.

## Data and statistics caveats

* The 10 emails and the catalog are synthetic. Repeats (`--copies`) are the same emails, so misses cluster and sample
  size is effectively 10, not 100 or 300. Sweep settings were run once or twice; differences of 1-2 points are noise, the
  81.5% at 33/call is not.
* Accuracy at B=16 and B=14 used in the capacity math is **interpolated** between measured B=12 and B=18.
* An earlier summary said "97 abstentions" at 36/call; the correct count from the scorecard is 95 (55 in pass 1, 40 in
  pass 2).
* Earlier wording labelled failures "WRONG" when the gold SKU was not shortlisted; the scoring now treats `none` as
  correct there, matching the ground-truth definition in METRICS.md.
* Ground-truth judgment calls: size unit IPS by default, "200 ft coil" as FT.

## Operational

* **Credentials:** keep `TYPESAFE_API_KEY` in the environment; it is never written to a file here, and `.env` is
  git-ignored. A key that has been pasted into a chat or terminal transcript should be rotated.
* Emails may contain customer PII in real use. TypeSafe states requests are not used for training and offers zero data
  retention on enterprise; see https://docs.typesafe.ai/legal before sending real data.
* `report.html` and `results.json` embed the full email text and are git-ignored for that reason and for size.
* `jev-latest` is an alias that moves; pin a versioned model once thresholds are tuned.
