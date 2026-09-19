# Adapting the pipeline to a new workflow

`demo.py` is one file; each piece below is a named block you can swap. The design rules come from the live TypeSafe
docs (https://docs.typesafe.ai/llms.txt): read the Choice/Noul/Score primitive pages, *How to build with System One*,
and the jaggedness page before changing question wording.

## 1. Ground rules (they explain why the code looks the way it does)

* **Jev picks, code copies.** Jev is not trained to generate text. Turn extraction into a `Choice` over a bounded set
  of options; find free-form values (numbers, part codes) in code with a regex and offer *those spans* as the options.
* **Options are per call.** A Choice's `criteria` is built for this request: the spans in this email, the size list for
  this unit, the 10 candidates for this product. There is no fixed schema.
* **Closed sets must be complete.** If the right value is not an option, Jev is forced onto the nearest wrong one
  (a DR list without `25` turned "DR 25" into DR26). Build lists from your catalog/customer data, not from memory.
* **Always offer an escape hatch** (`none`, `other`) and treat it, and low confidence, as abstention.
* **One narrow question each.** Independent questions over the same state go in one request; they run in parallel.
* **Question ids are not sent to the model.** Put the full meaning in `instructions` and `criteria`.
* **Refer to state by name.** With several products in one state, each question starts with
  ``About `products[k].text` only:`` (see `pass1_qs`). Filter state to what the question needs: unrelated content
  costs accuracy ("context rot").
* Choice allows at most 255 options. Beyond that, narrow in two stages.

## 2. Change what the emails / catalog are

| To change | Edit | Notes |
|---|---|---|
| Sample inputs | `EMAILS` (text, gold-SKU tuple or `None`) | `None` means the correct answer is "no stocked SKU" |
| Ground truth per field | `TRUTH` via `_t(...)` | One entry per email, same order. `angle` is `None` when not applicable |
| Catalog | `add(material, category, size, rating, angle=None)` loops near the top | Each SKU gets a `desc` string that Jev sees as the option description |
| Search | `DOCS` / `fast_search` | Swap in your real search (BM25, embeddings, SQL). It must return up to `TOP_K` dicts with `sku` and `desc`. It runs *outside* Jev and only sees the email text |

## 3. Change or add fields

Option sets are plain dicts/lists near `QTY_UOMS`: `CATEGORIES`, `MATERIALS`, `SIZE_UNITS`, `RATING_SYSTEMS`,
`SIZE_OPTIONS[unit]`, `RATING_OPTIONS[system]`. Edit them freely; keys are the values Jev returns.

**Add an independent field (pass 1):**
1. Add a question in `pass1_qs` (`f"{k}:my_field"` id, a `q(instructions, criteria)`).
2. Add it to the `it["spec"]` dict built in `run_batch`.
3. Add its ground truth to `_t(...)`/`TRUTH` and its name to `PASS1_FIELDS` so it is scored.

**Add a field whose options depend on an earlier answer (pass 2):** do the same in `pass2_qs`, choosing its options from
`spec[...]`; read the answer in the pass-2 block of `run_batch`; add to `PASS2_FIELDS`. `FOLLOWUPS` is the shortcut for
"if category == X, also ask Y": add `"category": ("field", "question", {option: description})`.

**Add a stage:** write `<stage>_qs(k, ...)`, call it with `call(R, bid, "<stage>", state, qs)` inside `run_batch`, and add
the stage name to `Tracker.STAGES` / `Tracker.LABELS` so the tracker and report show it. Prefer merging independent
questions into an existing call: a call is the scarce resource (see METRICS.md).

## 4. Tune behaviour without code

* `--min-conf X`: raise to abstain more, lower to answer more. Tune it on your data: confidence measures how
  concentrated the distribution is, not overall correctness.
* `--budget N`: products per call. Re-run `sweep.py` when questions, option-list sizes or the model change.
* `MODEL = "jev-latest"` moves when a new model ships; pin a versioned id (e.g. `jev-1.13.0`) once you have tuned
  thresholds against it.

## 5. Re-estimating token cost after your changes

Option lists dominate cost. `per_product_chars` sizes a product in the *worst* case across the three stages so one
packing works for the whole pipeline; if you add stages or big option lists, update it (the `rr = 1100` term is the
rerank size). The estimator recalibrates chars/token from real usage after each call, but it is only a guide: compare
"est vs actual" in the tracker output and keep a safety margin under the 64k limit. Then re-run
`python capacity.py --tau <your tokens per product> --calls <your calls per batch>` to see the new throughput ceiling.

## 6. Checklist for a new workflow

1. Write 10-20 real inputs and label the ground truth for every field and for the correct catalog answer (or none).
2. Build closed option lists from real data; add `none`/`other`.
3. Run `--copies 1` and read `report.html`: for each wrong answer, inspect the exact state, options and distribution
   before touching the prompt (missing evidence, model error, code error and service failure look different).
4. Read the per-phase scorecard; fix the earliest failing phase first.
5. Sweep `--budget`, pick the knee, set `--min-conf`, and only then scale up.
