"""Slot-filling pipeline for records whose fields are SPREAD OUT across a document, e.g.:

    "We opened a sales order for Jimbo Plumbing. The number is 43328282"

This is a different problem from entity_gram_pipeline.py. That pipeline sweeps for a single
CONTIGUOUS named-entity mention and resolves it (good for "find the one landmark in this
sentence"). Here the two fields that make up one logical record -- a customer name and an
order number -- are in different sentences, different types, linked only by discourse (the
second sentence's "The number" anaphorically refers back to the sales order just opened).
No amount of span-sweeping or interval-scheduling fixes that; it's slot-filling, same as
demo.py's own pass1/pass2 (direct per-field Choice questions over the whole document), plus
a small attachment step to link fields together when a document has more than one record.

Stage 1  candidate spans   Plain code. Capitalized-word runs, EXCLUDING each sentence's first
                           word (that capital is just grammar, not evidence of a proper noun).
                           This replaces entity_gram_pipeline.py's per-word Jev detect sweep:
                           on an unfamous name like "Jimbo Plumbing", per-word confidence
                           gating breaks (the first token alone scores under threshold,
                           truncating the span) because the model has no world-knowledge
                           prior for a made-up company the way it does for a landmark.
                           Capitalization is free and doesn't have that failure mode.
Stage 2  classify + slot-fill   Jev, 1 batched call: classify each candidate span AS A WHOLE
                           (not per word) against a type taxonomy, and in the same call,
                           directly ask for the order number from a candidate set extracted
                           by regex (same pattern demo.py uses for qty: with_none(opts(...))).
Stage 3  resolve customer   Fuzzy search (plain code, real source system) against a customers
                           index, then Jev reranks only if the shortlist has >1 real candidate.
Stage 4  attach             Plain code if there's exactly one customer and one order number
                           (nothing to disambiguate). Jev, 1 batched call, only if a document
                           has multiple customers AND multiple order numbers and needs to
                           decide which number belongs to which customer's record.

Run:
    TYPESAFE_API_KEY=... .venv/bin/python sales_order_pipeline.py
"""
import os
import re
import sys
import time

MODEL = "jev-latest"
MIN_CONF = 0.6

TEXT = "We opened a sales order for Jimbo Plumbing. The number is 43328282"

TYPES = {
    "organization": "A company, agency, team, or institution.",
    "place": "A geographic location, landmark, building, or address.",
    "product": "A product, brand, or model name.",
    "person": "An individual person's name.",
    "none": "The fragment does not name a specific entity of any kind.",
}

# the "source system" -- the only place customer identity resolution happens, never Jev.
# "jimbo's plumbing" is a deliberate near-miss distractor (shares "jimbo") to exercise rerank.
CUSTOMERS_INDEX = {
    "jimbo plumbing": {"id": "CUST-0042", "desc": "Jimbo Plumbing, a plumbing contractor, account since 2019."},
    "jimbo's plumbing": {"id": "CUST-0107", "desc": "Jimbo's Plumbing (different company, note the apostrophe), account since 2023."},
    "ace plumbing": {"id": "CUST-0003", "desc": "Ace Plumbing, a plumbing contractor, no relation to Jimbo."},
}


def fuzzy_search_customer(text: str) -> list[str]:
    words = set(text.lower().replace("'", " ").split())
    return [name for name in CUSTOMERS_INDEX if words & set(name.replace("'", " ").split())]


def capitalized_runs(text: str) -> list[tuple[int, int, str]]:
    """Candidate spans: consecutive capitalized words, excluding each sentence's first word
    (whose capital is grammatical, not evidence of a proper noun)."""
    spans, offset = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        words = re.findall(r"[A-Za-z']+|\d+", sentence)
        cap = [w[0].isupper() and not w.isdigit() for w in words]
        cap[0:1] = [False]  # sentence-initial capital doesn't count as evidence
        i = 0
        while i < len(words):
            if cap[i]:
                j = i + 1
                while j < len(words) and cap[j]:
                    j += 1
                spans.append((offset + i, offset + j, " ".join(words[i:j])))
                i = j
            else:
                i += 1
        offset += len(words)
    return spans


def main() -> None:
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("Set TYPESAFE_API_KEY first.")
    from typesafe_sdk import Choice, TypeSafeClient

    client = TypeSafeClient()
    client.system_one(state={}, model=MODEL, questions={"w": Choice(instructions="Say ok.", criteria={"ok": "ok", "none": "n/a"})})

    # ---- stage 1: candidate spans (plain code, no Jev, no per-word confidence gating)
    t0 = time.perf_counter()
    candidates = capitalized_runs(TEXT)
    numbers = re.findall(r"\d+", TEXT)  # candidate order numbers, extracted by code like demo.py's qty
    t_candidates = time.perf_counter() - t0

    # ---- stage 2: classify each whole candidate span + slot-fill the order number, 1 batched call
    t0 = time.perf_counter()
    qs = {
        f"span{k}": Choice(
            instructions=f'Within the text, classify the exact phrase "{text}" as a whole. What kind of entity is it?',
            criteria=TYPES,
        )
        for k, (i, j, text) in enumerate(candidates)
    }
    qs["order_number"] = Choice(
        instructions="What is the sales order number mentioned in the text?",
        criteria={**{n: f"a number found in the text: {n}" for n in numbers}, "none": "no order number is stated"},
    )
    resp1 = client.system_one(state={"text": TEXT}, model=MODEL, questions=qs)
    t_classify = time.perf_counter() - t0

    org_candidates = [text for k, (i, j, text) in enumerate(candidates)
                       if resp1.choices[f"span{k}"].choice == "organization" and resp1.choices[f"span{k}"].confidence >= MIN_CONF]
    order_number = resp1.choices["order_number"].choice
    order_number_conf = resp1.choices["order_number"].confidence

    # ---- stage 3: resolve each organization candidate against the real customers index
    t0 = time.perf_counter()
    resolved_customers = []
    shortlists = {name: fuzzy_search_customer(name) for name in org_candidates}
    multi = {name: cands for name, cands in shortlists.items() if len(cands) > 1}
    rerank_qs, qid_of = {}, {}
    for k, (name, cands) in enumerate(multi.items()):
        qid_of[name] = f"c{k}"
        rerank_qs[f"c{k}"] = Choice(
            instructions=f'Within the text, the phrase "{name}" was flagged as an organization and fuzzy-matched these '
                         "candidates from the real customers index. Which one does it actually refer to here, if any?",
            criteria={**{c: CUSTOMERS_INDEX[c]["desc"] for c in cands}, "none": "None of these candidates is correct."},
        )
    resp2 = client.system_one(state={"text": TEXT}, model=MODEL, questions=rerank_qs) if rerank_qs else None
    for name in org_candidates:
        cands = shortlists[name]
        if len(cands) == 1:
            resolved_customers.append((name, cands[0], 1.0))
        elif len(cands) > 1:
            c = resp2.choices[qid_of[name]]
            if c.choice != "none":
                resolved_customers.append((name, c.choice, c.confidence))
    t_resolve = time.perf_counter() - t0

    # ---- stage 4: attach customer <-> order number. Trivial (zip) if exactly one of each;
    # otherwise this is where a real attachment Choice (candidates = the resolved customers)
    # would be needed to decide which number belongs to which customer's record.
    t0 = time.perf_counter()
    records = []
    if len(resolved_customers) == 1 and order_number != "none":
        name, identity, conf = resolved_customers[0]
        records.append({"customer_raw": name, "customer_id": CUSTOMERS_INDEX[identity]["id"], "order_number": order_number})
    else:
        for (name, identity, conf) in resolved_customers:
            records.append({"customer_raw": name, "customer_id": CUSTOMERS_INDEX[identity]["id"], "order_number": None})
        if len(resolved_customers) != 1:
            print(f"NOTE: {len(resolved_customers)} customers detected -- attachment to order numbers needs an explicit "
                  "Jev linking step (not implemented; this demo text only has one of each).")
    t_attach = time.perf_counter() - t0

    print(f"candidate spans: {[text for (i, j, text) in candidates]}")
    print(f"organization candidates: {org_candidates}")
    print(f"order number candidates (regex): {numbers}  ->  chosen: {order_number!r} (conf={order_number_conf:.2f})\n")
    print("resolved sales order record(s):")
    for r in records:
        print(f"  {r}")

    print(f"\nstage 1 candidate spans : {t_candidates * 1000:7.2f} ms   (pure code)")
    print(f"stage 2 classify+slotfill: {t_classify * 1000:7.1f} ms   ({resp1.usage.input_tokens} tok)")
    print(f"stage 3 resolve customer : {t_resolve * 1000:7.1f} ms   ({resp2.usage.input_tokens if resp2 else 0} tok)")
    print(f"stage 4 attach           : {t_attach * 1000:7.2f} ms   (pure code)")
    print(f"e2e                      : {(t_candidates + t_classify + t_resolve + t_attach) * 1000:7.1f} ms")


if __name__ == "__main__":
    main()
