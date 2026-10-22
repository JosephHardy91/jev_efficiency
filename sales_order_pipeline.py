"""Slot-filling pipeline for records whose fields are SPREAD OUT across a document, e.g.:

    "We opened a sales order for Jimbo Plumbing. The number is 43328282"

This is a different problem from entity_gram_pipeline.py. That pipeline sweeps for a single
CONTIGUOUS named-entity mention and resolves it (good for "find the one landmark in this
sentence"). Here the two fields that make up one logical record -- a customer name and an
order number -- are in different sentences, different types, linked only by discourse (the
second sentence's "The number" anaphorically refers back to the sales order just opened).
No amount of span-sweeping or interval-scheduling fixes that; it's slot-filling, same as
demo.py's own pass1/pass2 (direct per-field Choice questions over the whole document), plus
a real attachment step to link fields together across however many records a document has.

Stage 1  candidate spans   Plain code. Capitalized-word runs, EXCLUDING each sentence's first
                           word (that capital is just grammar, not evidence of a proper noun).
                           This replaces entity_gram_pipeline.py's per-word Jev detect sweep:
                           on an unfamous name like "Jimbo Plumbing", per-word confidence
                           gating breaks (the first token alone scores under threshold,
                           truncating the span) because the model has no world-knowledge
                           prior for a made-up company the way it does for a landmark.
                           Capitalization is free and doesn't have that failure mode.
                           Order-number candidates come from digit-run regex UNION a
                           structural spaCy check: any copular clause ("X is Y") whose
                           complement looks code-shaped (has a digit, or a hyphen joining
                           non-dictionary segments). No trigger WORD list ("number"/"code"/
                           "id"/"order") -- that was the same brittleness as digits-only,
                           just moved up one level: a fixed keyword enumeration still misses
                           any synonym never anticipated ("reference", "ticket", "tracking").
                           The structural check never reads what the subject noun IS, only
                           that a copula links some abstract noun to a code-shaped value, so
                           it generalizes to any subject noun for free. Digits-only would
                           miss a non-numeric identifier entirely, and even a digit-containing
                           but hyphenated one ("XJ-4471") would be truncated to just "4471" --
                           Jev never free-text-generates a value (README.md: "Jev only makes
                           typed judgments ... whose options are built per call"), so if code
                           never proposes the right candidate, there's nothing for any later
                           stage to pick.
Stage 2  classify          Jev, 1 batched call: classify each candidate span AS A WHOLE (not
                           per word) against a type taxonomy.
Stage 3  resolve customer  Fuzzy search (plain code, real source system) against a customers
                           index, then Jev reranks only if the shortlist has >1 real candidate.
Stage 4  attach            Jev, 1 batched call: for EVERY candidate order number, which (if
                           any) resolved customer does it belong to. Always a real Jev call,
                           never a count-based shortcut -- "there's only one customer and one
                           number" is not evidence they're linked, it's just an easy case.
                           Verified against an adversarial document where the customers and
                           numbers are introduced in mismatched order, so pairing by order of
                           mention would silently get both assignments backwards.

Run:
    pip install -r requirements-nlp.txt && python -m spacy download en_core_web_sm
    TYPESAFE_API_KEY=... .venv/bin/python sales_order_pipeline.py
"""
import os
import re
import sys
import time
from collections import Counter

MODEL = "jev-latest"
MIN_CONF = 0.6

DOCUMENTS = [
    "We opened a sales order for Jimbo Plumbing. The number is 43328282",
    # adversarial: customers are introduced Jimbo-then-Ace, but their numbers are stated
    # Ace-then-Jimbo -- pairing by order of mention would get both assignments backwards.
    "We opened two sales orders: one for Jimbo Plumbing and one for Ace Plumbing. "
    "The order for Ace Plumbing is 91234567, and Jimbo's order is 43328282.",
    # non-numeric identifier -- a digits-only regex would propose zero candidates here
    "We opened a sales order for Jimbo Plumbing. The order code is french-fry-bagel-butt.",
    # a trigger noun ("reference") never in the old hardcoded {number,code,id,order} list,
    # AND a hyphenated alphanumeric code a digits-only regex would truncate to just "4471"
    "We opened a sales order for Jimbo Plumbing. The reference is XJ-4471.",
]

NUMERIC_RE = re.compile(r"\d+")


def looks_like_code(text: str) -> bool:
    """Orthographic shape, not vocabulary: a code has a digit, or a hyphen joining segments
    that aren't a normal dictionary phrase (a real hyphenated English phrase this heuristic
    will also catch, e.g. "well-known" -- that's fine, Jev's classify/confirm stages are what
    filter false positives; this stage's job is to never silently drop a true one)."""
    return bool(re.search(r"\d", text)) or "-" in text


def structural_identifiers(nlp, text: str) -> list[str]:
    """Any copular clause ("X is Y") whose complement is code-shaped names a candidate
    identifier, regardless of what X's text actually is -- no trigger-word list, since the
    subject's literal text is never inspected, only its syntactic role."""
    doc = nlp(text)
    out = []
    for tok in doc:
        if tok.dep_ in ("attr", "acomp") and tok.head.pos_ == "AUX":
            span_text = doc[tok.left_edge.i:tok.right_edge.i + 1].text
            if looks_like_code(span_text):
                out.append(span_text)
    return out


def identifier_candidates(nlp, text: str) -> list[str]:
    """Order-identifier candidates: digit runs UNION the structural spaCy check. Digits-only
    would miss a non-numeric identifier entirely, or truncate a hyphenated alphanumeric one
    -- code must propose every plausible candidate, since Jev only ever picks from what it's
    given, never free-text-generates. A digit run that's a strict substring of a longer
    structural match (e.g. "4471" inside "XJ-4471") is dropped -- it's a fragment of an
    identifier already found whole, not a second, independent one."""
    numeric = NUMERIC_RE.findall(text)
    structural = structural_identifiers(nlp, text)
    merged = list(dict.fromkeys(numeric + structural))
    return [c for c in merged if not any(c != other and c in other for other in merged)]

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


# word document-frequency across the index: a word shared by EVERY entry (e.g. "plumbing"
# here) carries no discriminating signal and must not drive a match on its own, or every
# customer in a one-trade index would fuzzy-match every other one.
_CUSTOMER_WORDS = {name: set(name.replace("'", " ").split()) for name in CUSTOMERS_INDEX}
_WORD_DOC_FREQ = Counter(w for ws in _CUSTOMER_WORDS.values() for w in ws)
_GENERIC_WORDS = {w for w, c in _WORD_DOC_FREQ.items() if c == len(CUSTOMERS_INDEX)}


def fuzzy_search_customer(text: str) -> list[str]:
    exact = text.lower()
    if exact in CUSTOMERS_INDEX:  # an exact match always wins over partial word overlap
        return [exact]
    words = set(exact.replace("'", " ").split()) - _GENERIC_WORDS
    return [name for name in CUSTOMERS_INDEX if words & (_CUSTOMER_WORDS[name] - _GENERIC_WORDS)]


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


def process(client, Choice, nlp, text: str) -> None:
    t_total0 = time.perf_counter()

    # ---- stage 1: candidate spans (plain code, no Jev, no per-word confidence gating)
    candidates = capitalized_runs(text)
    numbers = identifier_candidates(nlp, text)

    # ---- stage 2: classify each whole candidate span, 1 batched call
    t0 = time.perf_counter()
    classify_qs = {
        f"span{k}": Choice(
            instructions=f'Within the text, classify the exact phrase "{span_text}" as a whole. What kind of entity is it?',
            criteria=TYPES,
        )
        for k, (i, j, span_text) in enumerate(candidates)
    }
    resp1 = client.system_one(state={"text": text}, model=MODEL, questions=classify_qs) if classify_qs else None
    t_classify = time.perf_counter() - t0

    org_candidates = [span_text for k, (i, j, span_text) in enumerate(candidates)
                      if resp1.choices[f"span{k}"].choice == "organization" and resp1.choices[f"span{k}"].confidence >= MIN_CONF] if resp1 else []

    # ---- stage 3: resolve each organization candidate against the real customers index
    t0 = time.perf_counter()
    resolved_customers = []  # (raw_name, identity, confidence)
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
    resp2 = client.system_one(state={"text": text}, model=MODEL, questions=rerank_qs) if rerank_qs else None
    for name in org_candidates:
        cands = shortlists[name]
        if len(cands) == 1:
            resolved_customers.append((name, cands[0], 1.0))
        elif len(cands) > 1:
            c = resp2.choices[qid_of[name]]
            if c.choice != "none":
                resolved_customers.append((name, c.choice, c.confidence))
    t_resolve = time.perf_counter() - t0

    # ---- stage 4: attach. Always a real Jev call per candidate number, never a count-based
    # shortcut -- one customer and one number existing is not evidence they're linked.
    t0 = time.perf_counter()
    attach_qs = {
        f"num{k}": Choice(
            instructions=f'Within the text, the number "{n}" was identified as a candidate sales order number. '
                         "Which customer does this specific order belong to, if any?",
            criteria={**{name: CUSTOMERS_INDEX[identity]["desc"] for (name, identity, _) in resolved_customers},
                      "none": "Not a sales order number, or it cannot be attached to a customer."},
        )
        for k, n in enumerate(numbers)
    } if resolved_customers else {}
    resp3 = client.system_one(state={"text": text}, model=MODEL, questions=attach_qs) if attach_qs else None

    records = []
    for k, n in enumerate(numbers):
        if resp3 is None:
            continue
        c = resp3.choices[f"num{k}"]
        if c.choice == "none":
            continue
        identity = next(ident for (name, ident, _) in resolved_customers if name == c.choice)
        records.append({"customer_raw": c.choice, "customer_id": CUSTOMERS_INDEX[identity]["id"], "order_number": n, "conf": c.confidence})
    t_attach = time.perf_counter() - t0

    print(f"text: {text!r}")
    print(f"candidate spans: {[s for (i, j, s) in candidates]}")
    print(f"organization candidates: {org_candidates}")
    print(f"order number candidates (regex + structural parse): {numbers}\n")
    print("resolved sales order record(s):")
    for r in records:
        print(f"  {r}")

    tok2 = resp1.usage.input_tokens if resp1 else 0
    tok3 = resp2.usage.input_tokens if resp2 else 0
    tok4 = resp3.usage.input_tokens if resp3 else 0
    print(f"\nstage 2 classify : {t_classify * 1000:7.1f} ms   ({tok2} tok)")
    print(f"stage 3 resolve  : {t_resolve * 1000:7.1f} ms   ({tok3} tok)")
    print(f"stage 4 attach   : {t_attach * 1000:7.1f} ms   ({tok4} tok)")
    print(f"e2e              : {(time.perf_counter() - t_total0) * 1000:7.1f} ms")
    print("=" * 80)


def main() -> None:
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("Set TYPESAFE_API_KEY first.")
    import spacy
    from typesafe_sdk import Choice, TypeSafeClient

    nlp = spacy.load("en_core_web_sm")
    client = TypeSafeClient()
    client.system_one(state={}, model=MODEL, questions={"w": Choice(instructions="Say ok.", criteria={"ok": "ok", "none": "n/a"})})

    for text in DOCUMENTS:
        process(client, Choice, nlp, text)


if __name__ == "__main__":
    main()
