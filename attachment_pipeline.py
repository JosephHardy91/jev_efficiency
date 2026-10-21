"""Hypothesize-then-confirm attachment: a local dependency parser proposes which customer
each order identifier belongs to, and Jev is only called to arbitrate when the parser
genuinely can't tell -- never as a blanket "ask Jev about everything" step.

This refines sales_order_pipeline.py's stage 4. That version calls Jev once per candidate
number unconditionally, which is correct but wasteful: in a sentence like

    "The order for Ace Plumbing is 91234567, and Jimbo's order is 43328282."

the attachment isn't actually ambiguous -- it's PP-attachment ("order for Ace Plumbing") and
possessive attachment ("Jimbo's order"), both resolvable from the dependency tree alone, for
free, with no model call. Jev should only be spent on the genuinely unresolvable case:

    "Two orders were placed, for Jimbo Plumbing and Ace Plumbing. The numbers are 91234567
    and 43328282."

where neither number's syntactic subject ("numbers") carries any modifier that names a
customer -- the parser can only say "it's one of these candidates, not which one."

Stage 1  candidate spans    Plain code, same as sales_order_pipeline.py: capitalized-word
                            runs (minus each sentence's first word) for customer names,
                            digit-run regex for order identifiers.
Stage 2  classify + resolve Jev, 2 batched calls (classify each span; rerank only genuinely
                            ambiguous fuzzy-search shortlists), same as sales_order_pipeline.py.
Stage 3  hypothesize        Plain code, spaCy dependency parse. For each identifier token,
                            walk to its copular subject and check that subject for a
                            poss ("Jimbo's order") or prep-for ("order for Ace Plumbing")
                            modifier naming a resolved customer. Exactly one match -> a
                            CONFIDENT hypothesis, resolved here, zero Jev cost. No match, or
                            the subject is shared by multiple coordinated identifiers (so
                            syntax can't tell them apart) -> AMBIGUOUS, candidates = every
                            customer resolved anywhere in the document.
Stage 4  confirm            Jev, 1 batched call, ONLY for the ambiguous identifiers, with
                            real candidates from stage 3 -- same "never invent distractors"
                            rule as the other pipelines. Skipped entirely if everything was
                            resolved with confidence in stage 3.

Run:
    pip install -r requirements-nlp.txt && python -m spacy download en_core_web_sm
    TYPESAFE_API_KEY=... .venv/bin/python attachment_pipeline.py
"""
import os
import re
import sys
import time
from collections import Counter

MODEL = "jev-latest"
MIN_CONF = 0.6

DOCUMENTS = [
    # unambiguous syntax: PP-attachment + possessive attachment each name their customer
    # directly -- stage 3 should resolve both with ZERO Jev calls for attachment.
    "The order for Ace Plumbing is 91234567, and Jimbo's order is 43328282.",
    # genuinely ambiguous syntax: "the numbers" has no modifier naming either customer --
    # stage 3 can only say "ambiguous", Jev must use "respectively" to get it right.
    "Two orders were placed, for Jimbo Plumbing and Ace Plumbing respectively. "
    "The numbers are 91234567 and 43328282.",
    # same ambiguous syntax, but no "respectively" and no other disambiguating cue --
    # there's no actual basis to decide; worth seeing whether Jev stays honest about that
    # rather than confidently guessing.
    "Two orders were placed, for Jimbo Plumbing and Ace Plumbing. "
    "The numbers are 91234567 and 43328282.",
]

TYPES = {
    "organization": "A company, agency, team, or institution.",
    "place": "A geographic location, landmark, building, or address.",
    "product": "A product, brand, or model name.",
    "person": "An individual person's name.",
    "none": "The fragment does not name a specific entity of any kind.",
}

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
    spans, offset = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        words = re.findall(r"[A-Za-z']+|\d+", sentence)
        cap = [w[0].isupper() and not w.isdigit() for w in words]
        cap[0:1] = [False]
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


def hypothesize(nlp_doc, number_token, known_names: list[str]) -> tuple[str | None, bool]:
    """Returns (matched_customer_name_or_None, confident). confident=True only when the
    dependency tree names exactly one known customer and this identifier's subject isn't
    shared with another coordinated identifier (which syntax can't individuate between)."""
    tok = number_token
    while tok.dep_ == "conj":  # coordinated identifiers share one subject -- can't tell them apart
        return (None, False)
    copula = tok.head
    subject = next((c for c in copula.children if c.dep_ in ("nsubj", "nsubjpass")), None)
    if subject is None:
        return (None, False)

    candidate_name = None
    for child in subject.children:
        if child.dep_ == "poss":
            candidate_name = child.text
            break
        if child.dep_ == "prep" and child.text.lower() == "for":
            pobj = next((c for c in child.children if c.dep_ == "pobj"), None)
            if pobj is not None:
                candidate_name = nlp_doc[pobj.left_edge.i:pobj.right_edge.i + 1].text
                break
    if candidate_name is None:
        return (None, False)

    matches = [name for name in known_names if candidate_name.lower() in name.lower() or name.lower() in candidate_name.lower()]
    if len(matches) == 1:
        return (matches[0], True)
    return (None, False)


def process(client, Choice, nlp, text: str) -> None:
    t_total0 = time.perf_counter()

    # ---- stage 1: candidate spans
    candidates = capitalized_runs(text)
    numbers = re.findall(r"\d+", text)

    # ---- stage 2: classify spans, then resolve against the customers index
    t0 = time.perf_counter()
    classify_qs = {
        f"span{k}": Choice(
            instructions=f'Within the text, classify the exact phrase "{span_text}" as a whole. What kind of entity is it?',
            criteria=TYPES,
        )
        for k, (i, j, span_text) in enumerate(candidates)
    }
    resp1 = client.system_one(state={"text": text}, model=MODEL, questions=classify_qs) if classify_qs else None
    # type classification is a prior, not a gate: a short possessive reference ("Jimbo's")
    # can correctly classify as "person" in isolation and still be a real customer mention,
    # so anything that gets a real fuzzy-search hit is a candidate regardless of its type.
    org_candidates = [span_text for k, (i, j, span_text) in enumerate(candidates)
                      if (resp1.choices[f"span{k}"].choice == "organization" and resp1.choices[f"span{k}"].confidence >= MIN_CONF)
                      or fuzzy_search_customer(span_text)] if resp1 else []

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
        elif len(cands) > 1 and resp2.choices[qid_of[name]].choice != "none":
            c = resp2.choices[qid_of[name]]
            resolved_customers.append((name, c.choice, c.confidence))
    t_resolve = time.perf_counter() - t0
    known_names = [name for (name, _, _) in resolved_customers]

    # ---- stage 3: hypothesize attachment via dependency parse (plain code, no Jev)
    t0 = time.perf_counter()
    doc = nlp(text)
    num_tokens = {}  # number string -> spaCy token, by character offset
    for m in re.finditer(r"\d+", text):
        span = doc.char_span(m.start(), m.end())
        if span is not None:
            num_tokens[m.group()] = span[0]

    confident, ambiguous = {}, []
    for n in numbers:
        tok = num_tokens.get(n)
        if tok is None:
            ambiguous.append(n)
            continue
        name, is_confident = hypothesize(doc, tok, known_names)
        if is_confident:
            confident[n] = name
        else:
            ambiguous.append(n)
    t_hypothesize = time.perf_counter() - t0

    # ---- stage 4: confirm, ONLY for ambiguous identifiers, candidates = every resolved customer
    t0 = time.perf_counter()
    confirm_qs = {
        f"num{k}": Choice(
            instructions=f'Within the text, the number "{n}" is a sales order identifier, but it could not be '
                         "attached by sentence structure alone. Which customer does this specific order belong to, if any?",
            criteria={**{name: CUSTOMERS_INDEX[identity]["desc"] for (name, identity, _) in resolved_customers},
                      "none": "Cannot be determined, or not a sales order number."},
        )
        for k, n in enumerate(ambiguous)
    } if ambiguous and resolved_customers else {}
    resp3 = client.system_one(state={"text": text}, model=MODEL, questions=confirm_qs) if confirm_qs else None
    t_confirm = time.perf_counter() - t0

    records = []
    for n, name in confident.items():
        identity = next(ident for (nm, ident, _) in resolved_customers if nm == name)
        records.append({"customer_raw": name, "customer_id": CUSTOMERS_INDEX[identity]["id"], "order_number": n,
                         "source": "parsed (0 Jev cost)", "conf": 1.0})
    for k, n in enumerate(ambiguous):
        if resp3 is None:
            continue
        c = resp3.choices[f"num{k}"]
        if c.choice == "none":
            records.append({"customer_raw": None, "customer_id": None, "order_number": n,
                             "source": "Jev confirm (abstained)", "conf": c.confidence})
            continue
        identity = next(ident for (nm, ident, _) in resolved_customers if nm == c.choice)
        records.append({"customer_raw": c.choice, "customer_id": CUSTOMERS_INDEX[identity]["id"], "order_number": n,
                         "source": "Jev confirm", "conf": c.confidence})

    print(f"text: {text!r}")
    print(f"resolved customers: {known_names}")
    print(f"parsed with confidence (0 Jev cost): {confident}")
    print(f"ambiguous, sent to Jev to confirm   : {ambiguous}\n")
    print("final record(s):")
    for r in records:
        print(f"  {r}")

    tok2 = (resp1.usage.input_tokens if resp1 else 0) + (resp2.usage.input_tokens if resp2 else 0)
    tok4 = resp3.usage.input_tokens if resp3 else 0
    print(f"\nstage 2 classify+resolve: {t_resolve * 1000:7.1f} ms   ({tok2} tok)")
    print(f"stage 3 hypothesize     : {t_hypothesize * 1000:7.2f} ms   (pure code, spaCy, 0 tok)")
    print(f"stage 4 confirm         : {t_confirm * 1000:7.1f} ms   ({tok4} tok, {len(confirm_qs)} questions)")
    print(f"e2e                     : {(time.perf_counter() - t_total0) * 1000:7.1f} ms")
    print("=" * 90)


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
