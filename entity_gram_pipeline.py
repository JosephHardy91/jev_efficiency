"""Full gram-based entity resolution pipeline against Jev. Self-contained (no imports from
the other scratch probes in this session).

Stage 1  detect + type   Jev, 1 batched call. Sweep every word n-gram (n=1..MAXN) across the
                         WHOLE sentence (boundaries are never given to Jev or hardcoded).
                         Each gram is typed place / organization / product / person / none.
                         Jev never sees candidate identities at this stage.
Stage 2  fuzzy search    Plain code. The real "source system" -- token-overlap match against
                         a places index. Can return >1 candidate (genuine ambiguity) or 0.
                         This is where a real deployment would hit a real search/DB index.
Stage 3  rerank          Jev, 1 batched call, only for grams whose shortlist has >1 candidate.
                         Candidates are REAL records from stage 2, never invented distractors.
Stage 4  reduce          Plain code, no Jev. Group resolved spans by identity; merge any
                         touching/overlapping spans that agree on identity into one node
                         (weight = best single member, never summed -- corroboration isn't
                         additional evidence); THEN run weighted interval scheduling only
                         across nodes that disagree on identity, to pick the final
                         non-overlapping set of entity mentions in the sentence.

Run:
    TYPESAFE_API_KEY=... .venv/bin/python entity_gram_pipeline.py
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

MODEL = "jev-latest"
MAXN = 3
SEARCH_HIT_MS = 0.100  # simulated source-system round trip per fuzzy_search hit

SENTENCE = "I drove across the Golden Gate Bridge yesterday on my way to the office."

TYPES = {
    "place": "A geographic location, landmark, bridge, building, or address.",
    "organization": "A company, agency, team, or institution.",
    "product": "A product, brand, or model name.",
    "person": "An individual person's name.",
    "none": "The fragment does not name a specific entity of any kind.",
}

# the "source system" -- the only place real identity resolution happens, never Jev.
# includes a deliberate contaminating distractor ("golden casino") that fuzzy-matches
# any gram containing the word "golden", to stress-test stage 3 + stage 4.
PLACES_INDEX = {
    "golden gate bridge": {"id": "GGB-SF-001", "desc": "Suspension bridge across the Golden Gate strait, San Francisco to Marin County."},
    "bay bridge": {"id": "BAYB-SF-002", "desc": "Bridge connecting San Francisco to Oakland."},
    "golden gate park": {"id": "GGPARK-SF-003", "desc": "Large urban park in San Francisco. Not a bridge."},
    "golden casino": {"id": "CASINO-LV-004", "desc": "A casino named Golden Casino. Unrelated to San Francisco or bridges."},
}


def fuzzy_search(text: str) -> list[str]:
    time.sleep(SEARCH_HIT_MS)  # simulated network round trip
    words = set(text.lower().split())
    return [name for name in PLACES_INDEX if words & set(name.split())]


def ngram_spans(words: list[str], maxn: int) -> list[tuple[int, int, str]]:
    return [(i, i + n, " ".join(words[i:i + n]))
            for n in range(1, maxn + 1)
            for i in range(0, len(words) - n + 1)]


def merge_and_reduce(resolved: list[tuple[int, int, str, str, float]]) -> list[tuple[int, int, str, str, float]]:
    """resolved: (start, end, text, identity, confidence). Returns the final non-overlapping set."""
    by_id: dict[str, list[tuple[int, int, str, str, float]]] = {}
    for r in resolved:
        by_id.setdefault(r[3], []).append(r)

    nodes = []
    for identity, items in by_id.items():
        items = sorted(items, key=lambda r: r[0])
        run = [items[0]]
        for r in items[1:]:
            if r[0] <= run[-1][1]:  # touches or overlaps the current run
                run.append(r)
            else:
                nodes.append(_flush(run, identity))
                run = [r]
        nodes.append(_flush(run, identity))

    nodes.sort(key=lambda n: n[1])
    dp = [0.0] * (len(nodes) + 1)
    back = [None] * (len(nodes) + 1)
    for k, node in enumerate(nodes):
        p = k - 1
        while p >= 0 and nodes[p][1] > node[0]:
            p -= 1
        take = node[4] + dp[p + 1]
        if take > dp[k]:
            dp[k + 1], back[k + 1] = take, ("take", p)
        else:
            dp[k + 1], back[k + 1] = dp[k], ("skip", k - 1)

    sel, k = [], len(nodes)
    while k > 0:
        action, prev = back[k]
        if action == "take":
            sel.append(nodes[k - 1])
        k = prev + 1
    return list(reversed(sel))


def _flush(run, identity):
    i, j = min(r[0] for r in run), max(r[1] for r in run)
    best = max(run, key=lambda r: r[4] * (r[1] - r[0]))
    return (i, j, best[2], identity, best[4] * (best[1] - best[0]))


def main() -> None:
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("Set TYPESAFE_API_KEY first.")
    from typesafe_sdk import Choice, TypeSafeClient

    words = SENTENCE.rstrip(".").split()
    spans = ngram_spans(words, MAXN)
    client = TypeSafeClient()

    # warm the connection so stage timings reflect inference, not TLS/connect setup
    client.system_one(state={}, model=MODEL, questions={"w": Choice(instructions="Say ok.", criteria={"ok": "ok", "none": "n/a"})})

    # ---- stage 1: detect + type
    t0 = time.perf_counter()
    detect_qs = {
        f"g{k}": Choice(
            instructions=f'Within the sentence, consider the exact word span "{text}" (word positions {i}-{j - 1}, 0-indexed). '
                         "Does this span name a specific entity, and if so what kind?",
            criteria=TYPES,
        )
        for k, (i, j, text) in enumerate(spans)
    }
    resp1 = client.system_one(state={"sentence": SENTENCE}, model=MODEL, questions=detect_qs)
    t_detect = time.perf_counter() - t0

    flagged = [(i, j, text) for k, (i, j, text) in enumerate(spans)
               if resp1.choices[f"g{k}"].choice == "place" and resp1.choices[f"g{k}"].confidence >= 0.6]

    # ---- stage 2: fuzzy search (real source system), concurrent hits
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(flagged) or 1) as ex:
        shortlist_results = list(ex.map(lambda f: fuzzy_search(f[2]), flagged))
    t_search = time.perf_counter() - t0
    shortlists = {text: cands for (i, j, text), cands in zip(flagged, shortlist_results)}

    # ---- stage 3: rerank, only grams with a genuine multi-candidate shortlist
    rerank_qs, rid_of = {}, {}
    multi = [f for f in flagged if len(shortlists[f[2]]) > 1]
    for k, (i, j, text) in enumerate(multi):
        cands = shortlists[text]
        rid_of[f"r{k}"] = (i, j, text)
        rerank_qs[f"r{k}"] = Choice(
            instructions=f'Within the sentence, the span "{text}" was flagged as a place and fuzzy-matched these '
                         "candidates from the real places index. Which one does it actually refer to here, if any?",
            criteria={**{c: PLACES_INDEX[c]["desc"] for c in cands}, "none": "None of these candidates is correct."},
        )
    t_rerank = 0.0
    resp2 = None
    if rerank_qs:
        t0 = time.perf_counter()
        resp2 = client.system_one(state={"sentence": SENTENCE}, model=MODEL, questions=rerank_qs)
        t_rerank = time.perf_counter() - t0

    resolved = []  # (start, end, text, identity, confidence)
    for (i, j, text) in flagged:
        cands = shortlists[text]
        if len(cands) == 1:
            resolved.append((i, j, text, cands[0], 1.0))
        elif len(cands) > 1:
            rid = next(k for k, v in rid_of.items() if v == (i, j, text))
            c = resp2.choices[rid]
            if c.choice != "none":
                resolved.append((i, j, text, c.choice, c.confidence))

    # ---- stage 4: reduce (no Jev)
    t0 = time.perf_counter()
    final = merge_and_reduce(resolved)
    t_reduce = time.perf_counter() - t0

    print(f"grams swept: {len(spans)}   flagged place: {len(flagged)}   needed rerank: {len(rerank_qs)}\n")
    print("final resolved entity mentions:")
    for (i, j, text, identity, weight) in final:
        rec = PLACES_INDEX[identity]
        print(f"  [{i}:{j}] {text!r:22} -> {identity:20} ({rec['id']})  weight={weight:.2f}")

    print(f"\nstage 1 detect+type : {t_detect * 1000:7.1f} ms   ({resp1.usage.input_tokens} tok)")
    print(f"stage 2 fuzzy search: {t_search * 1000:7.1f} ms   ({len(flagged)} concurrent hits)")
    print(f"stage 3 rerank      : {t_rerank * 1000:7.1f} ms   ({resp2.usage.input_tokens if resp2 else 0} tok)")
    print(f"stage 4 reduce      : {t_reduce * 1000:7.1f} ms   (pure code)")
    print(f"e2e                 : {(t_detect + t_search + t_rerank + t_reduce) * 1000:7.1f} ms")


if __name__ == "__main__":
    main()
