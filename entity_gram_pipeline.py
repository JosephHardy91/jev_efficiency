"""Full gram-based entity resolution pipeline against Jev. Self-contained (no imports from
the other scratch probes in this session).

Stage 1  detect + type   Jev, chunked + concurrent calls. One Choice per WORD (not every
                         n-gram window -- verified empirically that 2/3-word windows add no
                         detection signal over single words with full-chunk context; multi-
                         word spans are reconstructed in stage 1b, in code, for free).
                         The document is split into chunks of <= CHUNK_SIZE words (this
                         project's own measured B* ~16: CONSIDERATIONS.md notes per-call
                         accuracy holds ~99% to position ~13 then degrades, so one call is
                         never asked to carry more than B* items), chunked on sentence
                         boundaries so no entity straddles a chunk edge. Chunks run
                         concurrently -- wall clock stays ~flat as the document grows;
                         token cost grows linearly (O(L)), not combinatorially (O(MAXN*L)).
Stage 1b merge type-runs  Plain code. Contiguous same-type words (place/org/product/person)
                         above the confidence floor are joined into one candidate span.
Stage 2  fuzzy search    Plain code. The real "source system" -- token-overlap match against
                         a places index. Can return >1 candidate (genuine ambiguity) or 0.
                         This is where a real deployment would hit a real search/DB index.
Stage 3  rerank          Jev, 1 batched call, only for spans whose shortlist has >1 candidate.
                         Candidates are REAL records from stage 2, never invented distractors.
Stage 4  reduce          Plain code, no Jev. Group resolved spans by identity; merge any
                         touching/overlapping spans that agree on identity into one node
                         (weight = best single member, never summed -- corroboration isn't
                         additional evidence); THEN run weighted interval scheduling only
                         across nodes that disagree on identity, to pick the final
                         non-overlapping set of entity mentions in the document.

Run:
    TYPESAFE_API_KEY=... .venv/bin/python entity_gram_pipeline.py
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

MODEL = "jev-latest"
CHUNK_SIZE = 16  # this project's measured B* (CONSIDERATIONS.md): cap items per call near here
MIN_CONF = 0.6
SEARCH_HIT_MS = 0.100  # simulated source-system round trip per fuzzy_search hit

# 3 sentences, 2 separate entities, 41 words total -- long enough to need >1 chunk,
# with entities placed so neither straddles a chunk boundary.
SENTENCES = [
    "I drove across the Golden Gate Bridge yesterday on my way to the office.",
    "The office building downtown was busy with people rushing to morning meetings.",
    "Later that afternoon I walked through Golden Gate Park to clear my head before heading home.",
]

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


def chunk_sentences(sentences: list[str], chunk_size: int) -> list[list[str]]:
    """Group whole sentences into chunks of <= chunk_size words, so no entity straddles
    a chunk boundary. Assumes no single sentence exceeds chunk_size words."""
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        words = s.rstrip(".").split()
        if cur and cur_len + len(words) > chunk_size:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += len(words)
    if cur:
        chunks.append(cur)
    return chunks


def detect_chunk(client, Choice, chunk_sentences_: list[str]):
    """One system_one call: one Choice per word in this chunk, state scoped to the chunk only."""
    text = " ".join(chunk_sentences_)
    words = text.rstrip(".").split()
    qs = {
        f"w{i}": Choice(
            instructions=f'Within the text, consider the exact word "{w}" at word position {i} (0-indexed, this '
                         "chunk only). Does this word name (or is it part of naming) a specific entity, and if so what kind?",
            criteria=TYPES,
        )
        for i, w in enumerate(words)
    }
    resp = client.system_one(state={"text": text}, model=MODEL, questions=qs)
    return words, [(resp.choices[f"w{i}"].choice, resp.choices[f"w{i}"].confidence) for i in range(len(words))], resp.usage.input_tokens


def merge_type_runs(words: list[str], tags: list[tuple[str, float]], min_conf: float) -> list[tuple[int, int, str]]:
    """Merge contiguous words that agree on a (non-"none") type, above min_conf, into spans."""
    spans = []
    i = 0
    while i < len(words):
        choice, conf = tags[i]
        if choice == "none" or conf < min_conf:
            i += 1
            continue
        j = i + 1
        while j < len(words) and tags[j][0] == choice and tags[j][1] >= min_conf:
            j += 1
        spans.append((i, j, " ".join(words[i:j])))
        i = j
    return spans


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

    client = TypeSafeClient()
    client.system_one(state={}, model=MODEL, questions={"w": Choice(instructions="Say ok.", criteria={"ok": "ok", "none": "n/a"})})

    chunks = chunk_sentences(SENTENCES, CHUNK_SIZE)

    # ---- stage 1: detect + type, one call per chunk, chunks run concurrently
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        chunk_results = list(ex.map(lambda c: detect_chunk(client, Choice, c), chunks))
    t_detect = time.perf_counter() - t0
    detect_tokens = sum(tok for _, _, tok in chunk_results)

    all_words: list[str] = []
    all_tags: list[tuple[str, float]] = []
    for words, tags, _ in chunk_results:
        all_words.extend(words)
        all_tags.extend(tags)

    # ---- stage 1b: merge contiguous same-type word runs into candidate spans (no Jev)
    candidates = merge_type_runs(all_words, all_tags, MIN_CONF)
    place_candidates = [(i, j, text) for (i, j, text) in candidates
                         if all_tags[i][0] == "place"]

    # ---- stage 2: fuzzy search (real source system), concurrent hits
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(place_candidates) or 1) as ex:
        shortlist_results = list(ex.map(lambda f: fuzzy_search(f[2]), place_candidates))
    t_search = time.perf_counter() - t0
    shortlists = {text: cands for (i, j, text), cands in zip(place_candidates, shortlist_results)}

    # ---- stage 3: rerank, only spans with a genuine multi-candidate shortlist
    full_text = " ".join(SENTENCES)
    rerank_qs, rid_of = {}, {}
    multi = [f for f in place_candidates if len(shortlists[f[2]]) > 1]
    for k, (i, j, text) in enumerate(multi):
        cands = shortlists[text]
        rid_of[f"r{k}"] = (i, j, text)
        rerank_qs[f"r{k}"] = Choice(
            instructions=f'Within the document, the span "{text}" was flagged as a place and fuzzy-matched these '
                         "candidates from the real places index. Which one does it actually refer to here, if any?",
            criteria={**{c: PLACES_INDEX[c]["desc"] for c in cands}, "none": "None of these candidates is correct."},
        )
    t_rerank = 0.0
    resp2 = None
    if rerank_qs:
        t0 = time.perf_counter()
        resp2 = client.system_one(state={"text": full_text}, model=MODEL, questions=rerank_qs)
        t_rerank = time.perf_counter() - t0

    resolved = []  # (start, end, text, identity, confidence)
    for (i, j, text) in place_candidates:
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

    print(f"words: {len(all_words)}   chunks: {len(chunks)} (<= {CHUNK_SIZE} words each)   "
          f"place candidates: {len(place_candidates)}   needed rerank: {len(rerank_qs)}\n")
    print("final resolved entity mentions:")
    for (i, j, text, identity, weight) in final:
        rec = PLACES_INDEX[identity]
        print(f"  [{i}:{j}] {text!r:22} -> {identity:20} ({rec['id']})  weight={weight:.2f}")

    print(f"\nstage 1 detect+type : {t_detect * 1000:7.1f} ms   ({detect_tokens} tok across {len(chunks)} concurrent calls)")
    print(f"stage 2 fuzzy search: {t_search * 1000:7.1f} ms   ({len(place_candidates)} concurrent hits)")
    print(f"stage 3 rerank      : {t_rerank * 1000:7.1f} ms   ({resp2.usage.input_tokens if resp2 else 0} tok)")
    print(f"stage 4 reduce      : {t_reduce * 1000:7.1f} ms   (pure code)")
    print(f"e2e                 : {(t_detect + t_search + t_rerank + t_reduce) * 1000:7.1f} ms")


if __name__ == "__main__":
    main()
