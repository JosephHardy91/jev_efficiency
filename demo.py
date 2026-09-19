"""Email -> two-pass Jev spec extraction -> simulated product search -> Jev rerank.

Every field except qty is a Choice over a real option set. Options are framed per call:
pass 2's options depend on pass 1's answers (size list follows the size unit, rating list
follows the rating system, follow-up questions follow the category).

    .venv/bin/python demo.py                       # full pipeline (needs TYPESAFE_API_KEY)
    .venv/bin/python demo.py --copies 10           # 100 emails: watch the packer split into batches
    .venv/bin/python demo.py --budget 20000        # est. tokens per call (Jev limit: 64k)
    .venv/bin/python demo.py --min-conf 0.7        # fields below this confidence become null
    .venv/bin/python demo.py --min-conf 0          # no nulling: the min-conf curve is then exact for every threshold
    .venv/bin/python demo.py --wrong-cost 3        # curve utility: a wrong field costs 3, an abstention 0
    .venv/bin/python demo.py --offline             # search only, no API key
"""

import json
import math
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

MODEL = "jev-latest"
NONE = "none"
TOP_K = 10

# --------------------------------------------------------------------------- catalog
CATALOG: list[dict] = []


def add(material, category, size, rating, angle=None):
    label = f"{angle:g}° {category}" if angle else category
    CATALOG.append(
        {
            "sku": f"{material}-{category.replace(' ', '')}{f'{angle:g}' if angle else ''}-{size:g}-{rating}",
            "material": material, "category": category, "angle": angle, "size": float(size), "rating": rating,
            "desc": f"{material} {label}, {size:g} in IPS, {rating}",
        }
    )


for s in (2, 3, 4, 6, 8, 10, 12, 14, 16):
    for dr in ("DR7", "DR9", "DR11", "DR13.5", "DR17", "DR21"):
        add("HDPE", "pipe", s, dr)
    for dr in ("DR11", "DR17"):
        add("HDPE", "elbow", s, dr, 90)
        add("HDPE", "elbow", s, dr, 45)
        for cat in ("tee", "stub end", "flange adapter"):
            add("HDPE", cat, s, dr)
for s in (1, 2, 3, 4, 6, 8, 10, 12):
    for sch in ("SCH40", "SCH80"):
        add("PVC", "pipe", s, sch)
for s in (4, 6, 8, 10, 12):
    for dr in ("DR18", "DR25"):
        add("FPVC", "pipe", s, dr)

# ---------------------------------------------------------------------------- emails
# gold = (material, category, angle, size_in, rating); None = no such SKU in the catalog
EMAILS = [
    ('Hi, need 2,400 ft of 6" DR 11 HDPE pipe, PE4710, for the Henderson lift station job. PO 88213. Need by Oct 14.',
     ("HDPE", "pipe", None, 6, "DR11")),
    ("Can you quote 12 pcs of 8 inch 90 degree elbows, butt fusion, DR17 poly? Thanks - Mike, 555-0142",
     ("HDPE", "elbow", 90, 8, "DR17")),
    ('We are replacing a 10" main. Looking for 500 feet, SDR 13.5 pipe in black HDPE. Job #4471, need in 3 weeks.',
     ("HDPE", "pipe", None, 10, "DR13.5")),
    ("Please send price on 4 tees, 12-inch, DR 11, HDPE. Also let me know lead time - was told 6 weeks last time.",
     ("HDPE", "tee", None, 12, "DR11")),
    ('Quote 1,000 LF of 8" FPVC DR 25 for the chemical containment job. Need it in 2 wks.',
     ("FPVC", "pipe", None, 8, "DR25")),
    ("Need 20 flange adapters, 6 in, DR 17 HDPE, w/ backup rings for site 3.",
     ("HDPE", "flange adapter", None, 6, "DR17")),
    ("Order for 300 ft 3 inch schedule 80 PVC pipe. PO#2291, ship to the Tulsa yard.",
     ("PVC", "pipe", None, 3, "SCH80")),
    ("Need 8 stub ends, 14-inch, DR 11 PE4710 poly. Same as the last order (#5530).",
     ("HDPE", "stub end", None, 14, "DR11")),
    ("Looking for 2 inch DR 9 poly pipe, 200 ft coil, for irrigation. Call 918-555-0177 if questions.",
     ("HDPE", "pipe", None, 2, "DR9")),
    ("Need 45 degree elbows, qty 30, 16 in, HDPE DR 21. Order needed by the 3rd.",
     None),  # DR21 fittings aren't stocked: correct answer is "none of these"
]


# Ground truth for every extraction field, per email (what the email actually asks for).
# size_unit is the US default (IPS): no email states it, so a null there is an abstention, IPS is right, DIPS/mm is wrong.
def _t(qty, uom, cat, mat, size, system, rating, angle=None):
    return {"qty": qty, "qty_uom": uom, "category": cat, "material": mat, "size_unit": "IPS", "rating_system": system,
            "size_in": float(size), "size_raw": f"{size} IPS", "rating": f"{system}{rating}", "angle": angle}


TRUTH = [
    _t(2400, "FT", "pipe", "HDPE", 6, "DR", "11"), _t(12, "EA", "elbow", "HDPE", 8, "DR", "17", "90"),
    _t(500, "FT", "pipe", "HDPE", 10, "DR", "13.5"), _t(4, "EA", "tee", "HDPE", 12, "DR", "11"),
    _t(1000, "FT", "pipe", "FPVC", 8, "DR", "25"), _t(20, "EA", "flange adapter", "HDPE", 6, "DR", "17"),
    _t(300, "FT", "pipe", "PVC", 3, "SCH", "80"), _t(8, "EA", "stub end", "HDPE", 14, "DR", "11"),
    _t(200, "FT", "pipe", "HDPE", 2, "DR", "9"), _t(30, "EA", "elbow", "HDPE", 16, "DR", "21", "45"),  # #10: no such SKU stocked
]
PASS1_FIELDS = ["qty", "qty_uom", "category", "material", "size_unit", "rating_system"]
PASS2_FIELDS = ["size_in", "rating", "angle"]


def gold_sku(gold) -> str | None:
    if gold is None:
        return None
    m, c, a, s, r = gold
    return next(p["sku"] for p in CATALOG if (p["material"], p["category"], p["angle"], p["size"], p["rating"]) == (m, c, a, float(s), r))


# ------------------------------------------------------ simulated fast search (no Jev)
def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9.]+", text.lower().replace(",", ""))


# keyword index over material/category/angle/size only: rating tokens are too ambiguous in raw email text
DOCS = {p["sku"]: tokens(f"{p['material']} {p['category']} {p['angle'] or ''} {p['size']:g} in inch") for p in CATALOG}
DF = Counter(t for toks in DOCS.values() for t in set(toks))


def fast_search(query: str, k: int = TOP_K) -> list[dict]:
    q = set(tokens(query))
    scored = [(sum(math.log(len(CATALOG) / DF[t]) for t in set(DOCS[p["sku"]]) if t in q), p) for p in CATALOG]
    scored.sort(key=lambda x: -x[0])  # stable: ties keep catalog order
    return [p for _, p in scored[:k]]


# ------------------------------------------------------------------ option sets (Jev)
QTY_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")  # qty is the one free field: offer every number found

QTY_UOMS = {
    "FT": "Feet / linear feet (LF), for pipe by the length", "EA": "Each / pieces / units (fittings, valves)",
    "RL": "Rolls or coils", "SQ IN": "Square inches", "SQ FT": "Square feet", "LB": "Pounds",
    "IN": "Linear inches", "M": "Meters", "BX": "Boxes / cartons", "KT": "Kits / sets", "PR": "Pairs",
}
CATEGORIES = {
    "pipe": "Straight pipe, by length or coil", "elbow": "Elbow / ell / bend, any angle", "tee": "Tee",
    "cross": "Cross", "wye": "Wye / lateral", "reducer": "Concentric or eccentric reducer",
    "stub end": "Stub end / flange stub", "flange adapter": "Flange adapter", "coupling": "Coupling / electrofusion coupler",
    "cap": "End cap", "spool": "Prefabricated spool / fabricated assembly", "saddle": "Saddle / branch saddle",
    "sweep": "Long-sweep bend", "mitered fitting": "Mitered / segmented fitting", "manhole": "Manhole / riser",
    "valve": "Valve", "gasket": "Gasket", "backing ring": "Backing / flange ring", "bolt kit": "Bolts / hardware kit",
    "transition": "Mechanical or transition fitting", "tap": "Tapping sleeve / tap", "sheet": "Sheet or plate", "other": "Anything else",
}
MATERIALS = {
    "HDPE": "Black polyethylene / poly / PE4710 / PE100", "PVC": "Standard PVC", "FPVC": "Fusible PVC",
    "DI": "Ductile iron", "PP": "Polypropylene", "PVDF": "PVDF", "SS": "Stainless steel", "CS": "Carbon steel", "other": "Anything else",
}
SIZE_UNITS = {
    "IPS": "Iron pipe size, nominal inches (default for US HDPE/PVC pipe: 4 in, 6 in, 12 in)",
    "DIPS": "Ductile iron pipe size, nominal inches (DR-DIPS product)", "mm": "Metric outside diameter in millimeters",
}
RATING_SYSTEMS = {
    "DR": "Dimension ratio: DR / SDR (HDPE, FPVC, PVC C900)", "SCH": "Pipe schedule (Sch 40, 80)", "CLASS": "Pressure class (Class 150, 250)",
}
SIZE_OPTIONS = {
    "IPS": ["1", "1.25", "1.5", "2", "2.5", "3", "4", "5", "6", "8", "10", "12", "14", "16", "18", "20", "22", "24", "26", "28", "30", "32", "34", "36", "42", "48", "54"],
    "DIPS": ["4", "6", "8", "10", "12", "14", "16", "18", "20", "24", "30", "36", "42", "48", "54", "60", "64"],
    "mm": ["20", "25", "32", "40", "50", "63", "75", "90", "110", "125", "140", "160", "180", "200", "225", "250", "280", "315", "355", "400", "450", "500", "560", "630"],
}
RATING_OPTIONS = {
    "DR": ["7", "9", "11", "13.5", "15.5", "17", "18", "19", "21", "25", "26", "32.5", "41"],
    "SCH": ["10", "40", "80", "120", "160"],
    "CLASS": ["125", "150", "200", "250", "300", "350", "400", "600"],
}
# category -> (field, question, options): asked in pass 2 only when that category was chosen
FOLLOWUPS = {
    "elbow": ("angle", "What bend angle (degrees) is the elbow?", {a: None for a in ("90", "60", "45", "30", "22.5", "11.25")}),
    "tee": ("configuration", "Is the tee equal or reducing (a smaller branch)?",
            {"equal": "Same size on all three ends", "reducing": "Branch is smaller than the run"}),
}


def with_none(options: dict, note="Not stated in the email.") -> dict:
    return {**options, NONE: note}


def opts(keys: list[str]) -> dict:
    return {k: None for k in keys}


# --------------------------------------------------------------------------- knobs
def arg(name: str, default, cast=float):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


BUDGET = arg("--budget", 12_000, int)  # est. input tokens per call; published limit is 64k (32k state + longest question). Swept 3k-48k: harmonic mean of accuracy and throughput peaks at ~12k (9 products/call)
CONF_MIN = arg("--min-conf", 0.6)  # Choice confidence below this -> the field is treated as null
COPIES = arg("--copies", 1, int)
MAX_RETRIES = arg("--max-retries", 2, int)  # SDK default 2 silently retries 429s; use 0 to make every 429 visible and counted
QUIET, NO_REPORT, NO_ORACLE = "--quiet" in sys.argv, "--no-report" in sys.argv, "--no-oracle" in sys.argv
WRONG_COST = arg("--wrong-cost", 1.0)  # min-conf curve utility: +1 per correct field, -WRONG_COST per wrong field, 0 per abstention
STATS = {"products": 0, "tokens": 0, "retries_or_splits": 0, "rate_limited": 0, "retry_after_s": []}
RATE_TPS, RATE_RPS = 250_000, 20  # published limits: 250k input tokens/s, 1,200 requests/min
# requests in flight so that (tokens per call x calls in flight) stays inside the token rate (assuming ~1 s per call), and inside the request rate
WORKERS = arg("--workers", max(1, min(RATE_RPS, RATE_TPS // BUDGET)), int)
CPT = 1.8  # chars per token, measured on this JSON-heavy payload (option lists tokenize densely); recalibrated after every call


# ------------------------------------------------- question specs (plain dicts, so we can size them)
def q(instr: str, crit: dict) -> dict:
    return {"instr": instr, "crit": crit}


def qchars(qd: dict) -> int:
    return len(qd["instr"]) + sum(len(k) + len(v or "") + 6 for k, v in qd["crit"].items())


def pass1_qs(k: int, email: str) -> dict:
    p = f"About `products[{k}].text` only: "
    qtys = list(dict.fromkeys(m.group() for m in QTY_RE.finditer(email)))
    return {
        f"{k}:qty": q(p + "how many of the requested product does the customer want to order?", with_none(opts(qtys))),
        f"{k}:qty_uom": q(p + "what unit of measure is the requested quantity in?", with_none(QTY_UOMS, "No unit stated.")),
        f"{k}:category": q(p + "what category of product is being requested?", CATEGORIES),
        f"{k}:material": q(p + "what material is the requested product?", MATERIALS),
        f"{k}:size_unit": q(p + "which sizing convention applies to the requested size?", with_none(SIZE_UNITS, "Nothing indicates a convention.")),
        f"{k}:rating_system": q(p + "which rating system does the requested pressure rating use?", with_none(RATING_SYSTEMS)),
    }


def pass2_qs(k: int, spec: dict, worst_case: bool = False) -> dict:
    """Options framed by pass 1: size list <- unit, rating list <- system, follow-ups <- category."""
    p = f"About `products[{k}]` only: "
    unit = "IPS" if worst_case else (spec["size_unit"] or "IPS")  # unit unknown/null: IPS is the working default
    system = "DR" if worst_case else spec["rating_system"]
    out = {f"{k}:size": q(p + f"what nominal size ({unit}) is requested?", with_none(opts(SIZE_OPTIONS[unit])))}
    if system:
        out[f"{k}:rating"] = q(p + f"which {system} value is requested?", with_none(opts(RATING_OPTIONS[system])))
    follow = FOLLOWUPS.get("elbow" if worst_case else spec["category"])
    if follow:
        out[f"{k}:{follow[0]}"] = q(p + follow[1], with_none(follow[2]))
    return out


def rerank_qs(k: int, shortlist: list[dict], rng: random.Random) -> dict:
    cands = shortlist[:]
    rng.shuffle(cands)  # don't let search order leak in as position bias
    crit = {c["sku"]: c["desc"] for c in cands}
    crit[NONE] = "None of these products is what the customer asked for."
    return {f"{k}:match": q(f"About `products[{k}]` only: which catalog product exactly matches what the customer is asking to buy "
                            "(category, bend angle, material, nominal size, and DR/schedule all agree; ignore any field that is null)?", crit)}


def per_product_chars(email: str) -> int:
    """Worst case over the three stages, so one packing works for the whole pipeline."""
    p1 = sum(qchars(v) for v in pass1_qs(0, email).values())
    p2 = sum(qchars(v) for v in pass2_qs(0, {}, worst_case=True).values())
    rr = 1100  # 10 candidate SKUs + descriptions
    return len(email) + max(p1 + 60, p2 + 60, rr + 250)


def pack(items: list[dict]) -> list[list[dict]]:
    """Greedy: keep adding products to the call until the estimated token budget is hit."""
    batches, cur, used = [], [], 0.0
    for it in items:
        cost = per_product_chars(it["email"]) / CPT
        if cur and used + cost > BUDGET:
            batches.append(cur)
            cur, used = [], 0.0
        cur.append(it)
        used += cost
    return batches + ([cur] if cur else [])


# ------------------------------------------------------------------------- live stage tracker
class Tracker:
    STAGES = ["search", "pass1", "pass2", "rerank"]
    LABELS = {"search": "search (code)", "pass1": "pass 1 · Jev", "pass2": "pass 2 · Jev", "rerank": "rerank · Jev"}

    def __init__(self):
        self.t0 = time.perf_counter()
        self.lock = threading.Lock()
        self.batches: dict[int, dict] = {}

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def new_batch(self, n: int) -> int:
        with self.lock:
            bid = len(self.batches) + 1
            self.batches[bid] = {"id": bid, "n": n, "stages": {}, "split": False}
            return bid

    def start(self, bid: int, stage: str, **info):
        with self.lock:
            self.batches[bid]["stages"][stage] = {"t0": self.now(), "t1": None, **info}
            QUIET or print(f"[{self.now():5.2f}s] batch {bid} ▶ {self.LABELS[stage]}", flush=True)

    def done(self, bid: int, stage: str, **info):
        with self.lock:
            st = self.batches[bid]["stages"][stage]
            st.update(t1=self.now(), **info)
            extra = f"  est {st['est']:,} / actual {st['actual']:,} tok" if "actual" in st else ""
            QUIET or print(f"[{self.now():5.2f}s] batch {bid} ✓ {self.LABELS[stage]:<14} {st['t1'] - st['t0']:5.2f}s  ({self.batches[bid]['n']} products{extra})", flush=True)


def call(R, bid: int, stage: str, state: dict, qs: dict):
    """One Jev request carrying every product's questions for this stage."""
    global CPT
    chars = len(json.dumps(state)) + sum(qchars(v) for v in qs.values())
    R["tracker"].start(bid, stage, est=int(chars / CPT))
    resp = R["client"].system_one(
        state=state, model=MODEL,
        questions={k: R["Choice"](instructions=v["instr"], criteria=v["crit"]) for k, v in qs.items()},
    )
    actual = resp.usage.input_tokens
    CPT = 0.5 * CPT + 0.5 * (chars / max(actual, 1))  # calibrate the estimator on real usage
    R["tracker"].done(bid, stage, actual=actual)
    with R["tracker"].lock:
        STATS["tokens"] += actual
    return resp.choices, actual


def unpack(c, k: int) -> dict:
    """Group one product's answers; low-confidence answers are treated as null."""
    out = {}
    for qid, v in c.items():
        kk, field = qid.split(":", 1)
        if int(kk) == k:
            low = v.confidence < CONF_MIN and v.choice != NONE
            out[field] = {"choice": v.choice, "conf": v.confidence, "probs": dict(v.probabilities),
                          "value": None if (v.choice == NONE or low) else v.choice, "nulled": low}
    return out


def run_batch(R, items: list[dict], tries: int = 0) -> list[dict]:
    T = R["tracker"]
    bid = T.new_batch(len(items))
    try:
        T.start(bid, "search")
        for it in items:
            it["shortlist"] = fast_search(it["email"])
        T.done(bid, "search")

        # ---- pass 1: independent fields for every product, one request
        qs = {}
        for k, it in enumerate(items):
            qs.update(pass1_qs(k, it["email"]))
        c1, tok1 = call(R, bid, "pass1", {"products": [{"text": it["email"]} for it in items]}, qs)
        for k, it in enumerate(items):
            raw = unpack(c1, k)
            v = {f: raw[f]["value"] for f in raw}
            it["spec"] = {"qty": int(float(v["qty"].replace(",", ""))) if v["qty"] else None, "qty_uom": v["qty_uom"],
                          "category": v["category"], "material": v["material"], "size_unit": v["size_unit"], "rating_system": v["rating_system"]}
            it["passes"] = [{"name": "pass 1 · independent fields", "fields": raw}]

        # ---- pass 2: options framed by pass 1's answers
        qs = {}
        for k, it in enumerate(items):
            qs.update(pass2_qs(k, it["spec"]))
        state = {"products": [{"text": it["email"], "category": it["spec"]["category"], "material": it["spec"]["material"]} for it in items]}
        c2, tok2 = call(R, bid, "pass2", state, qs)
        for k, it in enumerate(items):
            raw, sp = unpack(c2, k), it["spec"]
            unit = sp["size_unit"] or "IPS"
            sp["size_raw"] = f"{raw['size']['value']} {unit}" if raw["size"]["value"] else None
            sp["size_in"] = float(raw["size"]["value"]) if raw["size"]["value"] and unit == "IPS" else None
            sp["rating"] = f"{sp['rating_system']}{raw['rating']['value']}" if "rating" in raw and raw["rating"]["value"] else None
            for f in raw:
                if f not in ("size", "rating"):
                    sp[f] = raw[f]["value"]
            it["passes"].append({"name": "pass 2 · options framed by pass 1", "fields": raw})
            it["min_conf"] = min(v["conf"] for ps in it["passes"] for v in ps["fields"].values())

        # ---- rerank: each product's own 10 candidates are that question's options
        qs = {}
        for k, it in enumerate(items):
            qs.update(rerank_qs(k, it["shortlist"], random.Random(it["i"])))
        state = {"products": [{"text": it["email"], "extracted": {a: b for a, b in it["spec"].items()}} for it in items]}
        c3, tok3 = call(R, bid, "rerank", state, qs)
        for k, it in enumerate(items):
            ans = c3[f"{k}:match"]
            it["ranked"] = sorted(ans.probabilities.items(), key=lambda kv: -kv[1])
            it["rconf"], it["bid"] = ans.confidence, bid
        for it in items:
            it["tokens"] = (tok1 + tok2 + tok3) / len(items)
        with T.lock:
            STATS["products"] += len(items)
        return items
    except Exception as ex:
        from typesafe_sdk import TypeSafeRateLimitError

        T.batches[bid]["split"] = True  # this attempt is superseded either way
        if isinstance(ex, TypeSafeRateLimitError):
            # a 429 is nonfulfillment, not a size problem: wait as told and resend the SAME batch (halving would send more requests)
            wait = (ex.retry_after_ms or 1000) / 1000
            with T.lock:
                STATS["rate_limited"] += 1
                STATS["retry_after_s"].append(wait)
            if tries >= 5:
                raise
            print(f"[{T.now():5.2f}s] batch {bid} 429 -> Retry-After {wait:.2f}s, resending {len(items)} products", flush=True)
            time.sleep(wait)
            return run_batch(R, items, tries + 1)
        # anything else (e.g. context too large): split the batch and retry
        STATS["retries_or_splits"] += 1
        if len(items) == 1:
            raise
        print(f"[{T.now():5.2f}s] batch {bid} ✗ {type(ex).__name__}: {str(ex)[:90]} -> splitting {len(items)} products in half", flush=True)
        mid = len(items) // 2
        return run_batch(R, items[:mid]) + run_batch(R, items[mid:])



# ------------------------------------------------------------- per-phase scoring vs ground truth
def field_score(done: list[dict], fields: list[str]) -> list[str]:
    """Correct / wrong / abstained per field. A null is an abstention, not a wrong answer; accuracy is over answered fields."""
    lines = []
    for f in fields:
        right = wrong = abst = 0
        for it in done:
            want = TRUTH[it["i"] % len(EMAILS)][f]
            if want is None:
                continue  # field doesn't apply to this product (e.g. no bend angle on a pipe)
            got = it["spec"].get(f)
            right += got == want
            wrong += got is not None and got != want
            abst += got is None
        ans = right + wrong
        lines.append(f"    {f:<14} {right:>3}/{ans:<3} correct of answered ({(right / ans if ans else 0):6.1%})   abstained {abst:>3}   wrong {wrong:>3}")
    return lines


def rerank_breakdown(done: list[dict], top: dict[int, str]) -> str:
    """Ground truth for the rerank is defined by its input: the gold SKU if the shortlist has it, otherwise none."""
    with_gold = [it for it in done if it["gold"] in [p["sku"] for p in it["shortlist"]]]
    no_gold = [it for it in done if it not in with_gold]
    got = sum(top[it["i"]] == it["gold"] for it in with_gold)
    said_none = sum(top[it["i"]] == NONE for it in with_gold)
    right_none = sum(top[it["i"]] == NONE for it in no_gold)
    return (f"{got + right_none}/{len(done)} correct | gold in shortlist ({len(with_gold)}): picked gold {got}, other SKU {len(with_gold) - got - said_none}, said none {said_none}"
            f" | gold absent ({len(no_gold)}): said none {right_none}, picked a SKU {len(no_gold) - right_none}")


ORACLE_TOP: dict[int, str] = {}


def oracle_rerank(R, done: list[dict]) -> dict[int, str]:
    """Rerank fed the TRUE specs on the same shortlists: isolates the rerank from extraction errors."""
    def one(b):
        qs = {}
        for k, it in enumerate(b):
            qs.update(rerank_qs(k, it["shortlist"], random.Random(it["i"])))
        state = {"products": [{"text": it["email"], "extracted": TRUTH[it["i"] % len(EMAILS)]} for it in b]}
        r = R["client"].system_one(state=state, model=MODEL, questions={k: R["Choice"](instructions=v["instr"], criteria=v["crit"]) for k, v in qs.items()})
        return {it["i"]: r.choices[f"{k}:match"].choice for k, it in enumerate(b)}

    with ThreadPoolExecutor(8) as ex:
        return {i: c for part in ex.map(one, pack(done)) for i, c in part.items()}


# ------------------------------------------------ min-conf tuning: precision vs coverage, re-thresholded offline
CURVE_T = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
CURVE_ROWS: list[dict] = []


def spec_at(it: dict, t: float) -> dict:
    """Rebuild a product's spec as if the run had used --min-conf t, from the stored per-field confidences (no API calls).
    Mirrors run_batch's assembly, including the cascade (no rating system => no rating; category picks the follow-up).
    Exact for t >= the run's own --min-conf: fields nulled at run time were never asked in pass 2."""
    def val(f):
        return None if f["choice"] == NONE or f["conf"] < t else f["choice"]

    p1, p2 = it["passes"][0]["fields"], it["passes"][1]["fields"]
    v = {k: val(f) for k, f in p1.items()}
    sp = {"qty": int(float(v["qty"].replace(",", ""))) if v["qty"] else None, "qty_uom": v["qty_uom"], "category": v["category"],
          "material": v["material"], "size_unit": v["size_unit"], "rating_system": v["rating_system"]}
    unit = sp["size_unit"] or "IPS"
    size = val(p2["size"])
    sp["size_raw"] = f"{size} {unit}" if size else None
    sp["size_in"] = float(size) if size and unit == "IPS" else None
    rating = val(p2["rating"]) if "rating" in p2 and sp["rating_system"] else None
    sp["rating"] = f"{sp['rating_system']}{rating}" if rating else None
    follow = FOLLOWUPS.get(sp["category"])
    if follow:
        sp[follow[0]] = val(p2[follow[0]]) if follow[0] in p2 else None
    return sp


def conf_curve(done: list[dict]) -> str:
    """Accuracy on answered fields vs coverage at each threshold, per pass and pooled, plus HM(accuracy, coverage) and a
    utility (+1 right, -WRONG_COST wrong, 0 abstained; per applicable field). Picks the best threshold by each."""
    diff = sum(any(spec_at(it, CONF_MIN).get(k) != it["spec"].get(k) for k in set(spec_at(it, CONF_MIN)) | set(it["spec"])) for it in done)
    CURVE_ROWS.clear()
    for t in sorted(set(CURVE_T) | {CONF_MIN}):
        row = {"t": t}
        for name, fields in (("p1", PASS1_FIELDS), ("p2", PASS2_FIELDS), ("all", PASS1_FIELDS + PASS2_FIELDS)):
            right = wrong = abst = 0
            for it in done:
                sp, truth = spec_at(it, t), TRUTH[it["i"] % len(EMAILS)]
                for f in fields:
                    if truth[f] is None:
                        continue
                    got = sp.get(f)
                    right += got == truth[f]
                    wrong += got is not None and got != truth[f]
                    abst += got is None
            n, ans = right + wrong + abst, right + wrong
            acc, cov = (right / ans if ans else 1.0), (ans / n if n else 0.0)
            row[name] = {"acc": acc, "cov": cov, "right": right, "wrong": wrong, "abst": abst,
                         "hm": 2 * acc * cov / (acc + cov) if acc + cov else 0.0, "util": (right - WRONG_COST * wrong) / n if n else 0.0}
        CURVE_ROWS.append(row)
    exact = [r for r in CURVE_ROWS if r["t"] >= CONF_MIN]  # rows below the run's own threshold are approximate: never picked
    best_hm = max(exact, key=lambda r: (r["all"]["hm"], -r["t"]))["t"]
    best_u = max(exact, key=lambda r: (r["all"]["util"], -r["t"]))["t"]
    out = [f"  5. --min-conf curve (re-thresholded offline from stored confidences; accuracy is over answered fields, wrong-cost {WRONG_COST:g})",
           f"    check: rebuilding specs at the run's own threshold {CONF_MIN:g} reproduces them for {len(done) - diff}/{len(done)} products",
           "    t      | pass 1 acc / cov | pass 2 acc / cov | pooled acc  cov    wrong  abst |  HM(acc,cov)  utility"]
    for r in CURVE_ROWS:
        a, p1, p2 = r["all"], r["p1"], r["p2"]
        approx = "~" if r["t"] < CONF_MIN else " "
        tag = " <- run" if r["t"] == CONF_MIN else ""
        tag += " *HM" if r["t"] == best_hm else ""
        tag += " *utility" if r["t"] == best_u else ""
        out.append(f"   {approx}{r['t']:<5.2f} | {p1['acc']:>7.1%} / {p1['cov']:<6.1%} | {p2['acc']:>7.1%} / {p2['cov']:<6.1%} | "
                   f"{a['acc']:>9.1%} {a['cov']:>6.1%} {a['wrong']:>6} {a['abst']:>5} | {a['hm']:>11.3f} {a['util']:>8.3f}{tag}")
    out.append(f"    best by HM(accuracy, coverage): {best_hm:g}; best by utility: {best_u:g}."
               + ("  Rows marked ~ are below the run's --min-conf and approximate; run with --min-conf 0 for an exact curve." if CONF_MIN > 0 else ""))
    return "\n".join(out)


def scorecard(R, done: list[dict]) -> str:
    n = len(done)
    with_gold = [it for it in done if it["gold"]]
    in_top = sum(it["gold"] in [p["sku"] for p in it["shortlist"]] for it in with_gold)
    rank1 = sum(it["shortlist"][0]["sku"] == it["gold"] for it in with_gold)
    out = [f"PER-PHASE SCORES vs ground truth ({n} products; null = abstention, never counted as wrong)",
           f"  1. search (code)      gold SKU in top {TOP_K}: {in_top}/{len(with_gold)} products that have a stocked SKU (at rank 1: {rank1})",
           "  2. pass 1 (Jev)"] + field_score(done, PASS1_FIELDS) + ["  3. pass 2 (Jev), cascades from pass 1's answers"] + field_score(done, PASS2_FIELDS)
    top_pipe = {it["i"]: it["ranked"][0][0] for it in done}
    out.append("  4. rerank (Jev), scored against its own input (gold if shortlisted, else none)")
    out.append(f"    fed Jev's extracted specs: {rerank_breakdown(done, top_pipe)}")
    if not NO_ORACLE:
        ORACLE_TOP.update(oracle_rerank(R, done))
        out.append(f"    fed the true specs:        {rerank_breakdown(done, ORACLE_TOP)}")
    out.append(conf_curve(done))
    return "\n".join(out)


# ------------------------------------------------------------------------------ main
def show(v, dash="·"):
    return dash if v is None else v


def describe(spec: dict) -> str:
    kind = f"{spec['angle']}° {spec['category']}" if spec.get("angle") else spec["category"]
    return " | ".join(str(show(x, "?")) for x in (f"{show(spec['qty'], '?')} {show(spec['qty_uom'], '?')}", spec["material"], kind, spec["size_raw"], spec["rating"]))


def main() -> None:
    offline = "--offline" in sys.argv
    items = [{"i": n * len(EMAILS) + i, "email": e, "gold": gold_sku(g)} for n in range(COPIES) for i, (e, g) in enumerate(EMAILS)]

    if offline:
        for it in items[: len(EMAILS)]:
            skus = [p["sku"] for p in fast_search(it["email"])]
            print(f"#{it['i']:>2} gold={it['gold']} search_rank={skus.index(it['gold']) + 1 if it['gold'] in skus else None}")
        return

    import os

    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("Set TYPESAFE_API_KEY (or run with --offline).")
    from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient

    batches = pack(items)
    R = {"client": TypeSafeClient(timeout=120.0, retry=RetryPolicy(max_retries=MAX_RETRIES)), "Choice": Choice, "tracker": Tracker()}
    print(f"{len(items)} products packed into {len(batches)} batch(es) of {[len(b) for b in batches]} "
          f"(budget {BUDGET:,} est. tok/call, {WORKERS} in flight, null fields below conf {CONF_MIN})\n", flush=True)

    R["tracker"].t0 = time.perf_counter()
    stop = threading.Event()

    def ticker():  # sustained-throughput readout every 5 s
        last_p = last_t = 0
        while not stop.wait(5):
            now = R["tracker"].now()
            print(f"  [{now:5.1f}s] {STATS['products']:>6,} products done | last 5s: {(STATS['products'] - last_p) / 5:6.1f} products/s, "
                  f"{(STATS['tokens'] - last_t) / 5:>8,.0f} tok/s | 429s {STATS['rate_limited']} | splits/errors {STATS['retries_or_splits']}", flush=True)
            last_p, last_t = STATS["products"], STATS["tokens"]

    if QUIET:
        threading.Thread(target=ticker, daemon=True).start()
    with ThreadPoolExecutor(max(min(WORKERS, len(batches)), 1)) as ex:
        done = [it for res in ex.map(lambda b: run_batch(R, b), batches) for it in res]
    wall = R["tracker"].now()
    stop.set()
    done.sort(key=lambda it: it["i"])

    base_top1 = new_top1 = 0
    records = []
    for it in done:
        gold, skus = it["gold"], [p["sku"] for p in it["shortlist"]]
        before = skus.index(gold) + 1 if gold in skus else None
        order = [k for k, _ in it["ranked"]]
        ok = order[0] == (gold if gold in skus else NONE)
        base_top1 += before == 1
        new_top1 += ok
        nulls = [f for ps in it["passes"] for f, v in ps["fields"].items() if v["nulled"]]
        if it["i"] < len(EMAILS):
            print(f"\n#{it['i'] + 1} {it['email']}\n  extracted: {describe(it['spec'])}" + (f"   nulled (low conf): {', '.join(nulls)}" if nulls else ""))
            print(f"  => {'RIGHT' if ok else 'WRONG'}: top pick {order[0]} ({it['ranked'][0][1]:.2f}); gold {gold or 'none exists'}, search rank {before}")
        records.append({"i": it["i"] + 1, "email": it["email"], "gold": gold, "search_rank": before, "ok": ok, "spec": it["spec"],
                        "passes": it["passes"], "shortlist": [p["sku"] + "|" + p["desc"] for p in it["shortlist"]],
                        "ranked": it["ranked"], "rerank_conf": it["rconf"], "batch": it["bid"], "nulls": nulls})

    n = len(items)
    tokens = sum(it["tokens"] for it in done)
    card = scorecard(R, done)
    for rec in records:
        rec["oracle_top"] = ORACLE_TOP.get(rec["i"] - 1)
    print("\n" + card)
    print(f"{n} products in {wall:.2f}s wall = {wall / n * 1000:.0f} ms/product, {len(R['tracker'].batches)} batch(es), "
          f"~{int(tokens):,} input tokens (~${tokens * 0.042 / 1e6:.5f}), estimator chars/token now {CPT:.2f}")
    print("SUMMARY " + json.dumps({"budget": BUDGET, "workers": WORKERS, "n": n, "wall": wall, "products_per_s": n / wall, "tok_per_s": tokens / wall,
                                    "right": sum(r["ok"] for r in records) / n, "batches": len(R["tracker"].batches),
                                    "min_conf": CONF_MIN, "answered_acc": next(r for r in CURVE_ROWS if r["t"] == CONF_MIN)["all"]["acc"],
                                    "coverage": next(r for r in CURVE_ROWS if r["t"] == CONF_MIN)["all"]["cov"],
                                    "utility": next(r for r in CURVE_ROWS if r["t"] == CONF_MIN)["all"]["util"], "wrong_cost": WRONG_COST,
                                    "best_conf_hm": max((r for r in CURVE_ROWS if r["t"] >= CONF_MIN), key=lambda r: (r["all"]["hm"], -r["t"]))["t"],
                                    "rate_limited": STATS["rate_limited"], "max_retry_after_s": max(STATS["retry_after_s"], default=0), "sdk_max_retries": MAX_RETRIES}))
    if not NO_REPORT:
        write_report(records, R["tracker"], card, n, wall)



# ---------------------------------------------------------------------- x-ray report
CSS = """
:root{--bg:#fff;--fg:#1a1d21;--mut:#6b7280;--line:#e5e7eb;--card:#f8f9fb;--acc:#2f6fed;--ok:#1a7f4b;--bad:#c2410c;--bar:#9db8f5;--warn:#b45309}
@media(prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8eb;--mut:#9aa1ab;--line:#2a2e35;--card:#1b1e23;--acc:#7aa2ff;--ok:#4ade80;--bad:#fb923c;--bar:#35548f;--warn:#fbbf24}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px}h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);margin-bottom:18px}
details{border:1px solid var(--line);border-radius:8px;margin:8px 0;background:var(--card)}
summary{cursor:pointer;padding:10px 12px;display:flex;gap:10px;align-items:baseline;list-style:none}
summary::-webkit-details-marker{display:none}summary:before{content:"▸";color:var(--mut)}details[open]>summary:before{content:"▾"}
.n{font-weight:600;min-width:28px}.em{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tag{font-size:12px;padding:1px 8px;border-radius:99px;border:1px solid currentColor;white-space:nowrap}.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.body{padding:4px 14px 14px}h3{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);margin:16px 0 6px}
.email{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px 10px}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line);vertical-align:middle}
th{color:var(--mut);font-weight:500}code{font:12px ui-monospace,monospace}
.bar{position:relative;background:var(--bg);border:1px solid var(--line);border-radius:3px;height:16px;min-width:120px}
.bar i{position:absolute;inset:0 auto 0 0;background:var(--bar);border-radius:2px}.bar b{position:relative;font:11px ui-monospace,monospace;padding:0 5px;line-height:15px}
tr.pick td{font-weight:600}tr.gold td:first-child{box-shadow:inset 3px 0 var(--ok)}.dim{color:var(--mut)}
.track{border:1px solid var(--line);border-radius:10px;background:var(--card);padding:14px 16px;margin-bottom:18px}
.thead{display:flex;gap:12px;align-items:center;margin-bottom:10px}.thead b{font-size:15px}.clock{font:600 20px ui-monospace,monospace;margin-left:auto}
.thead select,.thead button{font:inherit;padding:3px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);cursor:pointer}
.brow{display:flex;align-items:center;gap:10px;margin:8px 0}.blabel{width:120px;font-size:12px;color:var(--mut)}
.steps{flex:1;display:flex;align-items:flex-start}.step{flex:1;text-align:center;position:relative;font-size:12px}
.step:not(:first-child):before{content:"";position:absolute;top:9px;right:50%;width:100%;height:3px;background:var(--line)}
.step.done:not(:first-child):before,.step.active:not(:first-child):before{background:var(--acc)}
.dot{width:20px;height:20px;border-radius:50%;margin:0 auto 3px;border:2px solid var(--line);background:var(--bg);position:relative;z-index:1;font-size:11px;line-height:16px}
.step.done .dot{background:var(--ok);border-color:var(--ok);color:#fff}.step.active .dot{border-color:var(--acc);animation:p 1s infinite}
@keyframes p{50%{box-shadow:0 0 0 6px color-mix(in srgb,var(--acc) 25%,transparent)}}
.step .t{font:11px ui-monospace,monospace;color:var(--mut);min-height:14px}.step.done .t{color:var(--ok)}
"""

JS = """
const D=JSON.parse(document.getElementById('tl').textContent);
const ST=D.stages, total=D.wall; let raf=null;
function render(t){
  D.batches.forEach(b=>{
    let allDone=true;
    ST.forEach(s=>{
      const el=document.getElementById(`s-${b.id}-${s}`), x=b.stages[s]; if(!el||!x){return}
      const st=t>=x.t1?'done':(t>=x.t0?'active':'pending'); if(st!=='done')allDone=false;
      el.className='step '+st; el.querySelector('.dot').textContent=st==='done'?'✓':'';
      el.querySelector('.t').textContent=st==='done'?(x.t1-x.t0).toFixed(2)+'s':(st==='active'?'…':'');
    });
    const d=document.getElementById(`s-${b.id}-done`), last=b.stages.rerank||b.stages[ST[ST.length-1]];
    if(d){d.className='step '+(allDone?'done':'pending'); d.querySelector('.dot').textContent=allDone?'✓':''; d.querySelector('.t').textContent=allDone?last.t1.toFixed(2)+'s total':'';}
  });
  document.getElementById('clock').textContent=Math.min(t,total).toFixed(2)+'s';
}
function play(){cancelAnimationFrame(raf);const sp=parseFloat(document.getElementById('spd').value),s0=performance.now();
  (function tick(n){const t=(n-s0)/1000*sp;render(t);if(t<total+.2)raf=requestAnimationFrame(tick);else render(total+1)})(s0)}
document.getElementById('rp').onclick=play;document.getElementById('spd').onchange=play;play();
"""


def bar(p: float) -> str:
    return f'<div class="bar"><i style="width:{max(p, 0) * 100:.1f}%"></i><b>{p:.3f}</b></div>'


def write_report(records: list[dict], T: Tracker, card: str, n: int, wall: float) -> None:
    import html
    from pathlib import Path

    e = html.escape
    tl = {"stages": Tracker.STAGES, "wall": wall, "batches": [
        {"id": b["id"], "n": b["n"], "stages": {k: {"t0": v["t0"], "t1": v["t1"] or wall} for k, v in b["stages"].items()}}
        for b in T.batches.values() if not b["split"]]}
    parts = [f"<!doctype html><meta charset=utf-8><title>Jev x-ray</title><style>{CSS}</style><main>",
             f"<h1>Jev pipeline x-ray</h1><div class=sub>model {MODEL} · {n} products in {wall:.2f}s ({wall / n * 1000:.0f} ms/product) · "
             f"fields under confidence {CONF_MIN} are nulled</div><pre class=email style='overflow:auto;margin-bottom:18px'>{html.escape(card)}</pre>",
             '<div class=track><div class=thead><b>Stage tracker</b><span class=dim>replay of the real recorded timings</span>'
             '<select id=spd><option value=1>real time</option><option value=0.25 selected>4× slower</option><option value=0.1>10× slower</option></select>'
             '<button id=rp>↻ replay</button><span class=clock id=clock>0.00s</span></div>']
    for b in tl["batches"]:
        steps = "".join(f'<div class="step pending" id="s-{b["id"]}-{k}"><div class=dot></div>{Tracker.LABELS[k].replace(" · ", "<br>") if "·" in Tracker.LABELS[k] else Tracker.LABELS[k]}<div class=t></div></div>'
                        for k in Tracker.STAGES if k in b["stages"])
        steps += f'<div class="step pending" id="s-{b["id"]}-done"><div class=dot></div>done<div class=t></div></div>'
        parts.append(f'<div class=brow><div class=blabel>batch {b["id"]}<br>{b["n"]} products</div><div class=steps>{steps}</div></div>')
    parts.append("</div>")
    for r in records:
        top = r["ranked"][0][0]
        tag = f'<span class="tag {"ok" if r["ok"] else "bad"}">{"RIGHT" if r["ok"] else "WRONG"}</span>'
        nul = f'<span class="tag warn">{len(r["nulls"])} null</span>' if r["nulls"] else ""
        ranks = f'search #{r["search_rank"]}' if r["search_rank"] else "gold not in shortlist"
        parts.append(f'<details><summary><span class=n>#{r["i"]}</span><span class=em>{e(r["email"])}</span>'
                     f'<span class=dim>{ranks} → <code>{e(top)}</code></span>{nul}{tag}</summary><div class=body>')
        parts.append(f'<h3>Email · rode in batch {r["batch"]}</h3><div class=email>{e(r["email"])}</div>')
        for ps in r["passes"]:
            parts.append(f"<h3>Extraction · {e(ps['name'])}</h3>")
            for field, raw in ps["fields"].items():
                items = sorted(raw["probs"].items(), key=lambda kv: -kv[1])
                rows = "".join(f'<tr class="{"pick" if k == raw["choice"] else ""}"><td><code>{e(k)}</code></td><td>{bar(p)}</td></tr>' for k, p in items[:5])
                more = f"top 5 of {len(items)} options offered" if len(items) > 5 else f"{len(items)} options offered"
                badge = f' <span class="tag warn">NULL: conf {raw["conf"]:.2f} &lt; {CONF_MIN}</span>' if raw["nulled"] else ""
                parts.append(f'<table><tr><th colspan=2>{field} <span class=dim>· picked <code>{e(raw["choice"])}</code> · conf {raw["conf"]:.2f} · {more}</span>{badge}</th></tr>{rows}</table>')
        parts.append(f'<h3>Normalized by code (null = unknown)</h3><div class=email><code>{e(json.dumps(r["spec"]))}</code></div>')
        probs = dict(r["ranked"])
        parts.append(f'<h3>Shortlist from search (10 candidates → Jev options) · rerank conf {r["rerank_conf"]:.2f}</h3>'
                     "<table><tr><th>search</th><th>candidate</th><th>rerank prob</th></tr>")
        for rank, item in enumerate(r["shortlist"], 1):
            sku, desc = item.split("|", 1)
            parts.append(f'<tr class="{"gold" if sku == r["gold"] else ""}{" pick" if sku == top else ""}"><td>#{rank}</td><td><code>{e(sku)}</code> <span class=dim>{e(desc)}</span></td><td>{bar(probs.get(sku, 0))}</td></tr>')
        parts.append(f'<tr class="{"pick" if top == NONE else ""}"><td>–</td><td><code>none</code> <span class=dim>escape hatch</span></td><td>{bar(probs.get(NONE, 0))}</td></tr></table>')
        parts.append(f'<div class=dim style="margin-top:6px">Gold: <code>{e(r["gold"] or "none of these")}</code>. Green edge = gold; bold = Jev top pick.'
                     f'{" Gold was not in the shortlist, so it could not be picked." if r["gold"] and not r["search_rank"] else ""}</div></div></details>')
    parts.append(f'</main><script type="application/json" id="tl">{json.dumps(tl)}</script><script>{JS}</script>')
    Path("report.html").write_text("".join(parts))
    Path("results.json").write_text(json.dumps(records, indent=1, default=str))
    print("\nx-ray report: report.html (raw data: results.json)")


if __name__ == "__main__":
    main()
