"""Run demo.py over a grid and keep every run's per-product records for fit.py.

    export TYPESAFE_API_KEY=...
    # designed experiment: 300 generated emails, batch size assigned directly, random product order, random run order
    .venv/bin/python collect.py --gen 300 --sizes 4,8,12,16,20,24,28,32 --min-confs 0,0.6 --reps 2 --shuffle
    # older design: token budgets on the 10 fixed emails
    .venv/bin/python collect.py --budgets 6000,12000,24000 --min-confs 0,0.6,0.9 --reps 2 --copies 15

Design notes: with --sizes the batch size (products per call) is assigned exactly, not inferred from a token budget; with
--shuffle every run shuffles which products share a call and in which position (a fresh seed per run); the order in which
runs are executed is randomized (--order-seed) so time drift is not confounded with the grid. Runs already present in
runs/ are skipped, so the script is resumable. Each run writes runs/run_<...>.json with demo.py's SUMMARY and the
per-product records (specs, per-field confidences, position, call size, truth, correctness).
"""

import argparse
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="", help="products per call to assign directly (comma separated); overrides --budgets")
    ap.add_argument("--budgets", default="6000,9000,12000,16000,19000,22000,25000,30000,36000,48000")
    ap.add_argument("--min-confs", default="0,0.6,0.9")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--copies", type=int, default=15, help="copies of the emails per run")
    ap.add_argument("--gen", type=int, default=0, help="generate this many distinct emails instead of the 10 fixed ones")
    ap.add_argument("--gen-seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true", help="randomize product order within every run (fresh seed per run)")
    ap.add_argument("--order-seed", type=int, default=0, help="seed for the randomized order in which runs are executed (-1 = grid order)")
    ap.add_argument("--out", default=os.path.join(HERE, "runs"))
    a = ap.parse_args()
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("Set TYPESAFE_API_KEY first.")
    os.makedirs(a.out, exist_ok=True)

    sizes = [int(x) for x in a.sizes.split(",")] if a.sizes else []
    levels = [("pc", n) for n in sizes] if sizes else [("b", int(x)) for x in a.budgets.split(",")]
    grid = [(kind, v, float(mc), rep) for kind, v in levels for mc in a.min_confs.split(",") for rep in range(a.reps)]
    if a.order_seed >= 0:
        random.Random(a.order_seed).shuffle(grid)

    for idx, (kind, v, mc, rep) in enumerate(grid):
        tag = f"g{a.gen}s{a.gen_seed}_" if a.gen else ""
        path = os.path.join(a.out, f"run_{tag}{kind}{v}_mc{mc:g}_r{rep}.json")
        if os.path.exists(path):
            continue
        cmd = [sys.executable, "-u", os.path.join(HERE, "demo.py"), "--copies", str(a.copies), "--min-conf", str(mc), "--quiet", "--no-oracle"]
        cmd += ["--per-call", str(v)] if kind == "pc" else ["--budget", str(v)]
        if a.gen:
            cmd += ["--gen", str(a.gen), "--gen-seed", str(a.gen_seed)]
        if a.shuffle:
            cmd += ["--shuffle", str(10_000 * (rep + 1) + idx)]
        res = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        line = next((ln for ln in res.stdout.splitlines() if ln.startswith("SUMMARY ")), None)
        if not line:
            print(f"FAILED {kind}={v} min-conf={mc} rep={rep}\n{res.stdout[-300:]}\n{res.stderr[-300:]}", flush=True)
            continue
        records = json.load(open(os.path.join(HERE, "results.json")))
        with open(path, "w") as f:
            json.dump({"summary": json.loads(line[len("SUMMARY "):]), "kind": kind, "value": v, "min_conf": mc, "rep": rep,
                       "gen": a.gen, "shuffled": a.shuffle, "records": records}, f)
        print(f"[{idx + 1}/{len(grid)}] ok {kind}={v} min-conf={mc:g} rep={rep}", flush=True)


if __name__ == "__main__":
    main()
