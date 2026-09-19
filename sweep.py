"""Sweep the per-call token budget, then rank settings by harmonic mean of accuracy and throughput.

    export TYPESAFE_API_KEY=...
    python sweep.py                                   # budgets 3k-48k, 2 reps each, 300 products per run (~$0.4)
    python sweep.py --budgets 8000,12000,16000 --reps 3 --copies 20
    python sweep.py --extra "--min-conf 0.7"          # pass any demo.py flags through

Accuracy = end-to-end RIGHT rate; throughput normalized to the best setting seen in the sweep.
Writes sweep_results.jsonl (one demo.py SUMMARY per run).
"""

import argparse
import collections
import json
import os
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budgets", default="3000,6000,9000,12000,18000,24000,36000,48000")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--copies", type=int, default=30, help="copies of the 10 sample emails per run")
    ap.add_argument("--extra", default="", help="extra flags for demo.py")
    a = ap.parse_args()
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("Set TYPESAFE_API_KEY first.")

    runs = collections.defaultdict(list)
    with open("sweep_results.jsonl", "w") as out:
        for b in (int(x) for x in a.budgets.split(",")):
            for _ in range(a.reps):
                cmd = [sys.executable, "-u", "demo.py", "--copies", str(a.copies), "--budget", str(b),
                       "--quiet", "--no-report", "--no-oracle", *a.extra.split()]
                res = subprocess.run(cmd, capture_output=True, text=True)
                line = next((ln for ln in res.stdout.splitlines() if ln.startswith("SUMMARY ")), None)
                if not line:
                    sys.exit(f"budget {b}: no SUMMARY line.\n{res.stdout[-800:]}\n{res.stderr[-800:]}")
                rec = json.loads(line[len("SUMMARY "):])
                runs[b].append(rec)
                out.write(json.dumps(rec) + "\n")

    agg = {b: {k: sum(r[k] for r in v) / len(v) for k in ("products_per_s", "tok_per_s", "right", "batches", "workers")} for b, v in runs.items()}
    best = max(x["products_per_s"] for x in agg.values())
    print(f"{'budget':>7} {'per call':>8} {'in flight':>9} {'prod/s':>7} {'tok/s':>9} {'RIGHT':>6} {'tput norm':>9} {'HM':>6}")
    for b, x in sorted(agg.items()):
        n = runs[b][0]["n"]
        tn = x["products_per_s"] / best
        print(f"{b:>7} {n / x['batches']:>8.0f} {x['workers']:>9.0f} {x['products_per_s']:>7.1f} {x['tok_per_s']:>9,.0f} {x['right']:>6.1%} {tn:>9.2f} {2 * x['right'] * tn / (x['right'] + tn):>6.3f}")


if __name__ == "__main__":
    main()
