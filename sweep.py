"""Sweep the per-call token budget and the null threshold, then rank settings.

    export TYPESAFE_API_KEY=...
    python sweep.py                                            # budgets 3k-48k at --min-conf 0.6, 2 reps each (~$0.4)
    python sweep.py --budgets 12000 --min-confs 0,0.5,0.6,0.7,0.8,0.9 --reps 3   # tune --min-conf at one batch size
    python sweep.py --budgets 12000,48000 --min-confs 0,0.6 --wrong-cost 3       # both axes; wrong field costs 3
    python sweep.py --extra "--copies 30"                      # pass any demo.py flags through

Each run is one demo.py run. Columns:
    RIGHT      end-to-end: rerank top-1 correct (gold if shortlisted, else none)
    acc / cov  extraction accuracy on ANSWERED fields / share of fields answered (nulls are abstentions, not errors)
    HM t/p     harmonic mean of RIGHT and throughput (normalized to the best setting seen)
    HM a/c     harmonic mean of answered accuracy and coverage
    util       (+1 right, -wrong_cost wrong, 0 abstained) per applicable field, from the offline curve at that threshold
Writes sweep_results.jsonl (one demo.py SUMMARY per run).
"""

import argparse
import collections
import json
import os
import subprocess
import sys


def hm(a: float, b: float) -> float:
    return 2 * a * b / (a + b) if a + b else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budgets", default="3000,6000,9000,12000,18000,24000,36000,48000")
    ap.add_argument("--min-confs", default="0.6", help="null thresholds to sweep, comma separated")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--copies", type=int, default=30, help="copies of the 10 sample emails per run")
    ap.add_argument("--wrong-cost", type=float, default=1.0, help="cost of a wrong field relative to an abstention, for the util column")
    ap.add_argument("--extra", default="", help="extra flags for demo.py")
    a = ap.parse_args()
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("Set TYPESAFE_API_KEY first.")

    runs = collections.defaultdict(list)
    with open("sweep_results.jsonl", "w") as out:
        for b in (int(x) for x in a.budgets.split(",")):
            for mc in (float(x) for x in a.min_confs.split(",")):
                for _ in range(a.reps):
                    cmd = [sys.executable, "-u", "demo.py", "--copies", str(a.copies), "--budget", str(b), "--min-conf", str(mc),
                           "--wrong-cost", str(a.wrong_cost), "--quiet", "--no-report", "--no-oracle", *a.extra.split()]
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    line = next((ln for ln in res.stdout.splitlines() if ln.startswith("SUMMARY ")), None)
                    if not line:
                        sys.exit(f"budget {b}, min-conf {mc}: no SUMMARY line.\n{res.stdout[-800:]}\n{res.stderr[-800:]}")
                    rec = json.loads(line[len("SUMMARY "):])
                    runs[(b, mc)].append(rec)
                    out.write(json.dumps(rec) + "\n")

    keys = ("products_per_s", "tok_per_s", "right", "batches", "workers", "answered_acc", "coverage", "utility")
    agg = {k: {f: sum(r[f] for r in v) / len(v) for f in keys} for k, v in runs.items()}
    best = max(x["products_per_s"] for x in agg.values())
    print(f"{'budget':>7} {'min-conf':>8} {'per call':>8} {'prod/s':>7} {'RIGHT':>6} {'acc':>6} {'cov':>6} {'HM t/p':>7} {'HM a/c':>7} {'util':>6}")
    for (b, mc), x in sorted(agg.items()):
        n = runs[(b, mc)][0]["n"]
        print(f"{b:>7} {mc:>8g} {n / x['batches']:>8.0f} {x['products_per_s']:>7.1f} {x['right']:>6.1%} {x['answered_acc']:>6.1%} {x['coverage']:>6.1%} "
              f"{hm(x['right'], x['products_per_s'] / best):>7.3f} {hm(x['answered_acc'], x['coverage']):>7.3f} {x['utility']:>6.3f}")


if __name__ == "__main__":
    main()
