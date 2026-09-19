"""Run demo.py over a grid of token budgets x null thresholds and keep every run's per-product records for fit.py.

    export TYPESAFE_API_KEY=...
    .venv/bin/python collect.py                                   # 10 budgets x 3 thresholds x 2 reps x 150 products (~$0.9)
    .venv/bin/python collect.py --budgets 12000,24000 --min-confs 0,0.6 --reps 3 --copies 20
    .venv/bin/python collect.py --budgets 6000,9000,12000         # resumable: runs that already exist in runs/ are skipped

Each run writes runs/run_b<budget>_mc<min-conf>_r<rep>.json holding demo.py's SUMMARY and the per-product records
(the same data as results.json: specs, per-field confidences, shortlist, rerank distribution, batch id, correctness).
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budgets", default="6000,9000,12000,16000,19000,22000,25000,30000,36000,48000")
    ap.add_argument("--min-confs", default="0,0.6,0.9")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--copies", type=int, default=15, help="copies of the 10 sample emails per run")
    ap.add_argument("--out", default=os.path.join(HERE, "runs"))
    a = ap.parse_args()
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("Set TYPESAFE_API_KEY first.")
    os.makedirs(a.out, exist_ok=True)

    for b in (int(x) for x in a.budgets.split(",")):
        for mc in (float(x) for x in a.min_confs.split(",")):
            for rep in range(a.reps):
                path = os.path.join(a.out, f"run_b{b}_mc{mc:g}_r{rep}.json")
                if os.path.exists(path):
                    continue
                cmd = [sys.executable, "-u", os.path.join(HERE, "demo.py"), "--copies", str(a.copies), "--budget", str(b),
                       "--min-conf", str(mc), "--quiet", "--no-oracle"]
                res = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
                line = next((ln for ln in res.stdout.splitlines() if ln.startswith("SUMMARY ")), None)
                if not line:
                    print(f"FAILED budget={b} min-conf={mc} rep={rep}\n{res.stdout[-300:]}\n{res.stderr[-300:]}", flush=True)
                    continue
                records = json.load(open(os.path.join(HERE, "results.json")))
                with open(path, "w") as f:
                    json.dump({"summary": json.loads(line[len("SUMMARY "):]), "budget": b, "min_conf": mc, "rep": rep, "records": records}, f)
                print(f"ok budget={b} min-conf={mc:g} rep={rep}", flush=True)


if __name__ == "__main__":
    main()
