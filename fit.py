"""Regress end-to-end correctness on products per call and its confounders, with a separate hinge (knot) per variable.

Nickname: "spindle regression". Standard names for this model: additive piecewise-linear (linear-spline) GAM; a single
variable's version is segmented / broken-stick regression; MARS (Friedman 1991) is the greedy-selection relative that uses
the same hinge terms max(0, x - knot). Here each variable gets its own knot (or none), chosen by cross-validation.

    .venv/bin/python collect.py                 # first: gather per-product records into runs/
    .venv/bin/python fit.py                     # needs numpy (requirements-analysis.txt)

Outcome: y = 1 if the rerank top-1 is right (gold if shortlisted, else none). Unit: one product in one run.
Variables (all observed):
    B             products in the call the product rode in            (design variable, we choose it via --budget)
    min_conf      null threshold of the run                            (design variable, 3 levels -> factor, no knot)
    pos           position of the product within its call
    email         which of the 10 sample emails (fixed effects absorb per-email difficulty)
    search_miss / no_stock   correct answer is not shortlisted / not stocked (email-level; used instead of FE)
    n_null        number of fields nulled for the product              (MEDIATOR: a consequence of B and min_conf)
Models are linear probability models (coefficients in percentage points) with standard errors clustered by call, since
products in the same call share fate. Knots are chosen per variable by leave-runs-out cross-validation, alternating over
variables; a hinge is kept only if it lowers CV error. Post-selection standard errors are optimistic: read knots as
"where the data bend", not as tested hypotheses. There is deliberately no p-value-driven stepwise selection.
"""

import collections
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(runs_dir: str) -> dict[str, np.ndarray]:
    rows = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        d = json.load(open(path))
        by_batch = collections.defaultdict(list)
        for r in d["records"]:
            by_batch[r["batch"]].append(r)
        for bid, rs in by_batch.items():
            for pos, r in enumerate(sorted(rs, key=lambda r: r["i"])):
                rows["y"].append(1.0 if r["ok"] else 0.0)
                rows["run"].append(os.path.basename(path))
                rows["cluster"].append(f"{os.path.basename(path)}:{bid}")
                rows["B"].append(len(rs))
                rows["pos"].append(pos)
                rows["min_conf"].append(d["min_conf"])
                rows["n_null"].append(len(r["nulls"]))
                rows["email"].append((r["i"] - 1) % 10)
                rows["search_miss"].append(int(r["gold"] is not None and r["search_rank"] is None))
                rows["no_stock"].append(int(r["gold"] is None))
    D = {k: np.array(v) for k, v in rows.items()}
    D["_mc_levels"] = np.array(sorted(set(D["min_conf"])))  # fixed factor levels, so counterfactual rows keep the same columns
    return D


def hinge(x, knot):
    return np.maximum(0.0, x - knot)


def design(D, spec, knots):
    """Columns for a spec: a list of terms. Returns (X, names)."""
    cols, names = [np.ones(len(D["y"]))], ["const"]
    for term in spec:
        if term == "B":
            cols.append(D["B"].astype(float)); names.append("B")
            if knots.get("B") is not None:
                cols.append(hinge(D["B"], knots["B"])); names.append(f"B_hinge@{knots['B']:g}")
        elif term == "pos":
            cols.append(D["pos"].astype(float)); names.append("pos")
            if knots.get("pos") is not None:
                cols.append(hinge(D["pos"], knots["pos"])); names.append(f"pos_hinge@{knots['pos']:g}")
        elif term == "n_null":
            cols.append(D["n_null"].astype(float)); names.append("n_null")
            if knots.get("n_null") is not None:
                cols.append(hinge(D["n_null"], knots["n_null"])); names.append(f"n_null_hinge@{knots['n_null']:g}")
        elif term == "min_conf":  # factor: the design only has a few levels
            for lv in D["_mc_levels"][1:]:
                cols.append((D["min_conf"] == lv).astype(float)); names.append(f"min_conf={lv:g}")
        elif term == "email_fe":
            for e in range(1, 10):
                cols.append((D["email"] == e).astype(float)); names.append(f"email{e}")
        elif term == "email_feat":
            cols += [D["search_miss"].astype(float), D["no_stock"].astype(float)]; names += ["search_miss", "no_stock"]
        elif term == "B_x_none":  # B x (correct answer is none): a difference-in-differences style interaction
            none = np.maximum(D["search_miss"], D["no_stock"]).astype(float)
            cols.append(D["B"] * none); names.append("B_x_answer_none")
    return np.column_stack(cols), names


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def cluster_se(X, y, beta, clusters):
    u = y - X @ beta
    _, inv = np.unique(clusters, return_inverse=True)
    G, N, K = inv.max() + 1, len(y), X.shape[1]
    XtX_inv = np.linalg.pinv(X.T @ X)
    scores = np.zeros((G, K))
    np.add.at(scores, inv, X * u[:, None])
    V = XtX_inv @ (scores.T @ scores) @ XtX_inv * (G / (G - 1)) * ((N - 1) / (N - K))
    return np.sqrt(np.maximum(np.diag(V), 0)), G


def cv_rmse(D, spec, knots, folds=5):
    """Leave-runs-out CV: whole runs are held out, so a candidate knot cannot exploit run-specific noise."""
    runs = np.unique(D["run"]); rng = np.random.default_rng(0); rng.shuffle(runs)
    X, _ = design(D, spec, knots)
    err = np.zeros(len(D["y"]))
    for f in range(folds):
        te = np.isin(D["run"], runs[f::folds])
        beta = ols(X[~te], D["y"][~te])
        err[te] = D["y"][te] - X[te] @ beta
    return float(np.sqrt(np.mean(err ** 2)))


def choose_knots(D, spec, vars_):
    """Per-variable knot by CV, alternating (backfitting) for two passes. None = the linear term alone did as well or better."""
    grids = {v: np.unique(np.percentile(D[v], np.linspace(15, 90, 14)).round()) for v in vars_}
    knots = {v: None for v in vars_}
    for _ in range(2):
        for v in vars_:
            best, best_k = cv_rmse(D, spec, {**knots, v: None}), None
            for k in grids[v]:
                if D[v].min() < k < D[v].max():
                    s = cv_rmse(D, spec, {**knots, v: float(k)})
                    if s < best - 1e-5:
                        best, best_k = s, float(k)
            knots[v] = best_k
    return knots


def report(D, name, spec, knots, show):
    X, names = design(D, spec, knots)
    y = D["y"]
    beta = ols(X, y)
    se, G = cluster_se(X, y, beta, D["cluster"])
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"\n{name}\n   R2 {r2:.3f}   CV-RMSE {cv_rmse(D, spec, knots):.4f}   N {len(y):,}   calls {G:,}")
    for nm, b, s in zip(names, beta, se):
        if any(nm.startswith(p) for p in show):
            print(f"   {nm:<18} {100 * b:>+8.2f} pts  (se {100 * s:.2f})")
    return dict(zip(names, beta)), X, names


def main() -> None:
    D = load(os.path.join(HERE, "runs"))
    if not len(D.get("y", [])):
        raise SystemExit("No runs found: run collect.py first.")
    y = D["y"]
    print(f"{len(y):,} product observations from {len(np.unique(D['run']))} runs, "
          f"B in [{D['B'].min()}, {D['B'].max()}], min_conf levels {sorted(set(D['min_conf']))}, overall RIGHT {y.mean():.1%}")

    show_core = ("B", "min_conf", "n_null", "pos", "search", "no_stock")
    report(D, "1. B only (linear)", ["B"], {}, ("B",))
    report(D, "2. + min_conf (factor)", ["B", "min_conf"], {}, show_core)
    report(D, "3. + email-level difficulty (search_miss, no_stock)", ["B", "min_conf", "email_feat"], {}, show_core)
    report(D, "4. email fixed effects instead", ["B", "min_conf", "email_fe"], {}, show_core)
    report(D, "5. + position in call (linear)", ["B", "min_conf", "email_fe", "pos"], {}, show_core)

    base = ["B", "min_conf", "email_fe", "pos"]
    knots = choose_knots(D, base, ["B", "pos"])
    lin = cv_rmse(D, base, {})
    print(f"\nPer-variable knots chosen by leave-runs-out CV: {knots}  (CV-RMSE linear {lin:.4f} -> {cv_rmse(D, base, knots):.4f})")
    beta, X, names = report(D, "6. TOTAL effect, each variable with its own hinge if CV supports it", base, knots, show_core + ("B_hinge", "pos_hinge"))
    if knots["B"] is not None:
        lo, hi = 100 * beta["B"], 100 * (beta["B"] + beta[f"B_hinge@{knots['B']:g}"])
        print(f"   => slope in B: {lo:+.2f} pts/product below B={knots['B']:g}, {hi:+.2f} above")

    n_null_knots = choose_knots(D, base + ["n_null"], ["n_null"])
    mk = {**knots, **n_null_knots}
    report(D, "7. DIRECT effect: adds n_null (a mediator, so B's coefficient shrinks if nulls carry its effect)",
           base + ["n_null"], mk, show_core + ("B_hinge", "pos_hinge", "n_null_hinge"))
    report(D, "8. DID-style: does B hurt products whose right answer is 'none' differently?", base + ["B_x_none"], knots, show_core + ("B_hinge", "pos_hinge"))

    # descriptive check behind the regression: raw RIGHT by position in the call
    print("\nRaw RIGHT by position in the call (all runs pooled; only calls large enough to have that position contribute):")
    for lo, hi in ((0, 4), (5, 9), (10, 13), (14, 17), (18, 24), (25, 40)):
        m = (D["pos"] >= lo) & (D["pos"] <= hi)
        if m.sum():
            print(f"   positions {lo:>2}-{hi:<2}  RIGHT {y[m].mean():6.1%}   n={int(m.sum()):,}")

    # counterfactual effect sizes from model 6: overwrite B and min_conf for every row, hold position at 0, average over emails
    Xa, _ = design(D, base, knots)
    b6 = ols(Xa, y)

    def avg_pred(B, mc, pos=0):
        n = len(y)
        D2 = {**D, "B": np.full(n, B), "min_conf": np.full(n, mc), "pos": np.full(n, pos)}
        return float(np.mean(design(D2, base, knots)[0] @ b6))

    b_lo, b_hi = int(np.percentile(D["B"], 5)), int(D["B"].max())
    print("\nEffect sizes on the observed ranges (model 6, position 0, averaged over emails):")
    print(f"   products per call {b_lo} -> {b_hi} at position 0: {100 * (avg_pred(b_hi, 0.0) - avg_pred(b_lo, 0.0)):+.1f} pts")
    for lv in sorted(set(D["min_conf"]))[1:]:
        print(f"   min_conf 0 -> {lv:g}: {100 * (avg_pred(b_lo, lv) - avg_pred(b_lo, 0.0)):+.1f} pts at B={b_lo}, "
              f"{100 * (avg_pred(b_hi, lv) - avg_pred(b_hi, 0.0)):+.1f} pts at B={b_hi}")
    for p in (10, 20, min(30, b_hi - 1)):
        print(f"   position 0 -> {p} within a call (B={b_hi}, min_conf 0): {100 * (avg_pred(b_hi, 0.0, p) - avg_pred(b_hi, 0.0, 0)):+.1f} pts")


if __name__ == "__main__":
    main()
