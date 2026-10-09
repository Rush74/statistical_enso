#!/usr/bin/env python3
"""
Reference LEPS skill calculation for tercile probability forecasts
(Bureau of Meteorology convention: Fawcett et al. 2005; Wang, "BoM's climate
forecast verification" slides).

Usage
-----
    python compute_leps.py                     # scores every case*.csv in this folder
    python compute_leps.py my_forecasts.csv    # scores one or more named files
    python compute_leps.py --diagnose          # also shows common wrong ways of computing it

Input CSV columns (header row required):
    p_below, p_normal, p_above   forecast probabilities p1, p2, p3 (each row sums to 1)
    obs                          observed tercile: below / normal / above (tercile 1 / 2 / 3)
Other columns are ignored.

If leps_answer_key.csv is in the same folder, each result is checked against it.

Definitions
-----------
Score for forecast i, given the observed tercile:
    tercile 1 observed:  s_i =  8/27 p1 - 1/27 p2 - 7/27 p3
    tercile 2 observed:  s_i = -1/27 p1 + 2/27 p2 - 1/27 p3
    tercile 3 observed:  s_i = -7/27 p1 - 1/27 p2 + 8/27 p3
i.e. s_i = sum_k p_k * M[k, observed], with the symmetric matrix M below.

Best (u_i) and worst (l_i) possible scores for that observation:
    tercile 1 or 3:  u_i = 8/27,  l_i = -7/27
    tercile 2:       u_i = 2/27,  l_i = -1/27

LEPS skill:
    sum(s) / sum(u)         if sum(s) >= 0
    sum(s) / |sum(l)|       if sum(s) <  0
Range -1 (worst possible) to 1 (perfect).  The slide writes the second denominator as
sum(l); because l_i < 0 that literal form would give a POSITIVE skill for bad forecasts,
so the magnitude is used here.

Key rule: sum s, u and l over all forecasts FIRST, then divide ONCE.
"""
import csv
import glob
import os
import sys

import numpy as np

CATS = ["below", "normal", "above"]
HERE = os.path.dirname(os.path.abspath(__file__))

# rows = forecast tercile (p1, p2, p3), columns = observed tercile
M = np.array([[8, -1, -7],
              [-1, 2, -1],
              [-7, -1, 8]]) / 27.0
U = M.max(axis=0)          # best possible score per observed tercile: 8/27, 2/27, 8/27
L = M.min(axis=0)          # worst possible score per observed tercile: -7/27, -1/27, -7/27


def load(path):
    """Return forecast probabilities P (n x 3) and observed tercile index j (n,)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    P = np.array([[float(r["p_" + c]) for c in CATS] for r in rows])
    obs = [r["obs"].strip().lower() for r in rows]
    bad = sorted(set(obs) - set(CATS))
    if bad:
        raise ValueError(f"{path}: unrecognised obs values {bad}")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{path}: some forecast rows do not sum to 1")
    return P, np.array([CATS.index(o) for o in obs])


def leps_skill(P, j):
    """Correct LEPS skill."""
    s = (P * M[:, j].T).sum(axis=1)       # s_i = sum_k p_k M[k, j_i]
    u = U[j]
    l = L[j]
    S = s.sum()
    denom = u.sum() if S >= 0 else abs(l.sum())
    return {"s": s, "mean_S": s.mean(), "sum_S": S, "sum_S_m": denom, "LEPS_skill": S / denom}


def wrong_ways(P, j):
    """Common mistakes, so you can see which one your code is making."""
    s = (P * M[:, j].T).sum(axis=1)
    u, l = U[j], L[j]
    S = s.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        per_year = np.where(s >= 0, s / u, s / np.abs(l))
    s_by_prob_only = P[np.arange(len(j)), j]           # just the probability on the observed tercile
    return {
        "slide taken literally: sum(s)/sum(l)": S / (u.sum() if S >= 0 else l.sum()),
        "always divide by sum(u)": S / u.sum(),
        "per-year skill, then averaged": per_year.mean(),
        "mean s on the 1/9 scale (3x too big)": 3 * s.mean(),
        "sum(s) reported instead of skill": S,
        "probability on observed tercile, mean": s_by_prob_only.mean(),
    }


def load_key():
    path = os.path.join(HERE, "leps_answer_key.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["dataset"]: r for r in csv.DictReader(f)}


def check(key_row, name, value, tol=1e-6):
    if key_row is None or not key_row.get(name):
        return ""
    expected = float(key_row[name])
    return "  PASS" if abs(value - expected) <= tol else f"  FAIL (expected {expected:.6f})"


def main(argv):
    diagnose = "--diagnose" in argv
    files = [a for a in argv if not a.startswith("--")]
    if not files:
        files = sorted(glob.glob(os.path.join(HERE, "case*.csv")))
    key = load_key()
    n_fail = 0

    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        P, j = load(path)
        krow = key.get(name)
        r = leps_skill(P, j)
        print(f"\n=== {name}  (n = {len(P)}, counts tercile 1/2/3 = "
              f"{np.bincount(j, minlength=3).tolist()}) ===")
        for label in ("mean_S", "sum_S", "sum_S_m", "LEPS_skill"):
            status = check(krow, label, r[label])
            n_fail += "FAIL" in status
            print(f"    {label:12s} {r[label]: .6f}{status}")

        if diagnose:
            print("  common mistakes -> value they would give:")
            for label, v in wrong_ways(P, j).items():
                print(f"    {label:40s} {v: .6f}")

    if key:
        print("\nAll checks passed." if n_fail == 0 else f"\n{n_fail} check(s) failed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
