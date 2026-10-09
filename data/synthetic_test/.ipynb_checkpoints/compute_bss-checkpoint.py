#!/usr/bin/env python3
"""
Reference Brier Skill Score (BSS) calculation for tercile probability forecasts.

Usage
-----
    python compute_bss.py                     # scores every case*.csv in this folder
    python compute_bss.py my_forecasts.csv    # scores one or more named files
    python compute_bss.py --diagnose          # also shows common wrong ways of computing BSS

Input CSV columns (header row required):
    p_below, p_normal, p_above   forecast probabilities (each row sums to 1)
    obs                          observed tercile: below / normal / above
Other columns (year, o_below, ...) are ignored; the one-hot outcome is rebuilt from `obs`.

If answer_key.csv is in the same folder, each result is checked against it.

Definitions
-----------
For each category k (below, normal, above) and each year t:
    p[t,k] = forecast probability,  o[t,k] = 1 if category k was observed, else 0

Per-category Brier score:      BS_k     = mean over t of (p[t,k] - o[t,k])^2
Per-category reference score:  BSref_k  = mean over t of (r_k - o[t,k])^2
Per-category skill:            BSS_k    = 1 - BS_k / BSref_k

Multi-category Brier score:    BS       = BS_below + BS_normal + BS_above
Multi-category reference:      BSref    = BSref_below + BSref_normal + BSref_above
Multi-category skill:          BSS      = 1 - BS / BSref

Reference forecast r_k:
    fixed  : 1/3 for every category (standard for tercile forecasts)
    sample : observed relative frequency of category k in the file

Key rule: average the squared errors over years FIRST, then form the ratio ONCE.
Never average per-year skill scores, and never average the three BSS_k to get BSS.
"""
import csv
import glob
import os
import sys

import numpy as np

CATS = ["below", "normal", "above"]
HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    """Return forecast probabilities P (n x 3) and one-hot outcomes O (n x 3)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    P = np.array([[float(r["p_" + c]) for c in CATS] for r in rows])
    obs = [r["obs"].strip().lower() for r in rows]
    bad = sorted(set(obs) - set(CATS))
    if bad:
        raise ValueError(f"{path}: unrecognised obs values {bad}")
    O = np.array([[1.0 if o == c else 0.0 for c in CATS] for o in obs])
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{path}: some forecast rows do not sum to 1")
    return P, O


def brier_skill(P, O, reference="fixed"):
    """Correct BSS: per-category and multi-category, for one reference forecast."""
    if reference == "fixed":
        R = np.full(3, 1.0 / 3.0)
    elif reference == "sample":
        R = O.mean(axis=0)
    else:
        raise ValueError("reference must be 'fixed' or 'sample'")

    bs_k = ((P - O) ** 2).mean(axis=0)          # mean over years, per category
    ref_k = ((R - O) ** 2).mean(axis=0)         # reference broadcast over years

    with np.errstate(divide="ignore", invalid="ignore"):
        bss_k = np.where(ref_k > 0, 1.0 - bs_k / ref_k, np.nan)

    bs = bs_k.sum()
    ref = ref_k.sum()
    return {
        "BS": bs,
        "BSref": ref,
        "BSS": 1.0 - bs / ref,                  # ratio of sums, not mean of ratios
        "BS_k": bs_k,
        "BSref_k": ref_k,
        "BSS_k": bss_k,
    }


def wrong_ways(P, O):
    """Common mistakes, so you can see which one your code is making."""
    R = np.full(3, 1.0 / 3.0)
    bs_k = ((P - O) ** 2).mean(axis=0)
    ref_k = ((R - O) ** 2).mean(axis=0)
    bss_k = 1.0 - bs_k / ref_k

    per_year_cat = 1.0 - (P - O) ** 2 / (R - O) ** 2           # n x 3
    per_year_multi = 1.0 - ((P - O) ** 2).sum(1) / ((R - O) ** 2).sum(1)

    return {
        "mean of the 3 per-category BSS": bss_k.mean(),
        "per-year BSS averaged (above)": per_year_cat[:, 2].mean(),
        "per-year multi-cat BSS averaged": per_year_multi.mean(),
        "sample-frequency reference": brier_skill(P, O, "sample")["BSS"],
        "BS/K (averaged not summed over k)": 1.0 - (bs_k.mean()) / (ref_k.mean()),
        "reference = 0.5 instead of 1/3": 1.0 - bs_k.sum() / ((0.5 - O) ** 2).mean(0).sum(),
        "BS summed over years, not averaged": 1.0 - ((P - O) ** 2).sum() / ref_k.sum(),
    }


def load_key():
    path = os.path.join(HERE, "answer_key.csv")
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
        P, O = load(path)
        krow = key.get(name)
        print(f"\n=== {name}  (n = {len(P)}, counts below/normal/above = "
              f"{O.sum(0).astype(int).tolist()}) ===")

        for ref in ("fixed", "sample"):
            r = brier_skill(P, O, ref)
            lines = [
                (f"BS_multi", r["BS"]),
                (f"BSref_{ref}_multi", r["BSref"]),
                (f"BSS_{ref}_multi", r["BSS"]),
            ]
            for k, c in enumerate(CATS):
                lines.append((f"BSS_{ref}_{c}", r["BSS_k"][k]))
            print(f"  reference = {ref}")
            for label, v in lines:
                status = check(krow, label, v)
                n_fail += "FAIL" in status
                print(f"    {label:22s} {v: .6f}{status}")

        if diagnose:
            print("  common mistakes (fixed 1/3 reference) -> value they would give:")
            for label, v in wrong_ways(P, O).items():
                print(f"    {label:38s} {v: .6f}")

    if key:
        print("\nAll checks passed." if n_fail == 0 else f"\n{n_fail} check(s) failed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
