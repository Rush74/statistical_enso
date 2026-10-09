"""
Generate synthetic tercile-forecast datasets with exactly known Brier Skill Scores.

Each dataset CSV has columns:
    year, p_below, p_normal, p_above, obs   (obs in {below, normal, above})
plus one-hot columns o_below, o_normal, o_above for convenience.

Probabilities are multiples of 0.01 and sum to exactly 1.00, so all scores are
computed exactly with rational arithmetic (fractions.Fraction).

Definitions used in the answer key
----------------------------------
Per-category Brier score (category k):
    BS_k = (1/n) * sum_i (p_ik - o_ik)^2
Multi-category Brier score (sum over the 3 categories):
    BS = (1/n) * sum_i sum_k (p_ik - o_ik)^2  = BS_below + BS_normal + BS_above
Brier Skill Score:
    BSS = 1 - BS / BS_ref
Two reference forecasts are reported:
    'fixed'  : climatological probabilities 1/3, 1/3, 1/3
    'sample' : observed relative frequency of each category in the dataset
Bonus: Ranked Probability Score / Skill Score (cumulative over ordered categories,
summed over the 2 non-trivial cumulative thresholds, no normalisation), fixed 1/3 ref.
"""
from fractions import Fraction as F
import csv, random, math

CATS = ["below", "normal", "above"]


def scores(rows):
    n = len(rows)
    P = [[F(r["p"][k]) for k in range(3)] for r in rows]
    O = [[1 if r["obs"] == CATS[k] else 0 for k in range(3)] for r in rows]

    out = {"n": n}
    freq = [F(sum(O[i][k] for i in range(n)), n) for k in range(3)]
    out["n_below"], out["n_normal"], out["n_above"] = [sum(O[i][k] for i in range(n)) for k in range(3)]

    bs_k = [sum((P[i][k] - O[i][k]) ** 2 for i in range(n)) / n for k in range(3)]
    ref_fixed_k = [sum((F(1, 3) - O[i][k]) ** 2 for i in range(n)) / n for k in range(3)]
    ref_sample_k = [sum((freq[k] - O[i][k]) ** 2 for i in range(n)) / n for k in range(3)]

    for k, c in enumerate(CATS):
        out[f"BS_{c}"] = bs_k[k]
        out[f"BSref_fixed_{c}"] = ref_fixed_k[k]
        out[f"BSS_fixed_{c}"] = 1 - bs_k[k] / ref_fixed_k[k]
        out[f"BSref_sample_{c}"] = ref_sample_k[k]
        out[f"BSS_sample_{c}"] = (1 - bs_k[k] / ref_sample_k[k]) if ref_sample_k[k] != 0 else None

    bs = sum(bs_k)
    out["BS_multi"] = bs
    out["BSref_fixed_multi"] = sum(ref_fixed_k)
    out["BSS_fixed_multi"] = 1 - bs / sum(ref_fixed_k)
    out["BSref_sample_multi"] = sum(ref_sample_k)
    out["BSS_sample_multi"] = 1 - bs / sum(ref_sample_k)

    # RPS (bonus)
    def rps(p, o):
        cp = [p[0], p[0] + p[1]]
        co = [o[0], o[0] + o[1]]
        return sum((cp[j] - co[j]) ** 2 for j in range(2))
    clim = [F(1, 3)] * 3
    rps_f = sum(rps(P[i], O[i]) for i in range(n)) / n
    rps_r = sum(rps(clim, O[i]) for i in range(n)) / n
    out["RPS"] = rps_f
    out["RPSref_fixed"] = rps_r
    out["RPSS_fixed"] = 1 - rps_f / rps_r
    return out


def write_dataset(name, rows):
    with open(f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "p_below", "p_normal", "p_above", "obs", "o_below", "o_normal", "o_above"])
        for r in rows:
            o = [1 if r["obs"] == c else 0 for c in CATS]
            w.writerow([r["year"]] + [f"{float(F(x)):.2f}" for x in r["p"]] + [r["obs"]] + o)


def mk(year, p, obs):
    p = [F(x) for x in p]
    assert sum(p) == 1, (year, p)
    return {"year": year, "p": p, "obs": obs}


def to_hundredths(probs):
    """Round probabilities to 0.01, forcing an exact sum of 1.00."""
    h = [int(round(x * 100)) for x in probs]
    h[max(range(3), key=lambda k: probs[k])] += 100 - sum(h)
    h = [max(h_, 0) for h_ in h]
    h[max(range(3), key=lambda k: h[k])] += 100 - sum(h)
    return [F(x, 100) for x in h]


datasets = {}

# 1. Hand-checkable case, 6 years, balanced observations
datasets["case1_hand_check"] = [
    mk(2001, ["0.5", "0.3", "0.2"], "below"),
    mk(2002, ["0.2", "0.5", "0.3"], "normal"),
    mk(2003, ["0.1", "0.3", "0.6"], "above"),
    mk(2004, ["0.4", "0.4", "0.2"], "above"),
    mk(2005, ["0.6", "0.3", "0.1"], "below"),
    mk(2006, ["0.3", "0.3", "0.4"], "normal"),
]

# 2. Perfect deterministic forecast -> BSS = 1
obs_cycle = ["below", "normal", "above"] * 10
datasets["case2_perfect"] = [
    mk(1991 + i, ["1" if c == o else "0" for c in CATS], o) for i, o in enumerate(obs_cycle)
]

# 3. Climatology forecast with balanced obs -> BSS_fixed = 0
datasets["case3_climatology"] = [
    mk(1991 + i, [F(1, 3), F(1, 3), F(1, 3)], o) for i, o in enumerate(obs_cycle)
]
# (1/3 isn't a multiple of 0.01, so this file is written with 0.33 values; see note below)

# 4. Confidently wrong: puts 1.0 on a wrong category every year -> large negative BSS
wrong = {"below": "above", "normal": "below", "above": "normal"}
datasets["case4_always_wrong"] = [
    mk(1991 + i, ["1" if c == wrong[o] else "0" for c in CATS], o) for i, o in enumerate(obs_cycle)
]

# 5. Realistic: 40 years, LDA-like forecasts from a synthetic ENSO predictor
random.seed(42)
def norm_cdf(z): return 0.5 * (1 + math.erf(z / math.sqrt(2)))
n = 40
enso = [random.gauss(0, 1) for _ in range(n)]
rho = 0.6
rain = [rho * x + math.sqrt(1 - rho ** 2) * random.gauss(0, 1) for x in enso]
s = sorted(rain)
q1, q2 = s[n // 3], s[2 * n // 3]
obs = ["below" if r < q1 else ("normal" if r < q2 else "above") for r in rain]
rows = []
z1, z2 = -0.4307, 0.4307                       # standard-normal tercile boundaries
sd = math.sqrt(1 - rho ** 2)
for i in range(n):
    m = rho * enso[i]
    pb = norm_cdf((z1 - m) / sd)
    pa = 1 - norm_cdf((z2 - m) / sd)
    pn = 1 - pb - pa
    rows.append(mk(1981 + i, to_hundredths([pb, pn, pa]), obs[i]))
datasets["case5_realistic_40yr"] = rows

# 6. Imbalanced observations (dry-heavy record): fixed vs sample reference differ
random.seed(7)
obs6 = ["below"] * 14 + ["normal"] * 9 + ["above"] * 7
random.shuffle(obs6)
rows = []
for i, o in enumerate(obs6):
    base = {"below": [0.50, 0.30, 0.20], "normal": [0.30, 0.45, 0.25], "above": [0.20, 0.30, 0.50]}[o]
    if i % 4 == 3:  # every 4th forecast leans the wrong way
        base = base[::-1]
    rows.append(mk(1991 + i, to_hundredths(base), o))
datasets["case6_imbalanced"] = rows

# ---- write datasets + answer key ----
# Case 3: 1/3 is not a multiple of 0.01, so the CSV holds 0.33/0.33/0.34
# and the key is computed from exactly those file values.
key_rows = []
for name, rows in datasets.items():
    if name == "case3_climatology":
        # write CSV with 0.33/0.33/0.34 so it sums to 1, and score THAT file
        rows = [mk(r["year"], ["0.33", "0.33", "0.34"], r["obs"]) for r in rows]
        datasets[name] = rows
    write_dataset(name, rows)
    sc = scores(rows)
    sc["dataset"] = name
    key_rows.append(sc)

fields = ["dataset", "n", "n_below", "n_normal", "n_above",
          "BS_multi", "BSref_fixed_multi", "BSS_fixed_multi", "BSref_sample_multi", "BSS_sample_multi"]
for c in CATS:
    fields += [f"BS_{c}", f"BSref_fixed_{c}", f"BSS_fixed_{c}", f"BSref_sample_{c}", f"BSS_sample_{c}"]
fields += ["RPS", "RPSref_fixed", "RPSS_fixed"]

def fmt(v):
    if isinstance(v, F):
        return f"{float(v):.10f}"
    return "" if v is None else v

with open("answer_key.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(fields)
    for sc in key_rows:
        w.writerow([fmt(sc.get(k)) for k in fields])

with open("answer_key_exact_fractions.txt", "w") as f:
    for sc in key_rows:
        f.write(f"== {sc['dataset']} ==\n")
        for k in fields[1:]:
            v = sc.get(k)
            f.write(f"  {k:22s} {str(v) if v is not None else 'undefined':>22s}"
                    f"   ({fmt(v)})\n")
        f.write("\n")

for sc in key_rows:
    print(f"{sc['dataset']:24s} BS={float(sc['BS_multi']):.6f}  "
          f"BSS_fixed={float(sc['BSS_fixed_multi']):+.6f}  "
          f"BSS_sample={float(sc['BSS_sample_multi']):+.6f}  "
          f"BSS_above_fixed={float(sc['BSS_fixed_above']):+.6f}")
