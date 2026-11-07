"""G06 check: RF trained on all training flows (validation sessions included; fit_full), recomputed from per-flow
predictions with the paper's own loaders and class-stratified E3 cluster bootstrap (helpers exec'd from make_numbers.py,
nothing written by them). Writes gen/g06.json. Usage: python -I check_g06.py"""
import os as _os
ROOT = _os.environ["ARTEFACT_ROOT"] if "ARTEFACT_ROOT" in _os.environ else _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, os, re
import numpy as np
SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_numbers.py")).read()
exec(SRC[:SRC.index("facts = {}; macros = []")])          # loaders, REDIRECT (G05), mf1, bexpr, boot_delta
R6 = f"{B}/runs/G06/results"
LAD = ("leak", "no_session", "Pstar", "evidence", "E2")
checks, out = {}, {}

def rf_orig(c, s):
    return g1any(c, "multi", "RF_seq_iat", s) if c != "E2" else load(f"{R3}/B_baseline_E2_multi_RF_seq_iat_{s}_predictions.csv.gz")
def rf_full(c, s):
    return load(f"{R6}/G06_{c}_multi_RF_seq_iat_fitfull_{s}_predictions.csv.gz", raw=True)
def deep(c, m, s):
    return g2(c, m, s) if c != "E2" else load(f"{R3}/G03_E2_multi_{m}_{s}_predictions.csv.gz")
DEEP = {"Fu": "11", "Cn": "8"}

# 1. per-run checks against the server's JSON and the original runs
for c in LAD:
    for s in SEEDS:
        f, o = rf_full(c, s), rf_orig(c, s)
        js = json.load(open(f"{R6}/G06_{c}_multi_RF_seq_iat_fitfull_{s}.json"))
        k = f"{c}_{s}"
        checks[f"{k}_same_flows"] = bool((f["ids"] == o["ids"]).all() and (f["t"] == o["t"]).all())
        checks[f"{k}_same_test_folds"] = bool((f["fold"] == o["fold"]).all())
        checks[f"{k}_macro_f1_matches_json"] = abs(mf1(f["t"], f["p"]) - js["metrics"]["macro_f1"]) < 1e-12
        out[k] = {"rf_full": 100 * mf1(f["t"], f["p"]), "rf_orig": 100 * mf1(o["t"], o["p"])}
import gzip, csv
# 1b. returned fit_full index arrays equal the pre-registered definition; test arrays equal the original runs'
def fit_full_def(y, test, seed, fold, cap=1000):
    rng = np.random.default_rng(seed + fold * 10007 + 500009)
    train = np.setdiff1d(np.arange(len(y)), test); parts = []
    for c in np.unique(y[train]):
        p = train[y[train] == c].copy(); rng.shuffle(p); parts.append(p[:cap])
    f = np.concatenate(parts); rng.shuffle(f); return f.astype("<i8")
X6 = f"{B}/runs/G06"
for c in LAD:
    for s in SEEDS:
        sp, pp = ((f"{R1}/B_baseline_leak_multi_RF_seq_iat_{s}_splits.npz", f"{R1}/B_baseline_leak_multi_RF_seq_iat_{s}_predictions.csv.gz")
                  if c == "leak" else (f"{X5}/data/splits/G05_{c}_{s}_splits.npz", f"{R5}/G05_{c}_multi_RF_seq_iat_{s}_predictions.csv.gz"))
        with gzip.open(pp, "rt", newline="") as fh: rows = list(csv.DictReader(fh))
        y = np.array([int(r["true_label"]) for r in rows]); fc = np.array([int(r["fold"]) for r in rows])
        z0, z6 = np.load(sp), np.load(f"{X6}/data/splits/G06_{c}_{s}_fitfull_splits.npz"); ok = True
        for k in range(5):
            t = z0[f"fold{k}_test"]
            ok &= np.array_equal(np.sort(t), np.flatnonzero(fc == k)) and np.array_equal(z6[f"fold{k}_test"], t)
            ok &= np.array_equal(z6[f"fold{k}_fit_full"], fit_full_def(y, t, s, k))
        checks[f"{c}_{s}_fit_full_matches_definition"] = bool(ok)
def probs(path):
    with gzip.open(path, "rt", newline="") as fh: rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: r["flow_id"]); return [[r[k] for k in r if k.startswith("prob_")] for r in rows]
checks["environment_bitwise"] = probs(f"{R6}/G06_environment_reproduction_predictions.csv.gz") == probs(f"{R5}/G05_Pstar_multi_RF_seq_iat_1_predictions.csv.gz")

# 2. ladder of RF_full and paired RF_full - RF per condition
EMAIL = 1  # class index of email (checked below)
summ = {}
for c in LAD:
    F = [rf_full(c, s) for s in SEEDS]; O = [rf_orig(c, s) for s in SEEDS]
    summ[c] = {"rf_full_mean": float(np.mean([out[f"{c}_{s}"]["rf_full"] for s in SEEDS])),
               "rf_orig_mean": float(np.mean([out[f"{c}_{s}"]["rf_orig"] for s in SEEDS])),
               "full_minus_orig": boot_delta(F, O, seed=21)}
    for k, m in DEEP.items():
        D = [deep(c, m, s) for s in SEEDS]
        summ[c][f"{k}_minus_full"] = boot_delta(D, F, seed=1)
        summ[c][f"{k}_minus_full_noemail"] = boot_delta(D, F, seed=1, drop=EMAIL)
        summ[c][f"{k}_minus_orig"] = boot_delta(D, O, seed=1)

# 3. change of deep - RF_full relative to flow-random (same construction as Table 4's bottom rows)
for c in ("no_session", "Pstar", "evidence", "E2"):
    for k, m in DEEP.items():
        terms = [(1, [deep(c, m, s) for s in SEEDS]), (-1, [rf_full(c, s) for s in SEEDS]),
                 (-1, [deep("leak", m, s) for s in SEEDS]), (1, [rf_full("leak", s) for s in SEEDS])]
        al = [(w, [align(r, terms[0][1][0]) for r in runs]) for w, runs in terms]
        summ[c][f"{k}_change_vs_flow_full"] = bexpr(al, seed=0)
        summ[c][f"{k}_change_vs_flow_full_noemail"] = bexpr(al, seed=0, drop=EMAIL)

checks["email_index"] = json.load(open(f"{G1}/config/final_classes.json"))["main_classes"][EMAIL] == "email"
# 4. B: identical frames in sibling pairs (measurements only)
import glob
PP = sorted(glob.glob(f"{B}/runs/G06/audit/packet_pairs/pair_*.json"))
pairs = [json.load(open(x)) for x in PP]
checks["B_13_pairs"] = len(pairs) == 13 and all(q["status"] == "completed" and q["G01_count_matches"] for q in pairs)
med = [abs(q["dt_quantiles_seconds"]["median"]) for q in pairs]
spread = [q["dt_quantiles_seconds"]["q95"] - q["dt_quantiles_seconds"]["q05"] for q in pairs]
inv = [q["timestamp_order"]["inversion_pair_fraction"] for q in pairs]
def meta(q, side): return q[side]["metadata"]
dump = [q for q in pairs if meta(q, "b")["format"] == "pcapng" and meta(q, "a")["format"] == "pcap"
        and any("Dumpcap" in u for sec in meta(q, "b")["sections"] for u in sec.get("shb_userappl", []))]
differ = [q for q in pairs if (meta(q, "a")["format"], meta(q, "a")["snaplens"]) != (meta(q, "b")["format"], meta(q, "b")["snaplens"])]
B6 = {"n_pairs": len(pairs), "dt_zero_pairs": sum(q["dt_counts"]["dt_eq_0"] > 0 for q in pairs),
      "median_abs_dt_min_s": min(med), "median_abs_dt_max_s": max(med), "spread_q05_q95_max_ms": 1000 * max(spread),
      "inversion_max_pct": 100 * max(inv), "n_dumpcap_b_pcap_a": len(dump),
      "dumpcap_a_snaplens": sorted({sl for q in dump for sl in meta(q, "a")["snaplens"]}), "n_format_or_snaplen_differ": len(differ)}
checks["B_no_zero_dt"] = B6["dt_zero_pairs"] == 0
print("B:", B6)
json.dump({"checks": checks, "runs": out, "summary": summ, "B": B6, "n_boot": NBOOT}, open(f"{OUT}/g06.json", "w"), indent=1)
print("checks:", sum(checks.values()), "/", len(checks), "failed:", [k for k, v in checks.items() if not v])
for c in LAD:
    s = summ[c]; r = lambda t: "%+.1f [%+.1f, %+.1f]" % t
    print(f"{c:10s} RF {s['rf_orig_mean']:.1f} -> full {s['rf_full_mean']:.1f}  full-orig {r(s['full_minus_orig'])}")
    for k in DEEP:
        line = f"   {k}: -orig {r(s[k+'_minus_orig'])}  -full {r(s[k+'_minus_full'])}  -full noemail {r(s[k+'_minus_full_noemail'])}"
        if f"{k}_change_vs_flow_full" in s:
            line += f"  change {r(s[k+'_change_vs_flow_full'])} noemail {r(s[k+'_change_vs_flow_full_noemail'])}"
        print(line)
