"""Verify the G04 return (coverage-repaired reruns) independently.

For each of the 5 repaired condition x seed:
 (1) the local replica of G01 folds() (sklearn StratifiedGroupKFold + numpy rng) reproduces the ORIGINAL split npz
     with random_state=seed (shows the local replica matches the server code);
 (2) every r' < r leaves some class without outer-training groups in some fold; r passes;
 (3) the replica with random_state=seed+1000r reproduces the NEW split npz (fit / validation / test) exactly;
 (4) new predictions: same flows, labels, files and groups as the original run, fold column = npz test fold,
     identical folds across models, no group in two test folds, every flow tested once, every class has fit
     flows in every fold;
 (5) metrics recomputed = reported; early-stopping / fixed-budget epoch rule from the validation curves;
 (6) exposure rates recomputed from per-flow flags.
Writes outputs/g04.json. Needs sklearn (any version: the 1.1.3 splitter is re-implemented above): run with
  python -I check_g04.py
"""
import os as _os
ROOT = _os.environ["ARTEFACT_ROOT"] if "ARTEFACT_ROOT" in _os.environ else _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import csv, gzip, json, os
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold as _SGKF
from sklearn.utils import check_random_state

class StratifiedGroupKFold(_SGKF):
    """scikit-learn 1.1.3 behaviour (the server's version): with shuffle=True it shuffles the rows of the
    per-group class-count table but not the group indices, so counts are decoupled from groups and the
    folds are not class-stratified. Later versions shuffle a permutation and remap groups (fixed)."""
    def _iter_test_indices(self, X, y, groups):
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
        n_classes = len(y_cnt)
        _, groups_inv, groups_cnt = np.unique(groups, return_inverse=True, return_counts=True)
        ycg = np.zeros((len(groups_cnt), n_classes))
        for ci, gi in zip(y_inv, groups_inv): ycg[gi, ci] += 1
        ycf = np.zeros((self.n_splits, n_classes)); gpf = defaultdict(set)
        if self.shuffle: rng.shuffle(ycg)
        for gi in np.argsort(-np.std(ycg, axis=1), kind="mergesort"):
            bf = self._find_best_fold(y_counts_per_fold=ycf, y_cnt=y_cnt, group_y_counts=ycg[gi])
            ycf[bf] += ycg[gi]; gpf[bf].add(gi)
        for i in range(self.n_splits):
            yield [idx for idx, gi in enumerate(groups_inv) if gi in gpf[i]]

class StratifiedGroupKFoldFixed(_SGKF):
    pass

B = ROOT
R1 = f"{B}/runs/G01/results/v3"
R2 = f"{B}/runs/G02/results"
X3 = ROOT + "/runs/G03"
X4 = ROOT + "/runs/G04"; R4 = f"{X4}/results/v3"

def raw(path):
    with gzip.open(path, "rt", newline="") as f: rows = list(csv.DictReader(f))
    return dict(ids=np.array([r["flow_id"] for r in rows]), t=np.array([int(r["true_label"]) for r in rows]),
                p=np.array([int(r["pred_label"]) for r in rows]), grp=np.array([r["session"] for r in rows]),
                file=np.array([r["file"] for r in rows]), fold=np.array([int(r["fold"]) for r in rows]))

def replica(y, groups, seed, rs):
    """G01 run_experiment.folds() with the outer random_state replaced by rs."""
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rs).split(np.zeros(len(y)), y, groups)
    out = []
    for fold, (train, test) in enumerate(outer):
        rng = np.random.default_rng(seed + fold * 10007); valgroups = set()
        for c in sorted(np.unique(y[train])):
            options = np.unique(groups[train[y[train] == c]])
            if len(options) > 1: valgroups.add(str(rng.choice(options)))
        val = train[np.isin(groups[train], list(valgroups))]; pool = train[~np.isin(groups[train], list(valgroups))]
        capped = []
        for c in sorted(np.unique(y[pool])):
            p = pool[y[pool] == c].copy(); rng.shuffle(p); capped.append(p[:1000])
        fit = np.concatenate(capped); rng.shuffle(fit)
        out.append((fit, val, test))
    return out

def outer_ok(y, groups, rs):
    """Every class keeps >= 1 group outside every test fold."""
    for train, test in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rs).split(np.zeros(len(y)), y, groups):
        if set(np.unique(y)) - set(np.unique(y[train])): return False
    return True

def same_npz(a, folds):
    z = np.load(a, allow_pickle=False)
    return all(np.array_equal(z[f"fold{k}_{n}"], arr) for k, (fit, val, te) in enumerate(folds) for n, arr in (("fit", fit), ("validation", val), ("test", te)))

def mf1(t, p):
    f = []
    for c in np.unique(t):
        tp = np.sum((t == c) & (p == c)); fp = np.sum((t != c) & (p == c)); fn = np.sum((t == c) & (p != c))
        f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))

CASES = [  # cond, seed, r, original npz, {model: (original preds, new tag)}
    ("evidence", 1, 2, f"{R1}/B_baseline_evidence_multi_RF_seq_iat_1_splits.npz",
     {"RF": (f"{R1}/B_baseline_evidence_multi_RF_seq_iat_1_predictions.csv.gz", "B_baseline_evidence_multi_RF_seq_iat_1"),
      "8": (f"{R2}/G02_evidence_multi_8_1_predictions.csv.gz", "G04_evidence_multi_8_1"),
      "11": (f"{R2}/G02_evidence_multi_11_1_predictions.csv.gz", "G04_evidence_multi_11_1")}),
    ("no_clean", 1, 1, f"{R1}/B_baseline_no_clean_multi_RF_seq_iat_1_splits.npz",
     {"RF": (f"{R1}/B_baseline_no_clean_multi_RF_seq_iat_1_predictions.csv.gz", "B_baseline_no_clean_multi_RF_seq_iat_1"),
      "11": (f"{R1}/B_deep_no_clean_multi_11_1_predictions.csv.gz", "G04_no_clean_multi_11_1"),
      "7": (f"{R1}/B_deep_no_clean_multi_7_1_predictions.csv.gz", "G04_no_clean_multi_7_1")}),
    ("E2", 1, 1, f"{X3}/data/splits/G03_E2_1_splits.npz",
     {"RF": (f"{X3}/results/v3/B_baseline_E2_multi_RF_seq_iat_1_predictions.csv.gz", "B_baseline_E2_multi_RF_seq_iat_1"),
      "8": (f"{X3}/results/v3/G03_E2_multi_8_1_predictions.csv.gz", "G04_E2_multi_8_1"),
      "11": (f"{X3}/results/v3/G03_E2_multi_11_1_predictions.csv.gz", "G04_E2_multi_11_1")}),
    ("Pstar_8", 7, 1, f"{X3}/data/splits/G03_Pstar_8_7_splits.npz",
     {"RF": (f"{X3}/results/v3/B_baseline_Pstar_8_multi_RF_seq_iat_7_predictions.csv.gz", "B_baseline_Pstar_8_multi_RF_seq_iat_7")}),
    ("E3_8", 7, 1, f"{X3}/data/splits/G03_E3_8_7_splits.npz",
     {"RF": (f"{X3}/results/v3/B_baseline_E3_8_multi_RF_seq_iat_7_predictions.csv.gz", "B_baseline_E3_8_multi_RF_seq_iat_7")}),
]

res = {"checks": {}, "runs": {}}
def ok(k, v):
    res["checks"][k] = bool(v); print(("OK  " if v else "FAIL"), k)

# the 1.1.3 replica must also reproduce untouched G01 splits (P*, no_session, evidence seeds 7/42)
for pre, tag in (("A", "Pstar"), ("B", "no_session"), ("B", "evidence")):
    for sd in (1, 7, 42):
        if tag == "evidence" and sd == 1: continue
        o = raw(f"{R1}/{pre}_baseline_{tag}_multi_RF_seq_iat_{sd}_predictions.csv.gz")
        gcol = o["file"] if tag == "no_session" else o["grp"]  # no_session groups by file
        ok(f"replica reproduces untouched G01 {tag}/{sd} split", same_npz(f"{R1}/{pre}_baseline_{tag}_multi_RF_seq_iat_{sd}_splits.npz", replica(o["t"], gcol, sd, sd)))

for cond, seed, r, orig_npz, models in CASES:
    o = raw(models["RF"][0]); y, g = o["t"], o["grp"]
    ok(f"{cond}/{seed}: local replica reproduces ORIGINAL split (rs={seed})", same_npz(orig_npz, replica(y, g, seed, seed)))
    ok(f"{cond}/{seed}: original split fails coverage", not outer_ok(y, g, seed))
    ok(f"{cond}/{seed}: every r' in 0..{r-1} fails coverage, r={r} passes",
       all(not outer_ok(y, g, seed + 1000 * q) for q in range(r)) and outer_ok(y, g, seed + 1000 * r))
    new_folds = replica(y, g, seed, seed + 1000 * r)
    new_npz = f"{X4}/data/splits/G04_{cond}_{seed}_splits.npz"
    ok(f"{cond}/{seed}: local replica (rs={seed + 1000 * r}) reproduces NEW split exactly", same_npz(new_npz, new_folds))
    fit_cov = all(set(np.unique(y)) <= set(np.unique(y[fit])) for fit, _, _ in new_folds)
    ok(f"{cond}/{seed}: every class has fit flows in every fold", fit_cov)
    tests = np.concatenate([te for _, _, te in new_folds])
    ok(f"{cond}/{seed}: every flow tested exactly once", len(tests) == len(y) and len(np.unique(tests)) == len(y))
    ok(f"{cond}/{seed}: fit/validation share no group, and neither shares a group with test",
       all(not (set(g[fit]) & set(g[val])) and not ((set(g[fit]) | set(g[val])) & set(g[te])) for fit, val, te in new_folds))
    fold_of = np.empty(len(y), int)
    for k, (_, _, te) in enumerate(new_folds): fold_of[te] = k
    for m, (orig_path, tag) in models.items():
        n = raw(f"{R4}/{tag}_predictions.csv.gz"); oo = raw(orig_path)
        ok(f"{cond}/{seed}/{m}: same flows, labels, files, groups as original run",
           all(np.array_equal(n[k], oo[k]) for k in ("ids", "t", "file", "grp")) and np.array_equal(n["ids"], o["ids"]))
        ok(f"{cond}/{seed}/{m}: fold column = new split", np.array_equal(n["fold"], fold_of))
        gf = defaultdict(set)
        for gg, ff in zip(n["grp"], n["fold"]): gf[gg].add(ff)
        ok(f"{cond}/{seed}/{m}: no group in two test folds", all(len(v) == 1 for v in gf.values()))
        j = json.load(open(f"{R4}/{tag}.json"))
        f1, acc = 100 * mf1(n["t"], n["p"]), 100 * np.mean(n["t"] == n["p"])
        ok(f"{cond}/{seed}/{m}: recomputed metrics = reported",
           abs(f1 - 100 * j["metrics"]["macro_f1"]) < 1e-6 and abs(acc - 100 * j["metrics"]["accuracy"]) < 1e-6)
        rec = dict(mF1=round(f1, 4), acc=round(acc, 4), orig_mF1=round(100 * mf1(oo["t"], oo["p"]), 4),
                   orig_acc=round(100 * np.mean(oo["t"] == oo["p"]), 4))
        if m != "RF":
            bad = 0; sel = []; stop = []
            for fd in j["fold_details"]:
                cv = np.array([e["validation_macro_f1"] for e in fd["validation_curve"]], float)
                first_max = int(np.argmax(cv)) + 1
                if cond == "no_clean":
                    stop_e = 30
                else:
                    best, since, stop_e = -1.0, 0, len(cv)
                    for e, v in enumerate(cv, 1):
                        if v > best: best, since = v, 0
                        else: since += 1
                        if since >= 30: stop_e = e; break
                if first_max != fd["selected_epoch"] or stop_e != fd["stop_epoch"] or len(cv) != fd["stop_epoch"]: bad += 1
                sel.append(fd["selected_epoch"]); stop.append(fd["stop_epoch"])
            ok(f"{cond}/{seed}/{m}: epoch rule (first max; {'fixed 30' if cond == 'no_clean' else 'patience 30, cap 300'})", bad == 0)
            rec.update(selected=sel, stopped=stop, capped=int(sum(s >= 300 for s in stop)))
        res["runs"][f"{cond}|{seed}|{m}"] = rec
        print("   ", cond, seed, m, rec)

# ---- exposure (new evidence/1 RF)
with gzip.open(f"{X4}/audit/near_duplicate_exposure/evidence_1_flows.csv.gz", "rt", newline="") as f: E = list(csv.DictReader(f))
rf = raw(f"{R4}/B_baseline_evidence_multi_RF_seq_iat_1_predictions.csv.gz")
pos = {k: i for i, k in enumerate(rf["ids"])}
ok("exposure: flows and predictions match the new evidence/1 RF run",
   len(E) == len(rf["ids"]) and all(int(e["pred_label"]) == rf["p"][pos[e["flow_id"]]] and int(e["fold"]) == rf["fold"][pos[e["flow_id"]]] for e in E))
summ = json.load(open(f"{X4}/audit/near_duplicate_exposure_summary.json"))
res["exposure"] = {}
for d in ("same_connection", "same_sequence", "same_sequence_raw"):
    a = np.array([int(e[f"{d}_any"]) for e in E], bool); c = np.array([int(e["correct"]) for e in E], bool)
    res["exposure"][d] = dict(rate=round(100 * a.mean(), 2), same_class=round(100 * np.mean([int(e[f"{d}_same_class"]) for e in E]), 2),
                              acc_exposed=round(100 * c[a].mean(), 2), acc_unexposed=round(100 * c[~a].mean(), 2))
print("exposure", res["exposure"]); res["exposure_summary_reported"] = summ
json.dump(res, open(f"{B}/outputs/g04.json", "w"), indent=1)
print("all checks pass:", all(res["checks"].values()), len(res["checks"]))
