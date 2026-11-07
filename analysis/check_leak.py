"""Check of the flow-random (leak) runs the paper uses (not rerun in G05: StratifiedKFold is unaffected by the
StratifiedGroupKFold defect). Per seed: RF (len+IAT) from G01, Fusion/1D-CNN early stopping from G02, and the G01
30-epoch deep runs. Checks: same flows in the same order; identical fit/validation/test arrays across all models;
each flow in exactly one test fold and fold column = that fold; every class has fit flows in every fold;
macro-F1/accuracy recomputed = reported. Writes outputs/leak_check.json."""
import os as _os
ROOT = _os.environ["ARTEFACT_ROOT"] if "ARTEFACT_ROOT" in _os.environ else _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import csv, glob, gzip, json, os
import numpy as np
R1 = ROOT + "/runs/G01/results/v3"
G2 = ROOT + "/runs/G02"
res = {}; fails = []
def ok(k, v):
    res[k] = bool(v)
    if not v: fails.append(k); print("FAIL", k)
def raw(path):
    with gzip.open(path, "rt", newline="") as f: rows = list(csv.DictReader(f))
    return dict(ids=np.array([r["flow_id"] for r in rows]), t=np.array([int(r["true_label"]) for r in rows]),
                p=np.array([int(r["pred_label"]) for r in rows]), fold=np.array([int(r["fold"]) for r in rows]))
def mf1(t, p):
    f = []
    for c in np.unique(t):
        tp = np.sum((t == c) & (p == c)); fp = np.sum((t != c) & (p == c)); fn = np.sum((t == c) & (p != c))
        f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))
def npz_for(stem):
    for c in (f"{stem}_splits.npz", f"{G2}/data/splits/{os.path.basename(stem)}_splits.npz"):
        if os.path.exists(c): return c
    hits = glob.glob(f"{G2}/data/splits/*{os.path.basename(stem).replace('G02_', '')}*")
    return hits[0] if len(hits) == 1 else None
for s in (1, 7, 42):
    stems = [f"{R1}/B_baseline_leak_multi_RF_seq_iat_{s}", f"{G2}/results/G02_leak_multi_8_{s}", f"{G2}/results/G02_leak_multi_11_{s}",
             f"{R1}/B_deep_leak_multi_8_{s}", f"{R1}/B_deep_leak_multi_11_{s}"]
    runs = {st: raw(st + "_predictions.csv.gz") for st in stems}
    ref = runs[stems[0]]
    Z = {}
    for st in stems:
        r = runs[st]; n = os.path.basename(st)
        ok(f"{n}: same flows in same order as RF run", np.array_equal(r["ids"], ref["ids"]) and np.array_equal(r["t"], ref["t"]))
        j = json.load(open(st + ".json")); m = j["metrics"]
        ok(f"{n}: macro-F1/accuracy recomputed = reported", abs(mf1(r["t"], r["p"]) - m["macro_f1"]) < 1e-9 and abs(np.mean(r["t"] == r["p"]) - m["accuracy"]) < 1e-9)
        z = npz_for(st); ok(f"{n}: split file found", z is not None)
        if z: Z[n] = np.load(z, allow_pickle=False)
    names = list(Z); z0 = Z[names[0]]
    for n in names[1:]:
        ok(f"seed {s}: {n} split arrays = RF split arrays", set(Z[n].files) == set(z0.files) and all(np.array_equal(Z[n][k], z0[k]) for k in z0.files))
    fold_of = np.full(len(ref["t"]), -1); tot = 0
    for k in range(5):
        fit, val, te = (z0[f"fold{k}_{x}"] for x in ("fit", "validation", "test"))
        fold_of[te] = k; tot += len(te)
        ok(f"seed {s} fold {k}: every class has fit flows", set(np.unique(ref["t"])) <= set(np.unique(ref["t"][fit])))
        ok(f"seed {s} fold {k}: fit, validation, test disjoint", not (set(fit) & set(val) or set(fit) & set(te) or set(val) & set(te)))
    ok(f"seed {s}: every flow in exactly one test fold", (fold_of >= 0).all() and tot == len(fold_of))
    for st in stems:
        ok(f"{os.path.basename(st)}: fold column = test fold of the split", np.array_equal(runs[st]["fold"], fold_of))
json.dump(dict(n_checks=len(res), n_fail=len(fails), checks=res), open(ROOT + "/outputs/leak_check.json", "w"), indent=1)
print(f"{len(res)} checks, {len(fails)} failed")
