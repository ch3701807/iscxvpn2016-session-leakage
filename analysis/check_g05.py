"""Independent check of the G05 return (fixed StratifiedGroupKFold, all grouped conditions). Run with
python -I (scikit-learn 1.9.1 = official fixed splitter).
(1) every returned split equals the split generated locally with the official fixed splitter + G01 validation/cap code;
(2) every run: same flows/labels/files as the condition's original runs, fold = test fold of the split, each flow once;
(3) macro-F1/accuracy recomputed from predictions = reported; (4) epoch rules from the validation curves, and the
30/100/early-stopping runs agree on their shared prefix; (5) environment; (6) exposure rates from per-flow flags.
Writes outputs/g05.json."""
import os as _os
ROOT = _os.environ["ARTEFACT_ROOT"] if "ARTEFACT_ROOT" in _os.environ else _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import csv, glob, gzip, json, os
import numpy as np
src = open(ROOT + "/analysis/fixed_split_plan.py").read().split("CONDS = [")[0]
ns = {}; exec(src, ns)
Fixed, raw, source, folds, digest, same = (ns[k] for k in ("Fixed", "raw", "source", "folds", "digest", "same"))
X5 = ROOT + "/runs/G05"
plan = json.load(open(ROOT + "/outputs/fixed_split_plan.json"))
res = {"checks": {}, "runs": {}}; fails = []
def ok(k, v):
    res["checks"][k] = bool(v)
    if not v: fails.append(k); print("FAIL", k)

# (1) splits
REF = {}
for key, p in plan.items():
    cond, s = key.split("|"); s = int(s)
    pred, _, _ = source(cond, s); o = raw(pred)
    g = o["grp"] if p["group_column"] == "session" else o["file"]
    F = folds(Fixed, o["t"], g, s, s)
    ok(f"split {key}: digest = plan", digest(F) == p["sha256"])
    ok(f"split {key}: returned npz = local official fixed splitter", same(f"{X5}/data/splits/G05_{cond}_{s}_splits.npz", F))
    ok(f"split {key}: every class has fit flows in every fold", all(set(np.unique(o["t"])) <= set(np.unique(o["t"][fit])) for fit, _, _ in F))
    fold_of = np.full(len(o["t"]), -1)
    for k, (_, _, te) in enumerate(F): fold_of[te] = k
    ok(f"split {key}: every flow in exactly one test fold", (fold_of >= 0).all() and sum(len(te) for _, _, te in F) == len(fold_of))
    REF[(cond, s)] = (o, fold_of)

# (2)-(4) runs
def mf1(t, p, K):
    f = []
    for c in range(K):
        if not (t == c).any(): continue
        tp = np.sum((t == c) & (p == c)); fp = np.sum((t != c) & (p == c)); fn = np.sum((t == c) & (p != c))
        f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))
CURVES = {}
jsons = sorted(glob.glob(f"{X5}/results/v3/G05_*.json"))
ok("126 run JSONs", len(jsons) == 126)
for jf in jsons:
    d = json.load(open(jf)); j = d["job"]; rid = j["id"]; cond, s = j["condition"], j["seed"]
    o, fold_of = REF[(cond, s)]
    p = raw(jf[:-5] + "_predictions.csv.gz")
    pos = {k: i for i, k in enumerate(o["ids"])}
    ok(f"{rid}: same flow set as the original runs", len(p["ids"]) == len(o["ids"]) and set(p["ids"]) == set(o["ids"]) and len(set(p["ids"])) == len(p["ids"]))
    ix = np.array([pos[k] for k in p["ids"]])
    # G05 writes the group actually used into the session column; G01 wrote the suffix session there even when grouping by file
    used = o["file"][ix] if plan[f"{cond}|{s}"]["group_column"] == "file" else o["grp"][ix]
    ok(f"{rid}: labels and files unchanged; session column = group actually used", (p["t"] == o["t"][ix]).all() and (p["file"] == o["file"][ix]).all() and (p["grp"] == used).all())
    ok(f"{rid}: fold column = test fold of the G05 split", (p["fold"] == fold_of[ix]).all())
    K = len(d["classes"]); m, a = mf1(p["t"], p["p"], K), float(np.mean(p["t"] == p["p"]))
    ok(f"{rid}: macro-F1/accuracy recomputed = reported", abs(m - d["metrics"]["macro_f1"]) < 1e-9 and abs(a - d["metrics"]["accuracy"]) < 1e-9)
    ok(f"{rid}: split file digest recorded = plan", j["split_array_digest"] == plan[f"{cond}|{s}"]["sha256"])
    res["runs"][rid] = dict(cond=cond, seed=s, model=j["model"], budget=j["budget"], mf1=round(100 * m, 4), acc=round(100 * a, 4))
    if j["kind"] == "deep":
        sel, stop = [], []
        for fd in d["fold_details"]:
            cv = np.array([e["validation_macro_f1"] for e in fd["validation_curve"]], float)
            first_max = int(np.argmax(cv)) + 1
            if j["budget"] == "es":
                rule = len(cv) == min(300, first_max + 30)
            else:
                rule = len(cv) == j["max_epochs"]
            ok(f"{rid} fold {fd['fold']}: selected epoch = first maximum of the validation curve", fd["selected_epoch"] == first_max)
            ok(f"{rid} fold {fd['fold']}: curve length follows the budget rule", rule)
            sel.append(first_max); stop.append(len(cv))
            CURVES[(cond, j["model"], s, j["budget"], fd["fold"])] = [(e["train_loss"], e["validation_macro_f1"]) for e in fd["validation_curve"]]
        res["runs"][rid].update(selected=sel, stopped=stop)
# shared-prefix determinism: 30-epoch run = first 30 epochs of the 100-epoch and early-stopping runs (same folds, same init)
npref = 0
for (cond, mdl, s, b, f), cv in CURVES.items():
    if b != "30epoch": continue
    for b2 in ("100epoch", "es"):
        c2 = CURVES.get((cond, mdl, s, b2, f))
        if c2 is None: continue
        ok(f"prefix {cond}/{mdl}/{s}/fold{f}: 30-epoch curve = first 30 epochs of {b2}", c2[:30] == cv); npref += 1
for (cond, mdl, s, b, f), cv in CURVES.items():
    if b == "100epoch" and (cond, mdl, s, "es", f) in CURVES:
        c2 = CURVES[(cond, mdl, s, "es", f)]; n = min(len(c2), 100)
        ok(f"prefix {cond}/{mdl}/{s}/fold{f}: 100-epoch and early-stopping curves agree on {n} shared epochs", c2[:n] == cv[:n]); npref += 1
res["n_prefix_checks"] = npref
es = [r for r in res["runs"].values() if r["budget"] == "es"]
res["es_folds"] = sum(len(r["stopped"]) for r in es); res["es_capped"] = sum(x >= 300 for r in es for x in r["stopped"])
res["es_selected_median"] = float(np.median([x for r in es for x in r["selected"]])); res["es_selected_max"] = int(max(x for r in es for x in r["selected"]))
# (5) environment
env = json.load(open(f"{X5}/audit/environment.json"))
ok("environment: scikit-learn still 1.1.3", env["scikit-learn"] == "1.1.3")
# (6) exposure
res["exposure"] = {}
for cond in ("Pstar", "no_session", "evidence"):
    for s in (1, 7, 42):
        with gzip.open(f"{X5}/audit/near_duplicate_exposure/{cond}_{s}_flows.csv.gz", "rt", newline="") as f: E = list(csv.DictReader(f))
        p = raw(f"{X5}/results/v3/G05_{cond}_multi_RF_seq_iat_{s}_predictions.csv.gz"); pp = dict(zip(p["ids"], zip(p["t"], p["p"], p["fold"])))
        ok(f"exposure {cond}/{s}: flows, labels, predictions and folds = the G05 RF run",
           len(E) == len(pp) and all(pp[e["flow_id"]] == (int(e["true_label"]), int(e["pred_label"]), int(e["fold"])) for e in E))
        summ = {x["definition"]: x for x in json.load(open(f"{X5}/audit/near_duplicate_exposure/{cond}_{s}_summary.json")) if x["class"] == "__all__" and x["fold"] is None}
        c = np.array([int(e["true_label"]) == int(e["pred_label"]) for e in E])
        for dfn in ("same_connection", "same_sequence"):
            av = np.array([int(e[f"{dfn}_any"]) for e in E], bool)
            r = dict(rate=100 * av.mean(), acc_exposed=100 * c[av].mean(), acc_unexposed=100 * c[~av].mean())
            ok(f"exposure {cond}/{s}/{dfn}: recomputed = reported", abs(r["rate"] / 100 - summ[dfn]["any_exposure_rate"]) < 1e-9
               and abs(r["acc_exposed"] / 100 - summ[dfn]["exposed_accuracy"]) < 1e-9 and abs(r["acc_unexposed"] / 100 - summ[dfn]["unexposed_accuracy"]) < 1e-9)
            res["exposure"][f"{cond}|{s}|{dfn}"] = {k: round(v, 3) for k, v in r.items()}
res["n_checks"] = len(res["checks"]); res["n_fail"] = len(fails)
json.dump(res, open(ROOT + "/outputs/g05.json", "w"), indent=1)
print(f"{len(res['checks'])} checks, {len(fails)} failed; prefix checks {npref}; early-stopping folds {res['es_folds']}, capped {res['es_capped']}, "
      f"median selected {res['es_selected_median']}, max {res['es_selected_max']}")
