"""Plan for G05 (answer to the second external review, 11-08): outer splits with the FIXED StratifiedGroupKFold
(scikit-learn >= 1.8 behaviour; here the official 1.9.1 class) for every grouped condition x seed of the paper.
For each cell: which column the runs grouped by (verified by reproducing the split actually used with the 1.1.3
replica), the smallest r giving class coverage with the fixed splitter, a SHA-256 of the full fit/validation/test
index arrays (for the server to match), and fold-balance diagnostics old vs fixed.
Run with python -I . Writes outputs/fixed_split_plan.json"""
import os as _os
ROOT = _os.environ["ARTEFACT_ROOT"] if "ARTEFACT_ROOT" in _os.environ else _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import hashlib, json
import numpy as np
src = open(ROOT + "/analysis/check_g04.py").read().split("CASES = [")[0]
ns = {}; exec(src, ns)
from sklearn.model_selection import StratifiedGroupKFold as Fixed
Old = ns["StratifiedGroupKFold"]; raw = ns["raw"]; R1, X3, X4 = ns["R1"], ns["X3"], ns["X4"]

def folds(S, y, groups, seed, rs):
    """G01 run_experiment.folds() (validation group and cap exactly as in G01) with outer splitter S and random_state rs."""
    out = []
    for fold, (train, test) in enumerate(S(n_splits=5, shuffle=True, random_state=rs).split(np.zeros(len(y)), y, groups)):
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

def covered(S, y, groups, rs):
    return all(set(np.unique(y)) <= set(np.unique(y[tr])) for tr, _ in S(n_splits=5, shuffle=True, random_state=rs).split(np.zeros(len(y)), y, groups))

def digest(F):
    h = hashlib.sha256()
    for fit, val, te in F:
        for a in (fit, val, te): h.update(np.asarray(a, dtype="<i8").tobytes())
    return h.hexdigest()

def same(npz, F):
    z = np.load(npz, allow_pickle=False)
    return all(np.array_equal(z[f"fold{k}_{n}"], a) for k, (fit, val, te) in enumerate(F) for n, a in (("fit", fit), ("validation", val), ("test", te)))

def diag(y, F, groups):
    share = np.bincount(y) / len(y)
    tv = np.mean([np.abs(np.bincount(y[te], minlength=len(share)) / len(te) - share).sum() / 2 for _, _, te in F])
    tr_groups = min(len(np.unique(groups[np.setdiff1d(np.arange(len(y)), te)][y[np.setdiff1d(np.arange(len(y)), te)] == c]))
                    for _, _, te in F for c in np.unique(y))
    nofit = sum(len(set(np.unique(y)) - set(np.unique(y[fit]))) for fit, _, _ in F)
    noval = sum(len(set(np.unique(y)) - set(np.unique(y[val]))) for _, val, _ in F)
    return dict(tv=round(float(tv), 3), min_train_groups=int(tr_groups), class_fold_cells_without_fit=int(nofit),
                class_fold_cells_without_validation=int(noval))

REP = {("evidence", 1): 2, ("no_clean", 1): 1, ("E2", 1): 1, ("Pstar_8", 7): 1, ("E3_8", 7): 1}
def source(cond, s):
    """(predictions file giving y/groups in server row order, split file actually used, outer rs actually used)"""
    r = REP.get((cond, s), 0)
    if cond in ("E2", "Pstar_8", "E3_8"):
        pred = f"{X3}/results/v3/B_baseline_{cond}_multi_RF_seq_iat_{s}_predictions.csv.gz"
        npz = f"{X4}/data/splits/G04_{cond}_{s}_splits.npz" if r else f"{X3}/data/splits/G03_{cond}_{s}_splits.npz"
    else:
        pre = "A" if cond == "Pstar" else "B"
        pred = f"{R1}/{pre}_baseline_{cond}_multi_RF_seq_iat_{s}_predictions.csv.gz"
        npz = f"{X4}/data/splits/G04_{cond}_{s}_splits.npz" if r else f"{R1}/{pre}_baseline_{cond}_multi_RF_seq_iat_{s}_splits.npz"
        if r: pred = f"{X4}/results/v3/B_baseline_{cond}_multi_RF_seq_iat_{s}_predictions.csv.gz"
    return pred, npz, s + 1000 * r

CONDS = ["Pstar", "no_session", "evidence", "no_clean", "no_sni", "no_norm", "interaction", "E2", "Pstar_8", "E3_8"]
plan = {}
for cond in CONDS:
    for s in (1, 7, 42):
        pred, npz, rs_used = source(cond, s); o = raw(pred); y = o["t"]
        col = None
        for name in ("grp", "file"):
            if same(npz, folds(Old, y, o[name], s, rs_used)): col = name; break
        assert col, f"{cond}/{s}: no group column reproduces the split actually used"
        g = o[col]
        r = next(q for q in range(50) if covered(Fixed, y, g, s + 1000 * q))
        F = folds(Fixed, y, g, s, s + 1000 * r); Fo = folds(Old, y, g, s, rs_used)
        plan[f"{cond}|{s}"] = dict(group_column="session" if col == "grp" else "file", n_flows=int(len(y)), n_groups=int(len(np.unique(g))),
                                   r_fixed=r, random_state=s + 1000 * r, sha256=digest(F), old=diag(y, Fo, g), fixed=diag(y, F, g),
                                   same_as_old=bool(digest(F) == digest(Fo)))
        p = plan[f"{cond}|{s}"]
        print(f"{cond:12s} s{s:<3d} col={p['group_column']:7s} groups={p['n_groups']:3d} r={r}  TV old {p['old']['tv']:.3f} -> fixed {p['fixed']['tv']:.3f}"
              f"  min train groups {p['old']['min_train_groups']} -> {p['fixed']['min_train_groups']}  no-val cells {p['old']['class_fold_cells_without_validation']} -> {p['fixed']['class_fold_cells_without_validation']}  sha {p['sha256'][:12]}")
# no_sni / no_norm must share P*'s split (same flows, same groups)
for s in (1, 7, 42):
    assert plan[f"no_sni|{s}"]["sha256"] == plan[f"Pstar|{s}"]["sha256"] == plan[f"no_norm|{s}"]["sha256"]
json.dump(plan, open(ROOT + "/outputs/fixed_split_plan.json", "w"), indent=1)
