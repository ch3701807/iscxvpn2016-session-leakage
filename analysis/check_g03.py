"""Verify the G03 return and recompute its numbers from per-flow predictions.

Checks: (1) E2/E3 groupings rebuilt locally from the G01 audit CSVs equal the returned file->group maps;
(2) prediction files: flow set = G01 P* flow set, session column = returned group, folds identical across
models within a seed, no group in two test folds, every flow tested once; (3) early-stopping rule from the
validation curves; (4) metrics recomputed; (5) paired suffix-session-cluster bootstrap CIs.
Writes outputs/g03.json. Run: python -I check_g03.py
"""
import os as _os
ROOT = _os.environ["ARTEFACT_ROOT"] if "ARTEFACT_ROOT" in _os.environ else _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import csv, gzip, hashlib, json, os, sys
from collections import defaultdict
import numpy as np

B = ROOT
G1 = f"{B}/runs/G01"; R1 = f"{G1}/results/v3"; AUD = f"{G1}/audit"
R2 = f"{B}/runs/G02/results"
X = ROOT + "/runs/G03"; R3 = f"{X}/results/v3"
SEEDS = (1, 7, 42); NBOOT = 1000
CLASSES9 = ["chat", "email", "file_transfer", "streaming", "voip", "vpn_chat", "vpn_file_transfer", "vpn_streaming", "vpn_voip"]

def mf1(t, p):
    f = []
    for c in np.unique(t):
        tp = np.sum((t == c) & (p == c)); fp = np.sum((t != c) & (p == c)); fn = np.sum((t == c) & (p != c))
        f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))

def pcf1(t, p, cs):
    out = []
    for c in cs:
        tp = np.sum((t == c) & (p == c)); fp = np.sum((t != c) & (p == c)); fn = np.sum((t == c) & (p != c))
        out.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return np.array(out)

def load(path):
    with gzip.open(path, "rt", newline="") as f: rows = list(csv.DictReader(f))
    d = dict(ids=np.array([r["flow_id"] for r in rows]), t=np.array([int(r["true_label"]) for r in rows]),
             p=np.array([int(r["pred_label"]) for r in rows]), grp=np.array([r["session"] for r in rows]),
             file=np.array([r["file"] for r in rows]), fold=np.array([int(r["fold"]) for r in rows]))
    o = np.argsort(d["ids"]); return {k: v[o] for k, v in d.items()}

def g1any(cond, model, seed):
    for pre in ("A", "B"):
        p = f"{R1}/{pre}_baseline_{cond}_multi_{model}_{seed}_predictions.csv.gz"
        if os.path.exists(p): return load(p)
    raise FileNotFoundError(cond)

def boot(A, Bm, clusters, seed=0):
    """Paired cluster bootstrap of mean over seeds of mF1(A_s) - mF1(B_s); all runs on the same flow order."""
    for a, b in zip(A, Bm): assert (a["ids"] == b["ids"]).all() and (a["ids"] == A[0]["ids"]).all()
    us = np.unique(clusters); idx = {u: np.where(clusters == u)[0] for u in us}
    rng = np.random.default_rng(seed); out = []
    for _ in range(NBOOT):
        pk = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        out.append(np.mean([100 * (mf1(a["t"][pk], a["p"][pk]) - mf1(b["t"][pk], b["p"][pk])) for a, b in zip(A, Bm)]))
    pt = float(np.mean([100 * (mf1(a["t"], a["p"]) - mf1(b["t"], b["p"])) for a, b in zip(A, Bm)]))
    lo, hi = np.percentile(out, [2.5, 97.5]); return [round(pt, 2), round(float(lo), 2), round(float(hi), 2)]

res = {"checks": {}}
ok = lambda k, v: res["checks"].__setitem__(k, bool(v)) or (print(("OK  " if v else "FAIL"), k) )

# ---- 1. groupings rebuilt locally
f2s = {r["file"]: (r["suffix_session"], r["class"]) for r in csv.DictReader(open(f"{AUD}/file_to_suffix_session.csv", newline=""))}
pairs = list(csv.DictReader(open(f"{AUD}/sibling_pairs.csv", newline="")))
def comps(edge_ok):
    par = {}
    def f(x):
        par.setdefault(x, x)
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for fl, (s, _) in f2s.items(): par.setdefault(s, s); par.setdefault(fl, fl); par[f(fl)] = f(s)
    for r in pairs:
        if edge_ok(r): par[f(r["file_a"])] = f(r["file_b"])
    g = defaultdict(set)
    for fl in f2s: g[f(fl)].add(fl)
    return {fl: min(m) for m in g.values() for fl in m}
conf = lambda r: r["verdict"] == "confirmed"
e2 = lambda r: conf(r) or (r["verdict"] == "shared_connection_insufficient" and r["time_relation"] == "overlap")
e3 = lambda r: e2(r) or (r["verdict"] == "shared_connection_insufficient" and r["time_relation"] == "separate"
                         and float(r["gap_seconds"]) <= 3600 and int(r["common_tcp_five_tuples"]) >= 1)
for name, rule in (("E2", e2), ("E3", e3)):
    mine = comps(rule)
    ret = {r["file"]: r["group"] for r in csv.DictReader(open(f"{X}/config/{name}_file_to_group.csv", newline=""))}
    part_mine = sorted(sorted(fl for fl in mine if mine[fl] == k) for k in set(mine.values()))
    part_ret = sorted(sorted(fl for fl in ret if ret[fl] == k) for k in set(ret.values()))
    ok(f"{name} partition equals local rebuild", part_mine == part_ret)
    names_ok = all(ret[fl] == "evidence_" + hashlib.sha256(mine[fl].encode()).hexdigest()[:16] for fl in mine)
    ok(f"{name} group names follow G01 rule", names_ok)
    cross = sum(len({f2s[fl][1] for fl in m}) > 1 for m in part_mine)
    per = defaultdict(int)
    for m in part_mine: per[f2s[m[0]][1]] += 1
    res[name] = dict(groups=len(part_mine), cross_class=cross, per_class={c: per[c] for c in CLASSES9})
    print(name, res[name])
    if name == "E2": E2MAP = ret
    else: E3MAP = ret

# ---- 2. references and G03 runs
ref = {s: g1any("Pstar", "RF_seq_iat", s) for s in SEEDS}             # suffix sessions in grp
ids = ref[1]["ids"]; SUF = ref[1]["grp"]
for s in SEEDS: assert (ref[s]["ids"] == ids).all() and (ref[s]["grp"] == SUF).all()
evd = {s: g1any("evidence", "RF_seq_iat", s) for s in SEEDS}
def g2(cond, m, s): return load(f"{R2}/G02_{cond}_multi_{m}_{s}_predictions.csv.gz")
REF = {("RF", "Pstar"): ref, ("RF", "evidence"): evd}
for m in ("8", "11"):
    for c in ("Pstar", "evidence"): REF[(m, c)] = {s: g2(c, m, s) for s in SEEDS}
run = {}
for m, tag in (("RF", "B_baseline_E2_multi_RF_seq_iat"), ("8", "G03_E2_multi_8"), ("11", "G03_E2_multi_11")):
    run[m] = {s: load(f"{R3}/{tag}_{s}_predictions.csv.gz") for s in SEEDS}
for s in SEEDS:
    a = run["RF"][s]
    ok(f"E2 seed {s}: flow set = P* flow set", (a["ids"] == ids).all())
    ok(f"E2 seed {s}: labels match P*", (a["t"] == ref[s]["t"]).all())
    ok(f"E2 seed {s}: session column = E2 group of file", all(E2MAP[f] == g for f, g in zip(a["file"], a["grp"])))
    ok(f"E2 seed {s}: same folds for RF, 8, 11", all((run[m][s]["fold"] == a["fold"]).all() and (run[m][s]["ids"] == ids).all() for m in ("8", "11")))
    gf = defaultdict(set)
    for g, f in zip(a["grp"], a["fold"]): gf[g].add(f)
    ok(f"E2 seed {s}: no group in two test folds", all(len(v) == 1 for v in gf.values()))
    z = np.load(f"{X}/data/splits/G03_E2_{s}_splits.npz", allow_pickle=False)
    tests = np.concatenate([z[f"fold{k}_test"] for k in range(5)])
    ok(f"E2 seed {s}: every flow tested exactly once (npz)", len(tests) == len(ids) and len(np.unique(tests)) == len(ids))
    for k in range(5):
        assert not (set(z[f"fold{k}_test"]) & (set(z[f"fold{k}_fit"]) | set(z[f"fold{k}_validation"])))

# early stopping
bad = 0; sel = []; stop = []
for m in ("8", "11"):
    for s in SEEDS:
        j = json.load(open(f"{R3}/G03_E2_multi_{m}_{s}.json"))
        for fd in j["fold_details"]:
            vc = fd["validation_curve"]
            if isinstance(vc[0], dict):
                key = [k for k in vc[0] if "f1" in k.lower()]; assert len(key) == 1, vc[0].keys()
                cv = np.array([e[key[0]] for e in vc], dtype=float)
            else: cv = np.array(vc, dtype=float)
            first_max = int(np.argmax(cv)) + 1
            best = -1; since = 0; stop_e = len(cv)
            for e, v in enumerate(cv, 1):
                if v > best: best = v; since = 0
                else: since += 1
                if since >= 30: stop_e = e; break
            if first_max != fd["selected_epoch"] or stop_e != fd["stop_epoch"] or len(cv) != fd["stop_epoch"]: bad += 1
            sel.append(fd["selected_epoch"]); stop.append(fd["stop_epoch"])
ok("E2 deep: selected = first max, stop = 30 epochs without strict improvement (30 folds)", bad == 0)
res["E2_epochs"] = dict(sel_median=float(np.median(sel)), sel_max=int(max(sel)), stop_max=int(max(stop)), capped=int(sum(x >= 300 for x in stop)), folds=len(sel))

# ---- 3. metrics and CIs (9-class, E2)
res["E2_metrics"] = {}
for m in ("RF", "8", "11"):
    v = np.array([100 * mf1(run[m][s]["t"], run[m][s]["p"]) for s in SEEDS]); a = np.array([100 * np.mean(run[m][s]["t"] == run[m][s]["p"]) for s in SEEDS])
    d = dict(mF1=round(v.mean(), 2), mF1_sd_ddof1=round(v.std(ddof=1), 2), mF1_sd_ddof0=round(v.std(), 2), acc=round(a.mean(), 2), per_seed=[round(x, 2) for x in v])
    for c in ("Pstar", "evidence"):
        R = [REF[(m, c)][s] for s in SEEDS]
        for r in R: assert (r["ids"] == ids).all()
        d[f"minus_{c}"] = boot([run[m][s] for s in SEEDS], R, SUF)
    res["E2_metrics"][m] = d; print("E2", m, d)
res["E2_rank"] = {m: boot([run[m][s] for s in SEEDS], [run["RF"][s] for s in SEEDS], SUF) for m in ("8", "11")}
print("E2 rank (deep - RF):", res["E2_rank"])
pc = {c: np.mean([pcf1(d["t"], d["p"], range(9)) for d in REF[("RF", c)].values()], axis=0) for c in ("Pstar", "evidence")}
pc["E2"] = np.mean([pcf1(run["RF"][s]["t"], run["RF"][s]["p"], range(9)) for s in SEEDS], axis=0)
res["E2_perclass_RF"] = {c: dict(zip(CLASSES9, np.round(100 * v, 1).tolist())) for c, v in pc.items()}

# ---- 4. part B (8-class RF)
p8 = {s: load(f"{R3}/B_baseline_Pstar_8_multi_RF_seq_iat_{s}_predictions.csv.gz") for s in SEEDS}
e38 = {s: load(f"{R3}/B_baseline_E3_8_multi_RF_seq_iat_{s}_predictions.csv.gz") for s in SEEDS}
keep = ref[1]["t"] != CLASSES9.index("vpn_streaming")
ok("B: P*_8 flow set = P* minus vpn_streaming", (p8[1]["ids"] == ids[keep]).all())
for s in SEEDS:
    ok(f"B seed {s}: E3_8 same flows as P*_8", (e38[s]["ids"] == p8[s]["ids"]).all())
    ok(f"B seed {s}: P*_8 session = suffix session", (p8[s]["grp"] == SUF[keep]).all())
    ok(f"B seed {s}: E3_8 session = E3 group", all(E3MAP[f] == g for f, g in zip(e38[s]["file"], e38[s]["grp"])))
    for d in (p8[s], e38[s]):
        gf = defaultdict(set)
        for g, f in zip(d["grp"], d["fold"]): gf[g].add(f)
        assert all(len(v) == 1 for v in gf.values())
cls8 = sorted(np.unique(p8[1]["t"]))
res["B"] = {}
for nm, D in (("Pstar_8", p8), ("E3_8", e38)):
    v = np.array([100 * mf1(D[s]["t"], D[s]["p"]) for s in SEEDS]); a = np.array([100 * np.mean(D[s]["t"] == D[s]["p"]) for s in SEEDS])
    pcv = np.mean([pcf1(D[s]["t"], D[s]["p"], cls8) for s in SEEDS], axis=0)
    res["B"][nm] = dict(mF1=round(v.mean(), 2), sd_ddof1=round(v.std(ddof=1), 2), acc=round(a.mean(), 2), per_seed=[round(x, 2) for x in v],
                        per_class={str(c): round(100 * x, 1) for c, x in zip(cls8, pcv)})
res["B"]["E3_8_minus_Pstar_8"] = boot([e38[s] for s in SEEDS], [p8[s] for s in SEEDS], SUF[keep])
res["B"]["E3_8_minus_Pstar_8_E3clusters"] = boot([e38[s] for s in SEEDS], [p8[s] for s in SEEDS], e38[1]["grp"])
print("B", json.dumps(res["B"], indent=1))
json.dump(res, open(f"{B}/outputs/g03.json", "w"), indent=1)
print("all checks pass:", all(res["checks"].values()))
