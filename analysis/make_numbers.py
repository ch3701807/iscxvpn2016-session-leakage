"""Build every number, table and figure-data file of paper A from the verified G01-G04 returns.

v2 (11-07, after G04): (a) the five condition x seed runs whose outer split left a class without training data
(evidence/1, E2/1, no_clean/1, Pstar_8/7, E3_8/7) are replaced by their G04 coverage-repaired reruns (verified by
check_g04.py) -- done once, in load(); (b) every bootstrap CI resamples E3 groups (the coarsest grouping we have,
G03, 63 groups), not suffix sessions; (c) new: gap shrinkage, email-excluded and leave-one-class-out comparisons,
per-class session table, split-bug numbers. The pre-G04 version is kept as make_numbers_v1_preG04.py.

Nothing here is typed by hand: all values are recomputed from per-flow predictions or read from the
returned audit JSON/CSV. Outputs go to outputs/ (numbers.tex, tab_*.tex, ladder.csv, facts.json).
Run: python -I make_numbers.py
"""
import os as _os
ROOT = _os.environ["ARTEFACT_ROOT"] if "ARTEFACT_ROOT" in _os.environ else _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import csv, glob, gzip, json, os, re
from collections import Counter, defaultdict
import numpy as np

B = ROOT
G1 = f"{B}/runs/G01"; R1 = f"{G1}/results/v3"; AUD = f"{G1}/audit"
R2 = f"{B}/runs/G02/results"
X3 = ROOT + "/runs/G03"; R3 = f"{X3}/results/v3"
X4 = ROOT + "/runs/G04"; R4 = f"{X4}/results/v3"
OUT = os.environ.get("OUT_DIR", f"{B}/outputs")  # OUT_DIR only for side checks (e.g. NBOOT=10000)
G4 = json.load(open(f"{OUT}/g04.json")); assert all(G4["checks"].values()), "check_g04.py reported a failed check"
REDIRECT = {  # original run -> G04 coverage-repaired rerun (same condition, seed, model, budget)
    f"{R1}/B_baseline_evidence_multi_RF_seq_iat_1": f"{R4}/B_baseline_evidence_multi_RF_seq_iat_1",
    f"{R2}/G02_evidence_multi_8_1": f"{R4}/G04_evidence_multi_8_1",
    f"{R2}/G02_evidence_multi_11_1": f"{R4}/G04_evidence_multi_11_1",
    f"{R1}/B_baseline_no_clean_multi_RF_seq_iat_1": f"{R4}/B_baseline_no_clean_multi_RF_seq_iat_1",
    f"{R1}/B_deep_no_clean_multi_11_1": f"{R4}/G04_no_clean_multi_11_1",
    f"{R1}/B_deep_no_clean_multi_7_1": f"{R4}/G04_no_clean_multi_7_1",
    f"{R3}/B_baseline_E2_multi_RF_seq_iat_1": f"{R4}/B_baseline_E2_multi_RF_seq_iat_1",
    f"{R3}/G03_E2_multi_8_1": f"{R4}/G04_E2_multi_8_1",
    f"{R3}/G03_E2_multi_11_1": f"{R4}/G04_E2_multi_11_1",
    f"{R3}/B_baseline_Pstar_8_multi_RF_seq_iat_7": f"{R4}/B_baseline_Pstar_8_multi_RF_seq_iat_7",
    f"{R3}/B_baseline_E3_8_multi_RF_seq_iat_7": f"{R4}/B_baseline_E3_8_multi_RF_seq_iat_7",
}
# v3 (11-08): SPLITTER=fixed (default) takes every grouped 9- and 8-class run from G05 (fixed StratifiedGroupKFold,
# checked by check_g05.py); flow-random runs stay G01/G02. SPLITTER=old reproduces v2 (scikit-learn 1.1.3 splits + G04 repairs).
SPLITTER = os.environ.get("SPLITTER", "fixed"); assert SPLITTER in ("fixed", "old")
X5 = ROOT + "/runs/G05"; R5 = f"{X5}/results/v3"
GROUPED = re.compile(r"_(Pstar_8|E3_8|Pstar|no_session|evidence|no_clean|no_sni|no_norm|interaction|E2)_multi_")
if SPLITTER == "fixed":
    G5 = json.load(open(f"{OUT}/g05.json")); assert G5["n_fail"] == 0 and G5["n_checks"] >= 1800, "check_g05.py reported a failure"
    REDIRECT = {}
    for rid, r in G5["runs"].items():
        c, m, sd, b = r["cond"], r["model"], r["seed"], r["budget"]
        if c == "Pstar":
            orig = {"baseline": f"{R1}/A_baseline_Pstar_multi_{m}_{sd}", "30epoch": f"{R1}/A_deep_Pstar_multi_{m}_{sd}",
                    "100epoch": f"{R1}/Bplus1_deep_Pstar_100epoch_multi_{m}_{sd}", "es": f"{R2}/G02_Pstar_multi_{m}_{sd}"}[b]
        elif c in ("E2", "Pstar_8", "E3_8"):
            orig = {"baseline": f"{R3}/B_baseline_{c}_multi_{m}_{sd}", "es": f"{R3}/G03_{c}_multi_{m}_{sd}"}[b]
        else:
            orig = {"baseline": f"{R1}/B_baseline_{c}_multi_{m}_{sd}", "30epoch": f"{R1}/B_deep_{c}_multi_{m}_{sd}", "es": f"{R2}/G02_{c}_multi_{m}_{sd}"}[b]
        assert orig not in REDIRECT; REDIRECT[orig] = f"{R5}/{rid}"
    assert len(REDIRECT) == 126
def guard(path):
    if SPLITTER == "fixed" and GROUPED.search(os.path.basename(path)) and not path.startswith(R5):
        raise AssertionError(f"grouped run not taken from G05: {path}")
USED_REDIRECT = set()
SEEDS = (1, 7, 42)
NBOOT = int(os.environ.get("NBOOT", "10000"))  # v3.1 (third review): 10,000 resamples; NBOOT=1000 gives v3

def mf1(t, p, classes=None):
    cs = np.unique(t) if classes is None else classes
    f = []
    for c in cs:
        tp = np.sum((t == c) & (p == c)); fp = np.sum((t != c) & (p == c)); fn = np.sum((t == c) & (p != c))
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f))

def per_class_f1(t, p, ncls):
    out = []
    for c in range(ncls):
        tp = np.sum((t == c) & (p == c)); fp = np.sum((t != c) & (p == c)); fn = np.sum((t == c) & (p != c))
        out.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return np.array(out)

_cache = {}
def load(path, raw=False):
    stem = path[:-len("_predictions.csv.gz")]
    if raw: assert os.path.exists(path), path
    elif stem in REDIRECT: USED_REDIRECT.add(stem); path = REDIRECT[stem] + "_predictions.csv.gz"
    if not raw: guard(path)
    if path in _cache: return _cache[path]
    with gzip.open(path, "rt", newline="") as f: rows = list(csv.DictReader(f))
    d = dict(ids=np.array([r["flow_id"] for r in rows]), t=np.array([int(r["true_label"]) for r in rows]),
             p=np.array([int(r["pred_label"]) for r in rows]), grp=np.array([r["session"] for r in rows]),
             file=np.array([r["file"] for r in rows]), fold=np.array([int(r["fold"]) for r in rows]))
    o = np.argsort(d["ids"]); d = {k: v[o] for k, v in d.items()}
    _cache[path] = d
    return d

def g1(prefix, cond, task, model, seed):
    return load(f"{R1}/{prefix}_{'deep' if model.isdigit() else 'baseline'}_{cond}_{task}_{model}_{seed}_predictions.csv.gz")

def g1any(cond, task, model, seed):
    for pre in ("A", "B"):
        p = f"{R1}/{pre}_{'deep' if model.isdigit() else 'baseline'}_{cond}_{task}_{model}_{seed}_predictions.csv.gz"
        st = p[:-len("_predictions.csv.gz")]  # a run redirected to G05 counts as present even if the original file is absent
        if os.path.exists(REDIRECT[st] + "_predictions.csv.gz" if st in REDIRECT else p): return load(p)
    return None

def g2(cond, model, seed):
    return load(f"{R2}/G02_{cond}_multi_{model}_{seed}_predictions.csv.gz")

def stats(runs, fn=lambda r: mf1(r["t"], r["p"])):
    v = np.array([fn(r) for r in runs]) * 100
    return float(v.mean()), float(v.std(ddof=1)), v

acc = lambda r: float(np.mean(r["t"] == r["p"]))

E3MAP = {r["file"]: r["group"] for r in csv.DictReader(open(f"{X3}/config/E3_file_to_group.csv", newline=""))}
SUFMAP = {r["file"]: r["suffix_session"] for r in csv.DictReader(open(f"{AUD}/file_to_suffix_session.csv", newline=""))}

def f1cm(cm, exclude=None):
    """macro-F1 from confusion matrices (..., K, K), averaged over classes present in the truth
    (with the class-stratified bootstrap every class is always present). exclude = class index left out of the average
    (its flows stay in the confusion matrix, so the other classes' F1 is unchanged)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1); sup = cm.sum(-1); den = sup + cm.sum(-2)
    f = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0); present = sup > 0
    if exclude is not None: present = present & (np.arange(cm.shape[-1]) != exclude)
    return (f * present).sum(-1) / present.sum(-1)

# v3 (second external review, 11-08): resample clusters within each class (strata), so that every resample keeps the
# full class set; leave-one-class-out removes the class's term from the macro average ("term"), not its test flows.
BOOT_STRAT = os.environ.get("BOOT_STRAT", "1") == "1"
DROP_MODE = os.environ.get("DROP_MODE", "term")
assert DROP_MODE in ("term", "flows")

def bexpr(terms, drop=None, cluster="E3", seed=0):
    """Paired cluster bootstrap of sum_i w_i * mean_seed mF1(run_i,s). terms = [(w, [run per seed]), ...], all runs
    on the same flows (sorted ids). drop = class index whose test flows are removed (leave-one-class-out).
    Clusters = E3 groups of the flows' files (or suffix sessions). Returns (point, lo, hi) in pp."""
    ref = terms[0][1][0]
    for _, runs in terms:
        for r in runs: assert (r["ids"] == ref["ids"]).all()
    keep = np.ones(len(ref["ids"]), bool) if (drop is None or DROP_MODE == "term") else ref["t"] != drop
    exc = drop if DROP_MODE == "term" else None
    cl = np.array([(E3MAP if cluster == "E3" else SUFMAP)[f] for f in ref["file"][keep]])
    us, inv = np.unique(cl, return_inverse=True); C = len(us)
    K = 1 + max(int(max(r["t"].max(), r["p"].max())) for _, runs in terms for r in runs)
    rng = np.random.default_rng(seed)
    if BOOT_STRAT:
        gcls = np.full(C, -1)
        for g, t in zip(inv, ref["t"][keep]):
            assert gcls[g] in (-1, t), "cluster spans two classes"; gcls[g] = t
        W = np.zeros((NBOOT, C))
        for c in np.unique(gcls):
            G = np.flatnonzero(gcls == c); n = len(G)
            pick = G[rng.integers(0, n, (NBOOT, n))]
            np.add.at(W, (np.repeat(np.arange(NBOOT), n), pick.ravel()), 1)
    else:
        W = np.stack([np.bincount(rng.integers(0, C, C), minlength=C) for _ in range(NBOOT)]).astype(float)
    tot = np.zeros(NBOOT); pt = 0.0
    for w, runs in terms:
        for r in runs:
            cm = np.zeros((C, K, K)); np.add.at(cm, (inv, r["t"][keep], r["p"][keep]), 1)
            tot += w * f1cm(np.einsum("bc,ckl->bkl", W, cm), exc) / len(runs); pt += w * f1cm(cm.sum(0), exc) / len(runs)
    lo, hi = np.percentile(100 * tot, [2.5, 97.5])
    return float(100 * pt), float(lo), float(hi)

def boot_delta(A, Bm, seed=0, drop=None, cluster="E3"):
    return bexpr([(1, A), (-1, Bm)], drop=drop, cluster=cluster, seed=seed)

def align(d, ref):
    """Re-index run d onto ref's flow order (same flow set required) and borrow ref's suffix sessions."""
    assert (d["ids"] == ref["ids"]).all() or set(d["ids"]) == set(ref["ids"])
    if not (d["ids"] == ref["ids"]).all():
        pos = {k: i for i, k in enumerate(d["ids"])}; ix = np.array([pos[k] for k in ref["ids"]])
        d = {k: v[ix] for k, v in d.items()}
    return dict(d, grp=ref["grp"])

facts = {}; macros = []
def M(name, value, meaning, fmt="{:.1f}"):
    num = isinstance(value, (int, float, np.floating, np.integer)) and fmt
    s = fmt.format(value) if num else str(value)
    if num and s.startswith("-"): s = "$-$" + s[1:]
    if num and s in ("$-$0.0", "$-$0"): s = s[3:]
    assert re.fullmatch(r"[A-Za-z]+", name), name
    macros.append(f"\\newcommand{{\\{name}}}{{{s}}}")
    facts[name] = dict(value=s, meaning=meaning)
    return s
sg = lambda x: "0.0" if abs(x) < 0.05 else ("+" if x > 0 else "$-$") + f"{abs(x):.1f}"

# ---------------------------------------------------------------- dataset / sibling audit
classes = json.load(open(f"{G1}/config/final_classes.json"))["main_classes"]
inv = json.load(open(f"{AUD}/inventory_summary.json"))
M("NFiles", inv["iscx_pcap_file_count"], "ISCXVPN2016 capture files examined", "{:d}")
M("NClasses", len(classes), "classes in the multi-class task (vpn_email, vpn_p2p excluded: one session each)", "{:d}")
ref = g1("A", "Pstar", "multi", "RF_seq_iat", 1)
M("NFlows", len(ref["ids"]), "flows in the 9-class task after cleaning", "{:d}")
M("NFlowsBin", len(g1("A", "Pstar", "binary", "RF_seq_iat", 1)["ids"]), "flows in the VPN/non-VPN task", "{:d}")
suf = list(csv.DictReader(open(f"{AUD}/file_to_suffix_session.csv")))
ev = json.load(open(f"{AUD}/evidence_sessions.json"))["file_to_session"]
M("NSuffixGroups", len({r["suffix_session"] for r in suf}), "suffix-name session groups over all files", "{:d}")
M("NEvidenceGroups", len(set(ev.values())), "evidence-reconstructed session groups over all files", "{:d}")
M("NSessEmail", len(np.unique(ref["grp"][ref["t"] == classes.index("email")])), "suffix sessions of class email in the 9-class task", "{:d}")
M("NSessMulti", len(np.unique(ref["grp"])), "suffix sessions in the 9-class task", "{:d}")
evd = g1("B", "evidence", "multi", "RF_seq_iat", 1)
M("NEvSessMulti", len(np.unique(evd["grp"])), "evidence sessions in the 9-class task", "{:d}")
ss = json.load(open(f"{AUD}/sibling_summary.json"))
M("NPairs", ss["pair_count"], "candidate file pairs examined (suffix pairs + time-adjacent pairs)", "{:d}")
for k, n in (("confirmed", "VConfirmed"), ("possible", "VPossible"), ("no_evidence", "VNone"), ("shared_connection_insufficient", "VSharedOnly")):
    M(n, ss["verdict_counts"][k], f"pairs with verdict {k}", "{:d}")
M("NNewEdges", ss["new_confirmed_edges"], "confirmed same-session edges joining different suffix groups", "{:d}")
pairs = list(csv.DictReader(open(f"{AUD}/sibling_pairs.csv")))
sp = [p for p in pairs if "suffix" in p["candidate_origins"]]
M("NSuffixPairs", len(sp), "suffix a/b pairs examined", "{:d}")
tr = Counter(p["time_relation"] for p in sp)
M("NSuffixOverlap", tr.get("overlap", 0), "suffix pairs whose capture intervals overlap in time", "{:d}")
M("NSuffixAdjacent", tr.get("adjacent", 0), "suffix pairs captured back-to-back (gap <= 60 s)", "{:d}")
M("NSuffixSeparate", tr.get("separate", 0), "suffix pairs separated by > 60 s", "{:d}")
fr = sorted(float(p["identical_packet_fraction_min"]) for p in sp)
M("NSuffixIdent", sum(f > 0 for f in fr), "suffix pairs sharing at least one byte-identical packet", "{:d}")
M("SuffixIdentMax", 100 * fr[-1], "largest share of byte-identical packets in a suffix pair (% of smaller file)")
M("NSuffixIdentForty", sum(f >= 0.40 for f in fr), "suffix pairs where >= 40% of the smaller file's packets are byte-identical", "{:d}")
M("NSuffixShared", sum(int(p["common_tcp_five_tuples"] or 0) + int(p["common_udp_five_tuples"] or 0) > 0 for p in sp), "suffix pairs sharing >= 1 TCP/UDP five-tuple", "{:d}")
vc = Counter(p["verdict"] for p in sp)
for k, n in (("confirmed", "SufConfirmed"), ("possible", "SufPossible"), ("no_evidence", "SufNone"), ("shared_connection_insufficient", "SufSharedOnly")):
    M(n, vc.get(k, 0), f"suffix pairs with verdict {k}", "{:d}")
M("NSuspectedDup", sum(p["suspected_duplicate"].lower() in ("true", "1") for p in pairs), "pairs flagged as duplicate captures (>= 90% identical packets, both >= 100 packets)", "{:d}")
M("NSuffixSplitTCP", sum(int(p["split_tcp_connections"]) > 0 for p in sp), "suffix pairs with a TCP connection split across the two files", "{:d}")
newedge = [p for p in pairs if p["verdict"] == "confirmed" and "suffix" not in p["candidate_origins"]]
facts["new_edges_list"] = [(p["file_a"], p["file_b"], p["class"], p["time_relation"]) for p in newedge]
M("NCrossClassEdges", sum(1 for p in newedge if "|" in p["class"] or "," in p["class"]), "confirmed edges whose files carry different class labels", "{:d}")
# cleaning and SNI
cl = json.load(open(f"{AUD}/v3_cleaning_inventory.json"))["classes"]
bf = sum(c["before_flows"] for c in cl if c["class"] in classes); af = sum(c["after_flows"] for c in cl if c["class"] in classes)
M("CleanRemovedPct", 100 * (1 - af / bf), "share of flows removed by background-port/broadcast cleaning (9 classes)")
facts["cleaning_by_class"] = {c["class"]: round(100 * c["removed_fraction"], 1) for c in cl}
sni = json.load(open(f"{AUD}/sni_visible_boundary_validation.json"))
M("NSNIRecords", sni["visible_sni_name_records_found"], "visible SNI host names zeroed (all clean flows, binary population)", "{:d}")
# encapsulation
enc = json.load(open(f"{AUD}/vpn_encapsulation_summary.json"))["groups"]
vpn = [g for g in enc if g["vpn_file"]][0]; nvpn = [g for g in enc if not g["vpn_file"]][0]
M("NVPNFiles", vpn["files"], "VPN-labelled capture files", "{:d}")
M("VPNESP", vpn["ESP_50_packets"] + vpn["GRE_47_packets"], "ESP+GRE packets in VPN files", "{:d}")
M("VPNIKE", vpn["port_category_packet_counts"]["UDP_500"], "UDP/500 packets in VPN files", "{:d}")
M("VPNNATT", vpn["port_category_packet_counts"]["UDP_4500"], "UDP/4500 packets in VPN files", "{:d}")
M("VPNOpenVPN", vpn["port_category_packet_counts"]["UDP_1194"], "UDP/1194 packets in VPN files", "{:d}")
M("VPNCHPct", 100 * vpn["tls_flow_statistics"]["after_clean"]["clienthello_flow_fraction"], "VPN-file flows with a visible TLS ClientHello (%)")
M("VPNSNIPct", 100 * vpn["tls_flow_statistics"]["after_clean"]["sni_flow_fraction"], "VPN-file flows with a visible SNI (%)")
M("NonVPNCHPct", 100 * nvpn["tls_flow_statistics"]["after_clean"]["clienthello_flow_fraction"], "non-VPN-file flows with a visible ClientHello (%)")
M("NonVPNSNIPct", 100 * nvpn["tls_flow_statistics"]["after_clean"]["sni_flow_fraction"], "non-VPN-file flows with a visible SNI (%)")
# USTC
us = json.load(open(f"{AUD}/ustc_summary.json"))["classes"]
M("NUSTCClasses", len(us), "USTC-TFC2016 classes inventoried", "{:d}")
M("NUSTCOneFile", sum(c["file_count"] == 1 for c in us), "USTC classes distributed as a single capture file", "{:d}")
M("NUSTCTwoSess", sum(c["suffix_sessions_with_clean_flows"] >= 2 for c in us), "USTC classes with >= 2 sessions holding clean flows", "{:d}")
M("NUSTCZero", sum(c["clean_flows"] == 0 for c in us), "USTC classes yielding no eligible flow under our extraction", "{:d}")
M("NUSTCConflict", sum(bool(c.get("cleaning_rule_conflict")) for c in us), "USTC classes conflicting with the cleaning rule", "{:d}")
facts["ustc_two_session_classes"] = [c["class"] for c in us if c["suffix_sessions_with_clean_flows"] >= 2]
facts["ustc_zero_classes"] = [c["class"] for c in us if c["clean_flows"] == 0]
facts["ustc_conflict_classes"] = [c["class"] for c in us if c.get("cleaning_rule_conflict")]

# ---------------------------------------------------------------- main table (P*)
NAMES = {"majority": "Majority class", "stratified_random": "Stratified random", "LR_stat": "LR (stat)",
         "LR_seq_iat": "LR (len+IAT)", "RF_stat": "RF (stat)", "RF_seq_iat": "RF (len+IAT)",
         "1": "1D-CNN (len)", "20": "1D-CNN (first 10 len)", "2": "CNN-BiLSTM (len)", "3": "Transformer (len)",
         "8": "1D-CNN (len+IAT)", "7": "1D-CNN (bytes)", "11": "Fusion (len+IAT+bytes+stat)", "25": "1D-CNN (len)"}
rows = []
for m in ("majority", "stratified_random", "LR_stat", "LR_seq_iat", "RF_stat", "RF_seq_iat", "7", "1", "20", "2", "3", "8", "11"):
    runs = [g1("A", "Pstar", "multi", m, s) for s in SEEDS]
    f, fs, _ = stats(runs); a, as_, _ = stats(runs, acc)
    e100 = conv = None
    if m.isdigit():
        e100 = stats([load(f"{R1}/Bplus1_deep_Pstar_100epoch_multi_{m}_{s}_predictions.csv.gz") for s in SEEDS])
    if m in ("8", "11"):
        conv = stats([g2("Pstar", m, s) for s in SEEDS])
    rows.append((m, f, fs, a, as_, e100, conv))
    facts[f"Pstar_multi_{m}"] = dict(f1=round(f, 2), sd=round(fs, 2), acc=round(a, 2), f1_100=round(e100[0], 2) if e100 else None, f1_conv=round(conv[0], 2) if conv else None)
L = ["\\begin{tabular}{llrrrr}", "\\toprule",
     " & Model & \\multicolumn{2}{c}{30 epochs (planned)} & 100 ep. & Early stop \\\\",
     " & & macro-F1 & accuracy & macro-F1 & macro-F1 \\\\", "\\midrule"]
for m, f, fs, a, as_, e100, conv in rows:
    kind = "Trivial" if m in ("majority", "stratified_random") else ("Shallow" if not m.isdigit() else "Deep")
    L.append(f"{kind} & {NAMES[m]} & {f:.1f} $\\pm$ {fs:.1f} & {a:.1f} & {e100[0]:.1f} & " if e100 else f"{kind} & {NAMES[m]} & {f:.1f} $\\pm$ {fs:.1f} & {a:.1f} & -- & ")
    L[-1] += (f"{conv[0]:.1f} \\\\" if conv else "-- \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{OUT}/tab_main.tex", "w").write("\n".join(L) + "\n")
rf = [r for r in rows if r[0] == "RF_seq_iat"][0]
M("PRF", rf[1], "P* 9-class macro-F1, RF(len+IAT), 30-epoch table (mean of 3 seeds, %)")
M("PRFsd", rf[2], "its seed s.d.")
M("PRFstat", [r for r in rows if r[0] == "RF_stat"][0][1], "P* macro-F1, RF(stat)")
M("PFuThirty", [r for r in rows if r[0] == "11"][0][1], "P* macro-F1, Fusion at 30 epochs")
M("PCnThirty", [r for r in rows if r[0] == "8"][0][1], "P* macro-F1, 1D-CNN(len+IAT) at 30 epochs")
M("PFuHundred", [r for r in rows if r[0] == "11"][0][5][0], "P* macro-F1, Fusion at 100 epochs")
M("PCnHundred", [r for r in rows if r[0] == "8"][0][5][0], "P* macro-F1, 1D-CNN(len+IAT) at 100 epochs")
M("PFuConv", [r for r in rows if r[0] == "11"][0][6][0], "P* macro-F1, Fusion with early stopping (<=300 epochs)")
M("PCnConv", [r for r in rows if r[0] == "8"][0][6][0], "P* macro-F1, 1D-CNN(len+IAT) with early stopping")
M("PBytes", [r for r in rows if r[0] == "7"][0][1], "P* macro-F1, bytes-only CNN")
M("PMajority", [r for r in rows if r[0] == "majority"][0][1], "P* macro-F1, majority class")
M("PStrat", [r for r in rows if r[0] == "stratified_random"][0][1], "P* macro-F1, stratified random")
M("PRFBin", stats([g1("A", "Pstar", "binary", "RF_seq_iat", s) for s in SEEDS])[0], "P* VPN/non-VPN macro-F1, RF(len+IAT)")
M("PCNNBin", stats([g1("A", "Pstar", "binary", "25", s) for s in SEEDS])[0], "P* VPN/non-VPN macro-F1, 1D-CNN(len), 30 epochs")
# 100 vs 30 epoch deltas with CI (dir 8, 11)
for m, nm in (("11", "Fu"), ("8", "Cn")):
    A = [load(f"{R1}/Bplus1_deep_Pstar_100epoch_multi_{m}_{s}_predictions.csv.gz") for s in SEEDS]
    Bm = [g1("A", "Pstar", "multi", m, s) for s in SEEDS]
    pt, lo, hi = boot_delta(A, Bm, seed=11)
    M(f"DEpoch{nm}", pt, f"100-epoch minus 30-epoch macro-F1, {NAMES[m]} (pp)"); M(f"DEpoch{nm}Lo", lo, "CI low"); M(f"DEpoch{nm}Hi", hi, "CI high")
# epochs in G02
sel = []; capped = 0
# explicit list (not a glob), so the result does not depend on which superseded G02 files are present
G02_ES = sorted(f"{R2}/G02_{c}_multi_{m}_{s}.json" for c in ("leak", "no_session", "Pstar", "evidence") for m in ("8", "11") for s in SEEDS)
if os.path.isdir(R2) and len(glob.glob(f"{R2}/G02_*_multi_*.json")) == 24: assert G02_ES == sorted(glob.glob(f"{R2}/G02_*_multi_*.json"))
for f in G02_ES:
    if f[:-5] in REDIRECT: f = REDIRECT[f[:-5]] + ".json"
    guard(f)
    for fd in json.load(open(f))["fold_details"]:
        sel.append(fd["selected_epoch"]); capped += fd["stop_epoch"] >= 300
M("ConvMaxSel", max(sel), "latest selected epoch over all early-stopping folds", "{:d}")
M("ConvMedSel", float(np.median(sel)), "median selected epoch, early stopping", "{:.0f}")
M("ConvCapped", capped, "early-stopping folds that reached the 300-epoch cap", "{:d}")
M("ConvFolds", len(sel), "early-stopping folds in total", "{:d}")

# ---------------------------------------------------------------- leakage ladder
LAD = (("leak", "Flow-random"), ("no_session", "File-grouped"), ("Pstar", "Suffix session (P*)"), ("evidence", "Evidence session"))
ladder = {}
rfP = [g1("A", "Pstar", "multi", "RF_seq_iat", s) for s in SEEDS]
for key, getter in (("RF", lambda c, s: g1any(c, "multi", "RF_seq_iat", s)), ("Fu", lambda c, s: g2(c, "11", s)), ("Cn", lambda c, s: g2(c, "8", s))):
    P = [getter("Pstar", s) for s in SEEDS]
    for c, _ in LAD:
        runs = [getter(c, s) for s in SEEDS]
        f, fs, _ = stats(runs)
        if c == "Pstar": ladder[(key, c)] = (f, fs, None, None, None); continue
        A2 = [align(r, p) for r, p in zip(runs, P)]
        pt, lo, hi = boot_delta(A2, [align(p, p) for p in P], seed=0)
        ladder[(key, c)] = (f, fs, pt, lo, hi)
CN = {"leak": "Flow", "no_session": "File", "Pstar": "Suffix", "evidence": "Evid"}
MN = {"RF": "RF(len+IAT)", "Fu": "Fusion, early stopping", "Cn": "1D-CNN(len+IAT), early stopping"}
for (k, c), (f, fs, pt, lo, hi) in ladder.items():
    M(f"Lad{k}{CN[c]}", f, f"ladder macro-F1, {MN[k]}, {c} (%)")
    if pt is not None:
        M(f"D{k}{CN[c]}", pt, f"ladder: {c} minus P*, {MN[k]} (pp)"); M(f"D{k}{CN[c]}Lo", lo, "CI low"); M(f"D{k}{CN[c]}Hi", hi, "CI high")
with open(f"{OUT}/ladder.csv", "w") as fh:
    fh.write("model,condition,macro_f1,sd,delta,lo,hi\n")
    for (k, c), v in ladder.items(): fh.write(f"{k},{c}," + ",".join("" if x is None else f"{x:.3f}" for x in v) + "\n")
L = ["\\begin{tabular}{lccc}", "\\toprule", "Split & RF (len+IAT) & Fusion$^\\dagger$ & 1D-CNN (len+IAT)$^\\dagger$ \\\\", "\\midrule"]
for c, lab in LAD:
    vals = [ladder[(k, c)] for k in ("RF", "Fu", "Cn")]
    L.append(f"{lab} & " + " & ".join(f"{v[0]:.1f}" for v in vals) + " \\\\")
    if vals[0][2] is not None:
        L.append("\\quad $\\Delta$ vs.\\ P* & " + " & ".join(f"{sg(v[2])} [{sg(v[3])}, {sg(v[4])}]" for v in vals) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{OUT}/tab_ladder.tex", "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- ranking: deep - RF per split (all classes / email excluded)
EMAIL = classes.index("email")
DEEP = {"Fu": "11", "Cn": "8"}
def runs_of(k, c): return [g1any(c, "multi", "RF_seq_iat", s) if k == "RF" else g2(c, DEEP[k], s) for s in SEEDS]
L = ["\\begin{tabular}{lcccc}", "\\toprule",
     " & \\multicolumn{2}{c}{Fusion $-$ RF} & \\multicolumn{2}{c}{1D-CNN (len+IAT) $-$ RF} \\\\",
     "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "Split & all classes & w/o \\texttt{email} & all classes & w/o \\texttt{email} \\\\", "\\midrule"]
def two_rows(lab, vals):
    L.append(f"{lab} & " + " & ".join(sg(v[0]) for v in vals) + " \\\\")
    L.append(" & " + " & ".join(f"[{sg(v[1])}, {sg(v[2])}]" for v in vals) + " \\\\[1pt]")
for c, lab in LAD:
    cells = []
    for k in ("Fu", "Cn"):
        for drop, sfx in ((None, ""), (EMAIL, "NE")):
            pt, lo, hi = boot_delta(runs_of(k, c), runs_of("RF", c), seed=1, drop=drop)
            what = "" if drop is None else ", email test flows removed"
            M(f"R{k}{CN[c]}{sfx}", pt, f"{MN[k]} minus RF(len+IAT) under {c}{what} (pp)"); M(f"R{k}{CN[c]}{sfx}Lo", lo, "CI low"); M(f"R{k}{CN[c]}{sfx}Hi", hi, "CI high")
            if sfx: M(f"R{k}{CN[c]}{sfx}Abs", abs(pt), "absolute value of the above (for prose: 'RF leads by')")
            cells.append((pt, lo, hi))
    two_rows(lab, cells)
# shrinkage of the gap: (deep - RF)_split - (deep - RF)_flow-random
L.append("\\midrule")
SH = {}
for c, lab in LAD[1:]:
    cells = []
    for k in ("Fu", "Cn"):
        for drop, sfx in ((None, ""), (EMAIL, "NE")):
            pt, lo, hi = bexpr([(1, runs_of(k, c)), (-1, runs_of("RF", c)), (-1, runs_of(k, "leak")), (1, runs_of("RF", "leak"))], drop=drop, seed=2)
            SH[(k, c, sfx)] = (pt, lo, hi)
            M(f"S{k}{CN[c]}{sfx}", pt, f"change of ({MN[k]} minus RF) from flow-random to {c}{'' if drop is None else ', email removed'} (pp)")
            M(f"S{k}{CN[c]}{sfx}Lo", lo, "CI low"); M(f"S{k}{CN[c]}{sfx}Hi", hi, "CI high")
            cells.append((pt, lo, hi))
    if c == "no_session": L.append("\\multicolumn{5}{l}{\\emph{Change of the difference from flow-random to:}} \\\\")
    if c != "evidence": two_rows(f"\\quad {lab.split(' (')[0].lower()}", cells)
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{OUT}/tab_rank.tex", "w").write("\n".join(L) + "\n")
# change of (deep - RF) from file-grouped to suffix sessions (supports "file grouping still overstates RF's lead")
for k in ("Fu", "Cn"):
    pt, lo, hi = bexpr([(1, runs_of(k, "Pstar")), (-1, runs_of("RF", "Pstar")), (-1, runs_of(k, "no_session")), (1, runs_of("RF", "no_session"))], seed=2)
    M(f"S{k}FileToSuffix", pt, f"change of ({MN[k]} minus RF) from file-grouped to suffix sessions (pp)"); M(f"S{k}FileToSuffixLo", lo, "CI low"); M(f"S{k}FileToSuffixHi", hi, "CI high")
# leave-one-class-out under P*: which single class decides the sign?
loco = {}
for i, cls in enumerate(classes):
    loco[cls] = {k: [round(x, 2) for x in boot_delta(runs_of(k, "Pstar"), runs_of("RF", "Pstar"), seed=1, drop=i)] for k in ("Fu", "Cn")}
facts["loco_Pstar_deep_minus_RF"] = loco
excl = [cls for cls, v in loco.items() if any(x[1] > 0 or x[2] < 0 for x in v.values())]
facts["loco_Pstar_classes_whose_removal_excludes_zero"] = excl
M("NLocoDecisive", len(excl), "classes whose removal makes a P* deep-minus-RF CI exclude zero", "{:d}")
assert "email" in excl, excl
for cls, nm in (("vpn_file_transfer", "Vft"),):
    for k in ("Fu", "Cn"):
        pt, lo, hi = loco[cls][k]
        M(f"Loco{nm}{k}", pt, f"P*: {MN[k]} minus RF with the {cls} term removed (pp)"); M(f"Loco{nm}{k}Lo", lo, "CI low"); M(f"Loco{nm}{k}Hi", hi, "CI high")
if SPLITTER == "fixed": assert excl == ["email", "vpn_file_transfer"] and loco["vpn_file_transfer"]["Fu"][2] > 0 > loco["vpn_file_transfer"]["Cn"][2], loco
facts["n_boot"] = NBOOT
facts["loco_mode"] = dict(strat=BOOT_STRAT, drop=DROP_MODE)
# per-class F1 of email for every model under P*
for k, nm in (("RF", "RF"), ("Fu", "Fu"), ("Cn", "Cn")):
    M(f"PcEmail{nm}P", float(np.mean([per_class_f1(r["t"], r["p"], len(classes))[EMAIL] for r in runs_of(k, "Pstar")])) * 100, f"{MN[k]} F1 of email under P* (%)")
# cluster-unit sensitivity (suffix-session clusters, as in the pre-G04 draft): facts only
facts["suffix_cluster_sensitivity"] = {
    "RF file-grouped minus P*": [round(x, 2) for x in boot_delta(runs_of("RF", "no_session"), runs_of("RF", "Pstar"), cluster="suffix")],
    "Fu file-grouped minus P*": [round(x, 2) for x in boot_delta(runs_of("Fu", "no_session"), runs_of("Fu", "Pstar"), cluster="suffix")],
    "Cn file-grouped minus P*": [round(x, 2) for x in boot_delta(runs_of("Cn", "no_session"), runs_of("Cn", "Pstar"), cluster="suffix")],
    "Fu minus RF, P*": [round(x, 2) for x in boot_delta(runs_of("Fu", "Pstar"), runs_of("RF", "Pstar"), cluster="suffix")],
    "Cn minus RF, P*": [round(x, 2) for x in boot_delta(runs_of("Cn", "Pstar"), runs_of("RF", "Pstar"), cluster="suffix")],
}

# ---------------------------------------------------------------- other single factors (30 epochs)
FAC = (("no_clean", "No background cleaning"), ("no_sni", "No SNI zeroing"), ("no_norm", "No standardisation"), ("interaction", "File grouping $+$ no cleaning"))
L = ["\\begin{tabular}{lccc}", "\\toprule", "Change from P* & RF (len+IAT) & Fusion (30 ep.) & Bytes CNN (30 ep.) \\\\", "\\midrule"]
for c, lab in FAC:
    cells = []
    for m in ("RF_seq_iat", "11", "7"):
        if SPLITTER == "fixed" and (c, m) in (("no_sni", "RF_seq_iat"), ("no_norm", "RF_seq_iat"), ("no_norm", "7")):
            cells.append("n/a$^\\S$"); continue  # input unaffected by the change (shown n/a in v2 as well); not rerun in G05
        A = [g1any(c, "multi", m, s) for s in SEEDS]
        if any(a is None for a in A): cells.append("--"); continue
        d = [100 * (mf1(a["t"], a["p"]) - mf1(b["t"], b["p"])) for a, b in zip(A, [g1("A", "Pstar", "multi", m, s) for s in SEEDS])]
        same = all((a["p"] == b["p"]).all() and (a["ids"] == b["ids"]).all() for a, b in zip(A, [g1("A", "Pstar", "multi", m, s) for s in SEEDS]) if len(a["ids"]) == len(b["ids"]))
        cells.append("n/a$^\\S$" if same and max(abs(x) for x in d) == 0 else f"{sg(np.mean(d))} [{sg(min(d))}, {sg(max(d))}]")
        facts[f"factor_{c}_{m}"] = [round(x, 2) for x in d]
    L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{OUT}/tab_factors.tex", "w").write("\n".join(L) + "\n")
if SPLITTER == "old":  # G05 reran no_norm for Fusion only
    for m in ("1", "20", "2", "3", "8", "11"):
        A = [g1any("no_norm", "multi", m, s) for s in SEEDS]; Bm = [g1("A", "Pstar", "multi", m, s) for s in SEEDS]
        facts[f"factor_no_norm_{m}"] = [round(100 * (mf1(a["t"], a["p"]) - mf1(b["t"], b["p"])), 2) for a, b in zip(A, Bm)]
    nn = {m: float(np.mean(facts[f"factor_no_norm_{m}"])) for m in ("1", "20", "2", "3", "8", "11")}
    M("FNormDeepMin", min(nn.values()), "smallest no-standardisation effect over deep sequence models (pp, 30 epochs)")
    M("FNormDeepMax", max(nn.values()), "largest no-standardisation effect over deep sequence models (pp, 30 epochs)")
    facts["no_norm_deep_means"] = {NAMES[k]: round(v, 2) for k, v in nn.items()}
# RF uses neither bytes nor standardisation, so no_sni / no_norm leave it unchanged by construction: no RF macros for those.
for c, nm in (("no_clean", "Clean"), ("interaction", "Inter")):
    d = facts[f"factor_{c}_RF_seq_iat"]; M(f"F{nm}RF", float(np.mean(d)), f"RF: {c} minus P* (pp, mean of seeds)")
    M(f"F{nm}RFLo", min(d), "lowest seed"); M(f"F{nm}RFHi", max(d), "highest seed")
d = facts["factor_no_norm_11"]; M("FNormFu", float(np.mean(d)), "Fusion at 30 epochs: no_norm minus P* (pp, mean of seeds)")
assert max(d) < 0, d; M("FNormFuAbs", abs(float(np.mean(d))), "absolute value of the above (all seeds negative)")
for c, nm in (("no_clean", "Clean"), ("no_sni", "SNI")):
    for m, mn in (("11", "Fu"), ("7", "Bytes")):
        d = facts[f"factor_{c}_{m}"]; M(f"F{nm}{mn}", float(np.mean(d)), f"{NAMES[m]} at 30 epochs: {c} minus P* (pp, mean of seeds)")

# ---------------------------------------------------------------- near-duplicate exposure (RF fit sets)
exs = [x for x in json.load(open(f"{AUD}/near_duplicate_exposure_summary.json")) if x["class"] == "__all__"]
EX = {(x["condition"], x["definition"]): x for x in exs}
# evidence/1 was coverage-repaired: replace its per-seed entry by the G04 exposure of the new split, re-average
ex4 = {x["definition"]: x["per_seed"][0] for x in json.load(open(f"{X4}/audit/near_duplicate_exposure_summary.json")) if x["class"] == "__all__"}
for dfn in ("same_connection", "same_sequence"):
    e = EX[("evidence", dfn)]; ps = [ex4[dfn] if q["seed"] == 1 else q for q in e["per_seed"]]
    assert sorted(q["seed"] for q in ps) == [1, 7, 42] and ex4[dfn]["seed"] == 1
    EX[("evidence", dfn)] = {k: {"mean": float(np.mean([q[k] for q in ps]))} for k in ("any_exposure_rate", "exposed_accuracy", "unexposed_accuracy")}
if SPLITTER == "fixed":  # grouped splits: exposure of the G05 RF runs (flow-random stays G01)
    ex5 = [x for x in json.load(open(f"{X5}/audit/near_duplicate_exposure_summary.json")) if x["class"] == "__all__"]
    for x in ex5:
        if x["condition"] in ("no_session", "Pstar", "evidence") and x["definition"] in ("same_connection", "same_sequence"):
            ps = x["per_seed"]; assert sorted(q["seed"] for q in ps) == [1, 7, 42]
            EX[(x["condition"], x["definition"])] = {k: {"mean": float(np.mean([q[k] for q in ps]))} for k in ("any_exposure_rate", "exposed_accuracy", "unexposed_accuracy")}
L = ["\\begin{tabular}{lccc}", "\\toprule", "Split & Shared five-tuple & Same length sequence & Acc.\\ exposed / not exposed$^\\ddagger$ \\\\", "\\midrule"]
for c, lab in LAD:
    a = EX[(c, "same_connection")]; b = EX[(c, "same_sequence")]
    L.append(f"{lab} & {100*a['any_exposure_rate']['mean']:.1f}\\% & {100*b['any_exposure_rate']['mean']:.1f}\\% & {100*a['exposed_accuracy']['mean']:.1f} / {100*a['unexposed_accuracy']['mean']:.1f} \\\\")
    M(f"Exp{CN[c]}", 100 * a["any_exposure_rate"]["mean"], f"share of test flows whose connection also appears in the fit set, {c} (%)")
    M(f"ExpSeq{CN[c]}", 100 * b["any_exposure_rate"]["mean"], f"share of test flows with an identical 50-length sequence in the fit set, {c} (%)")
    M(f"ExpAcc{CN[c]}", 100 * a["exposed_accuracy"]["mean"], f"RF accuracy on connection-exposed test flows, {c} (%)")
    M(f"UnexpAcc{CN[c]}", 100 * a["unexposed_accuracy"]["mean"], f"RF accuracy on non-exposed test flows, {c} (%)")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{OUT}/tab_exposure.tex", "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- per-class F1, RF, four splits
L = ["\\begin{tabular}{l" + "c" * 4 + "}", "\\toprule", "Class & " + " & ".join(l for _, l in LAD) + " \\\\", "\\midrule"]
pc = {}
for c, _ in LAD:
    pc[c] = np.mean([per_class_f1(r["t"], r["p"], len(classes)) for r in [g1any(c, "multi", "RF_seq_iat", s) for s in SEEDS]], axis=0) * 100
for i, cls in enumerate(classes):
    L.append(cls.replace("_", "\\_") + " & " + " & ".join(f"{pc[c][i]:.1f}" for c, _ in LAD) + " \\\\")
    facts[f"perclass_RF_{cls}"] = {c: round(float(pc[c][i]), 1) for c, _ in LAD}
    if cls in ("email", "vpn_file_transfer"):
        nm = {"email": "Email", "vpn_file_transfer": "VpnFt"}[cls]
        M(f"Pc{nm}Flow", pc["leak"][i], f"RF per-class F1 of {cls}, flow-random (%)")
        M(f"Pc{nm}Suffix", pc["Pstar"][i], f"RF per-class F1 of {cls}, suffix sessions P* (%)")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{OUT}/tab_perclass.tex", "w").write("\n".join(L) + "\n")

M("FInterExcessRF", float(np.mean(facts["factor_interaction_RF_seq_iat"])) - ladder[("RF", "no_session")][2] - float(np.mean(facts["factor_no_clean_RF_seq_iat"])),
  "RF: interaction effect minus the sum of the two single-factor effects (file grouping, no cleaning) (pp); non-zero = not additive")
# ---------------------------------------------------------------- G03 (post hoc broader merges), verified by check_g03.py
G3 = json.load(open(f"{OUT}/g03.json"))
assert all(G3["checks"].values()), "check_g03.py reported a failed check"
M("GTwoGroups", G3["E2"]["groups"], "E2 groups on all files (evidence + SCI pairs that overlap in time)", "{:d}")
M("GThreeGroups", G3["E3"]["groups"], "E3 groups on all files (E2 + SCI pairs separate <= 1 h with >= 1 common TCP 5-tuple)", "{:d}")
M("GThreeStream", G3["E3"]["per_class"]["streaming"], "E3 groups in class streaming", "{:d}")
M("GThreeVpnStream", G3["E3"]["per_class"]["vpn_streaming"], "E3 groups in class vpn_streaming", "{:d}")
E2R = {"RF": [load(f"{R3}/B_baseline_E2_multi_RF_seq_iat_{s}_predictions.csv.gz") for s in SEEDS],
       "Fu": [load(f"{R3}/G03_E2_multi_11_{s}_predictions.csv.gz") for s in SEEDS],
       "Cn": [load(f"{R3}/G03_E2_multi_8_{s}_predictions.csv.gz") for s in SEEDS]}
for k in ("RF", "Fu", "Cn"):
    M(f"GTwo{k}", stats(E2R[k])[0], f"{MN[k]} macro-F1 under E2, 9-class (%)")
    for c, suf in (("Pstar", "D"), ("evidence", "DEv")):
        pt, lo, hi = boot_delta(E2R[k], runs_of(k, c), seed=3)
        M(f"GTwo{suf}{k}", pt, f"{MN[k]}: E2 minus {c} (pp)"); M(f"GTwo{suf}{k}Lo", lo, "CI low"); M(f"GTwo{suf}{k}Hi", hi, "CI high")
        if SPLITTER == "fixed" and suf == "D" and k in ("RF", "Fu"):  # prose: "E2 lowers RF / Fusion by"
            assert hi < 0, (k, pt, lo, hi); M(f"GTwoD{k}Abs", abs(pt), "absolute value of the above")
for k in ("Fu", "Cn"):
    for drop, sfx in ((None, ""), (EMAIL, "NE")):
        pt, lo, hi = boot_delta(E2R[k], E2R["RF"], seed=1, drop=drop)
        M(f"GTwoR{k}{sfx}", pt, f"{MN[k]} minus RF under E2{'' if drop is None else ', email removed'} (pp)"); M(f"GTwoR{k}{sfx}Lo", lo, "CI low"); M(f"GTwoR{k}{sfx}Hi", hi, "CI high")
    pt, lo, hi = bexpr([(1, E2R[k]), (-1, E2R["RF"]), (-1, runs_of(k, "leak")), (1, runs_of("RF", "leak"))], seed=2)
    M(f"SE{k}", pt, f"change of ({MN[k]} minus RF) from flow-random to E2 (pp)"); M(f"SE{k}Lo", lo, "CI low"); M(f"SE{k}Hi", hi, "CI high")
cap = 0
for m in ("8", "11"):
    for s in SEEDS:
        f = f"{R3}/G03_E2_multi_{m}_{s}"; f = REDIRECT.get(f, f) + ".json"; guard(f)
        cap += sum(fd["stop_epoch"] >= 300 for fd in json.load(open(f))["fold_details"])
M("GTwoCapped", cap, "E2 early-stopping folds that reached the 300-epoch cap", "{:d}")
P8 = [load(f"{R3}/B_baseline_Pstar_8_multi_RF_seq_iat_{s}_predictions.csv.gz") for s in SEEDS]
E8 = [load(f"{R3}/B_baseline_E3_8_multi_RF_seq_iat_{s}_predictions.csv.gz") for s in SEEDS]
M("GEightP", stats(P8)[0], "RF macro-F1, 8-class task (no vpn_streaming), suffix sessions (%)")
M("GEightE", stats(E8)[0], "RF macro-F1, 8-class task, E3 groups (%)")
pt, lo, hi = boot_delta(E8, P8, seed=4)
M("GEightD", pt, "RF: E3_8 minus Pstar_8 (pp)"); M("GEightDLo", lo, "CI low"); M("GEightDHi", hi, "CI high")
cl8 = ["chat", "email", "file_transfer", "streaming", "voip", "vpn_chat", "vpn_file_transfer", "vpn_voip"]
assert sorted(np.unique(P8[0]["t"])) == list(range(8))
pP = dict(zip(cl8, np.round(100 * np.mean([per_class_f1(r["t"], r["p"], 8) for r in P8], axis=0), 1).tolist()))
pE = dict(zip(cl8, np.round(100 * np.mean([per_class_f1(r["t"], r["p"], 8) for r in E8], axis=0), 1).tolist()))
M("GEightStreamP", pP["streaming"], "RF F1 of streaming, 8-class, suffix sessions (%)")
M("GEightStreamE", pE["streaming"], "RF F1 of streaming, 8-class, E3 (%)")
M("GEightOtherFall", sum(pE[c] < pP[c] for c in cl8 if c != "streaming"), "classes other than streaming whose RF F1 is lower under E3_8", "{:d}")
M("GEightOther", len(cl8) - 1, "classes other than streaming in the 8-class task", "{:d}")
facts["g03_perclass_8"] = {"Pstar_8": pP, "E3_8": pE}

# ---------------------------------------------------------------- sessions per class (9-class task)
ev9 = runs_of("RF", "evidence")[0]; assert (ev9["ids"] == ref["ids"]).all()
sess = {}
for i, cls in enumerate(classes):
    m = ref["t"] == i
    sess[cls] = dict(flows=int(m.sum()), suffix=len(np.unique(ref["grp"][m])), evidence=len(np.unique(ev9["grp"][m])), e3=len({E3MAP[f] for f in ref["file"][m]}))
ABBR = {"chat": "Chat", "email": "Email", "file_transfer": "FT", "streaming": "Stream", "voip": "VoIP",
        "vpn_chat": "V-Chat", "vpn_file_transfer": "V-FT", "vpn_streaming": "V-Stream", "vpn_voip": "V-VoIP"}
L = ["\\begin{tabular}{l" + "r" * (len(classes) + 1) + "}", "\\toprule", " & " + " & ".join(ABBR[c] for c in classes) + " & Total \\\\", "\\midrule"]
for k, lab in (("flows", "Flows"), ("suffix", "Suffix sess."), ("evidence", "Evidence sess."), ("e3", "E3 groups")):
    L.append(f"{lab} & " + " & ".join(str(sess[c][k]) for c in classes) + f" & {sum(v[k] for v in sess.values())} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{OUT}/tab_sessions.tex", "w").write("\n".join(L) + "\n")
facts["sessions_per_class"] = sess
M("NSessVft", sess["vpn_file_transfer"]["suffix"], "suffix sessions of vpn_file_transfer", "{:d}")
M("NSessMin", min(v["suffix"] for v in sess.values()), "fewest suffix sessions of any class", "{:d}")
M("NClassesFewSess", sum(v["suffix"] <= 3 for v in sess.values()), "classes with at most 3 suffix sessions", "{:d}")
M("NEThreeMulti", len({E3MAP[f] for f in ref["file"]}), "E3 groups in the 9-class task (bootstrap clusters)", "{:d}")
M("NClassesFewEThree", sum(v["e3"] <= 4 for v in sess.values()), "classes with at most 4 E3 groups (bootstrap strata with few clusters)", "{:d}")
assert max(v["e3"] for v in sess.values() if v["e3"] <= 4) == 4 and min(v["e3"] for v in sess.values()) == 1

# ---------------------------------------------------------------- split repair and the scikit-learn splitter
rep4 = json.load(open(f"{X4}/split_repair.json"))
M("NCondSeeds", len(rep4["checks"]), "condition x seed combinations checked for class coverage", "{:d}")
M("NRepaired", sum(c["r"] > 0 for c in rep4["checks"]), "of them whose outer split left a class without training data (repaired)", "{:d}")
assert sum(c["r"] > 0 for c in rep4["checks"]) == 5
unused = sorted(set(REDIRECT) - USED_REDIRECT)
facts["runs_redirected_but_unused"] = [os.path.basename(u) for u in unused]
if unused: print("redirected runs not used:", [os.path.basename(u) for u in unused])
facts["splitter"] = SPLITTER
sb = json.load(open(f"{OUT}/split_bug.json"))
M("SplitBugMissOld", 100 * sb["Pstar|1.1.3"]["miss_share"], "share of 200 random_states for which scikit-learn 1.1.3 leaves a class untrained in some P* fold (%)", "{:.0f}")
M("SplitBugMissNew", 100 * sb["Pstar|fixed"]["miss_share"], "same with the fixed splitter (scikit-learn >= 1.8) (%)", "{:.0f}")
M("SplitBugStates", sb["Pstar|1.1.3"]["states"], "random_states tried", "{:d}")

# ---------------------------------------------------------------- old (scikit-learn 1.1.3) vs corrected splitter
def coassign(runs):  # mean over seed pairs of the share of session pairs whose "same test fold" status agrees
    S = []
    for r in runs:
        m = {}
        for g, f in zip(r["grp"], r["fold"]): assert m.setdefault(g, f) == f, "session split across test folds"
        k = sorted(m); f = np.array([m[x] for x in k]); iu = np.triu_indices(len(k), 1)
        S.append((k, (f[:, None] == f[None, :])[iu]))
    assert all(a[0] == S[0][0] for a in S)
    return float(np.mean([np.mean(S[i][1] == S[j][1]) for i in range(len(S)) for j in range(i + 1, len(S))]))
if SPLITTER == "fixed":
    oldP = [load(f"{R1}/A_baseline_Pstar_multi_RF_seq_iat_{s}_predictions.csv.gz", raw=True) for s in SEEDS]
    oldF = [load(f"{R1}/B_baseline_no_session_multi_RF_seq_iat_{s}_predictions.csv.gz", raw=True) for s in SEEDS]
    M("OldPRF", stats(oldP)[0], "RF macro-F1 under P* with the scikit-learn 1.1.3 splitter (first runs, %)")
    M("OldFileRF", stats(oldF)[0], "RF macro-F1, file-grouped, scikit-learn 1.1.3 splitter (first runs, %)")
    M("CoAssignOld", coassign(oldP), "P*: agreement of session co-assignment to test folds across seeds, 1.1.3 splitter", "{:.2f}")
    M("CoAssignFixed", coassign(rfP), "same with the corrected splitter", "{:.2f}")
    M("CoAssignRandom", 1 - 2 * 0.2 * 0.8, "expected agreement for independent random 5-fold assignment", "{:.2f}")
_, _, pt, lo, hi = ladder[("RF", "evidence")]
if SPLITTER == "fixed": assert hi < 0, (pt, lo, hi)
M("DRFEvidAbs", abs(pt), "RF: evidence minus P*, absolute value (pp; prose: 'lowers RF by a further')")
_, _, pt, lo, hi = ladder[("Fu", "evidence")]
if SPLITTER == "fixed": assert hi < 0 and NBOOT >= 10000, (pt, lo, hi)  # borderline: excludes zero only with 10,000 resamples
M("DFuEvidAbs", abs(pt), "Fusion: evidence minus P*, absolute value (pp)")
# ---------------------------------------------------------------- G06 (after the third review): RF on all training flows; identical-frame timing
G6 = json.load(open(f"{OUT}/g06.json")); assert G6["n_boot"] == NBOOT and all(G6["checks"].values()), "check_g06.py reported a failed check"
S6 = G6["summary"]
M("RFFullFlow", S6["leak"]["rf_full_mean"], "RF trained on all training flows (validation sessions included), flow-random, macro-F1 (%)")
pt, lo, hi = S6["leak"]["full_minus_orig"]; assert lo > 0
M("RFFullGainFlow", pt, "RF all-training-flows minus RF fit set, flow-random (pp)"); M("RFFullGainFlowLo", lo, "CI low"); M("RFFullGainFlowHi", hi, "CI high")
gs = [S6[c]["full_minus_orig"][0] for c in ("Pstar", "evidence", "E2")]
M("RFFullGainSessMin", min(gs), "RF all-training-flows minus RF fit set, smallest over suffix, evidence, E2 sessions (pp)")
M("RFFullGainSessMax", max(gs), "same, largest (pp)")
for c in ("Pstar", "evidence", "E2"):  # prose: with RF on all training flows, the session-split conclusions are unchanged
    for k in ("Fu", "Cn"):
        a = S6[c][f"{k}_minus_full"]; b = S6[c][f"{k}_minus_full_noemail"]
        assert a[1] < 0 < a[2] and b[2] < 0, (c, k, a, b)
for k, nm in (("Fu", "SFuFileFull"), ("Cn", "SCnFileFull")):
    pt, lo, hi = S6["no_session"][f"{k}_change_vs_flow_full"]; assert lo > 0, (k, pt, lo, hi)
    M(nm, pt, f"{k}: change of deep-minus-RF(all training flows), flow-random -> file-grouped (pp)")
    M(nm + "Lo", lo, "CI low"); M(nm + "Hi", hi, "CI high")
    pt, lo, hi = S6["no_session"][f"{k}_minus_full"]; assert hi < 0
    M(("RFuFileFullAbs", "RCnFileFullAbs")[k == "Cn"], abs(pt), f"RF(all training flows) lead over {k}, file-grouped (pp)")
B6 = G6["B"]; assert B6["n_pairs"] == int(facts["NSuffixIdent"]["value"]) and B6["dt_zero_pairs"] == 0
M("BMedDtMin", B6["median_abs_dt_min_s"], "identical frames: smallest per-pair median |ts_b - ts_a| (s)")
M("BMedDtMax", B6["median_abs_dt_max_s"], "largest per-pair median |ts_b - ts_a| (s)")
M("BSpreadMax", B6["spread_q05_q95_max_ms"], "largest per-pair 5th-95th percentile spread of ts_b - ts_a (ms)", "{:.0f}")
M("BInvMax", B6["inversion_max_pct"], "largest per-pair share of order inversions among matched frame pairs (%)", "{:.2f}")
M("BDumpcap", B6["n_dumpcap_b_pcap_a"], "pairs whose b file is pcapng written by Dumpcap and whose a file is pcap", "{:d}")
assert B6["dumpcap_a_snaplens"] == [65535]
M("BDiffer", B6["n_format_or_snaplen_differ"], "pairs whose two files differ in container format or snapshot length", "{:d}")
open(f"{OUT}/numbers.tex", "w").write("% generated by build/make_numbers.py -- do not edit\n" + "\n".join(macros) + "\n")
json.dump(facts, open(f"{OUT}/facts.json", "w"), indent=1, ensure_ascii=False)
print(len(macros), "macros;", "tables:", sorted(os.path.basename(p) for p in glob.glob(f"{OUT}/tab_*.tex")))
