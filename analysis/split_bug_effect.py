"""Effect of the scikit-learn 1.1.3 StratifiedGroupKFold(shuffle=True) bug (fixed in 1.8, PR #32540) on our data:
over 200 random_states, fold-vs-overall class distribution distance and share of states that leave a class with
no outer-training group in some fold, for the 1.1.3 behaviour (re-implemented in check_g04.py) and the fixed one.
Writes outputs/split_bug.json. Run with python -I"""
import os as _os
ROOT = _os.environ["ARTEFACT_ROOT"] if "ARTEFACT_ROOT" in _os.environ else _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import sys, importlib.util
src = open(ROOT + "/analysis/check_g04.py").read().split("CASES = [")[0]
ns = {}; exec(src, ns)
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold as Fixed
R1 = ns["R1"]; raw = ns["raw"]; Old = ns["StratifiedGroupKFold"]
def dev(y, g, S, rs):
    share = np.bincount(y) / len(y); out = []; miss = 0
    for tr, te in S(5, shuffle=True, random_state=rs).split(np.zeros(len(y)), y, g):
        out.append(np.abs(np.bincount(y[te], minlength=len(share)) / len(te) - share).sum() / 2)
        miss += len(set(np.unique(y)) - set(np.unique(y[tr])))
    return np.mean(out), miss
res = {}
for pre, tag in (("A", "Pstar"), ("B", "evidence")):
    o = raw(f"{R1}/{pre}_baseline_{tag}_multi_RF_seq_iat_1_predictions.csv.gz"); y, g = o["t"], o["grp"]
    for nm, S in (("1.1.3", Old), ("fixed", Fixed)):
        d = [dev(y, g, S, rs) for rs in range(200)]
        res[f"{tag}|{nm}"] = dict(tv=round(float(np.mean([x[0] for x in d])), 3), miss_share=round(float(np.mean([x[1] > 0 for x in d])), 3), states=200)
        print(tag, nm, "mean TV distance fold vs overall: %.3f" % np.mean([x[0] for x in d]),
              " share of seeds with a class missing from some outer-train: %.2f" % np.mean([x[1] > 0 for x in d]))
json.dump(res, open(ROOT + "/outputs/split_bug.json", "w"), indent=1)
