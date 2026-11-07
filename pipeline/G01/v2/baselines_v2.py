"""Baselines v2 -- same grouped (per-source-file) split as screen_v2.py,
seed 42, on the v2 features. Reuses the grouped_split + provenance logic
from screen_v2.py (imported, not copied) and the baseline set from audit B.

Multiclass (iscx_flows_v2.npz, vpn_p2p excluded -> 10 classes):
  majority class, stratified random, LogisticRegression and RandomForest
  (300 trees) on stat 6-dim and seq+iat 100-dim.
Binary (iscx_vpn_binary_v2.npz, 2 classes): same set.
Outputs: BASE/results_v2/baselines_seed42.json (+ per-model predictions
are NOT saved; only aggregate + per-class metrics).

Run on the GPU box with the demo venv python:
  python baselines_v2.py --data data/iscx_flows_v2.npz \
      --src data/ISCXVPN2016 --per-file data/iscx_v2 \
      --classes data/iscx_classes_v2.json --out results_v2/baselines_seed42.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import accuracy_and_macro_f1  # noqa: E402
from screen_v2 import build_row_file, grouped_split  # noqa: E402


def confusion(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf1(cm, names):
    out = []
    for c in range(cm.shape[0]):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        out.append({"class": names[c], "support": int(cm[c, :].sum()),
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4)})
    return out


def summarize(name, y_true, y_pred, names):
    n_classes = len(names)
    acc, mf1, _ = accuracy_and_macro_f1(y_true, y_pred, n_classes)
    cm = confusion(y_true, y_pred, n_classes)
    return {"name": name, "accuracy": round(acc, 4), "macro_f1": round(mf1, 4),
            "n_test": int(len(y_true)), "per_class": per_class_prf1(cm, names),
            "confusion_matrix": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--per-file", required=True)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = dict(np.load(args.data))
    row_file, _, _ = build_row_file(args.src, args.per_file,
                                    len(data["label"]), data)
    train_idx, test_idx, info = grouped_split(
        data, row_file, args.seed, train_cap=1000, include_single=False)
    excluded = [e["class"] for e in info["excluded_single_file_classes"]]
    names = None
    if args.classes:
        classes = json.loads(Path(args.classes).read_text())
        names = {int(v): k for k, v in classes.items()}
    y_all = data["label"]
    if excluded:
        kept = sorted(set(int(c) for c in np.unique(y_all)) - set(excluded))
        remap = {old: new for new, old in enumerate(kept)}
        y_all = np.array([remap.get(int(l), 0) for l in y_all], dtype=np.int64)
        names = [names[old] for old in kept]
    else:
        names = [names[c] for c in sorted(names)] if names else \
            [str(c) for c in sorted(np.unique(y_all))]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    n_classes = len(names)
    print(f"grouped split seed {args.seed}: train={len(train_idx)} "
          f"test={len(test_idx)} classes={names}", flush=True)

    out = {"split": info, "classes": names, "n_train": int(len(train_idx)),
           "n_test": int(len(test_idx)), "simple_baselines": {},
           "ml_baselines": {}}

    majority = int(np.bincount(y_train).argmax())
    out["simple_baselines"]["majority_class"] = summarize(
        "majority_class", y_test, np.full_like(y_test, majority), names)
    rng = np.random.default_rng(args.seed)
    probs = np.bincount(y_train, minlength=n_classes) / len(y_train)
    out["simple_baselines"]["stratified_random"] = summarize(
        "stratified_random", y_test,
        rng.choice(n_classes, size=len(y_test), p=probs), names)
    for k, v in out["simple_baselines"].items():
        print(f"{k}: acc={v['accuracy']:.4f} macroF1={v['macro_f1']:.4f}")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats = {
        "stat_6d": (data["stat"][train_idx].astype(np.float64),
                    data["stat"][test_idx].astype(np.float64)),
        "seq_50d": (data["seq"][train_idx].astype(np.float64),
                    data["seq"][test_idx].astype(np.float64)),
        "seq_iat_100d": (np.concatenate([data["seq"][train_idx],
                                         data["iat"][train_idx]], axis=1).astype(np.float64),
                         np.concatenate([data["seq"][test_idx],
                                         data["iat"][test_idx]], axis=1).astype(np.float64)),
    }
    for fname, (Xtr, Xte) in feats.items():
        rf = RandomForestClassifier(n_estimators=300, n_jobs=16,
                                    random_state=args.seed)
        rf.fit(Xtr, y_train)
        s = summarize(f"RandomForest_{fname}", y_test, rf.predict(Xte), names)
        out["ml_baselines"][f"RandomForest_{fname}"] = s
        print(f"RF {fname}: acc={s['accuracy']:.4f} macroF1={s['macro_f1']:.4f}")
        lr = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=1000, C=1.0,
                                              random_state=args.seed))
        lr.fit(Xtr, y_train)
        s = summarize(f"LogisticRegression_{fname}", y_test, lr.predict(Xte),
                      names)
        out["ml_baselines"][f"LogisticRegression_{fname}"] = s
        print(f"LR {fname}: acc={s['accuracy']:.4f} macroF1={s['macro_f1']:.4f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
