"""Pure-numpy metrics for the demo screening harness (no sklearn needed).

accuracy + macro-F1, computed from a confusion matrix so every direction
uses exactly the same code path.
"""
import numpy as np


def accuracy_and_macro_f1(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    acc = float(cm.diagonal().sum() / max(cm.sum(), 1))
    f1s = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
    return acc, float(np.mean(f1s)), f1s
