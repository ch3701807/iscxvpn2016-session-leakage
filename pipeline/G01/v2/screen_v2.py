"""Screening harness v2 -- grouped (per-source-file) evaluation protocol.

Fixes the two fatal audit findings on batch 1 (see audit/audit_findings.md):
  1. v1's random 80/20 flow split put 99.8% of test flows' source files in
     the training set (same-session leakage). v2 splits by SOURCE FILE:
     whole pcap files go to train or test, never both. Provenance is rebuilt
     from the per-file npz concatenation order (sorted .pcap, then sorted
     .pcapng), the same logic audit A verified (concat == merged).
  2. v1 trained 5 epochs (clearly undertrained; 15 epochs lifted dirs
     2/8/20 by +7/+14/+15pp). v2 trains 20 epochs, seeds {42, 1, 7},
     and reports mean +/- std -- no single-run ranking.
  Feature extraction is v2 (extract_pcap_v2.py): direction-canonicalised
  flow key, 120 s idle timeout, |length| clamped to 1500, raw = concatenated
  payload. Models, optimiser (Adam lr 1e-3), batch size (128) and metrics
  are unchanged from v1 so v1-vs-v2 differences are attributable to the
  protocol + features, not the models.

  Single-file classes (vpn_p2p: only vpn_bittorrent.pcap) are EXCLUDED from
  the main train/test split. With --include-single they are split 80/20
  randomly WITHIN their file (seed s) and reported separately, explicitly
  labelled within-session -- never read as generalisation.
  Train side is capped at 1000 flows/class (after grouping); test side is
  uncapped (all flows from test files).
  Per-epoch test macro-F1 is diagnostic only; the reported score is epoch
  20; no test-based model selection.
  NOTE (SNI): raw still contains TLS ClientHello incl. SNI, as in v1.
  An SNI ablation is a follow-up experiment, not part of v2.

Usage (two 3090s, one call per card per seed; split is shared per call):
  CUDA_VISIBLE_DEVICES=2 python screen_v2.py --data data/iscx_flows_v2.npz \
      --src data/ISCXVPN2016 --per-file data/iscx_v2 \
      --dirs 1 7 8 11 --seed 42 --epochs 20 --out results_v2/seed42_gpu2.json
  CUDA_VISIBLE_DEVICES=2 python screen_v2.py --data data/iscx_vpn_binary_v2.npz \
      --src data/ISCXVPN2016 --per-file data/iscx_v2 \
      --dirs 25 --seed 42 --epochs 20 --out results_v2/vpn_seed42.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import accuracy_and_macro_f1  # noqa: E402

SEQ_LEN = 50
N_BYTES = 256
SHORT_K = 10  # direction 20: first k packets only


# ---------------------------------------------------------------- models
# (identical to v1 screen.py -- models are NOT the thing being fixed)
class SeqCNN(nn.Module):
    """Direction 1: 1D-CNN over signed packet lengths (baseline)."""

    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten())
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        if x.dim() == 2:  # standalone direction 7 passes (B, 256); add channel dim
            x = x.unsqueeze(1)
        return self.head(self.net(x))


class BytesCNN(nn.Module):
    """Direction 7: 1D-CNN over the first N payload bytes."""

    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten())
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        if x.dim() == 2:  # standalone direction 7 passes (B, 256); add channel dim
            x = x.unsqueeze(1)
        return self.head(self.net(x))


class CNNBiLSTM(nn.Module):
    """Direction 2: CNN features + BiLSTM over the packet sequence."""

    def __init__(self, n_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        h = self.cnn(x).transpose(1, 2)  # (B, T, 64)
        _, (hn, _) = self.lstm(h)
        return self.head(torch.cat([hn[0], hn[1]], dim=1))


class TinyTransformer(nn.Module):
    """Direction 3: light Transformer encoder over packet-length sequence."""

    def __init__(self, n_classes, d=64, layers=2):
        super().__init__()
        self.proj = nn.Linear(1, d)
        enc = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=128,
                                         batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        h = self.proj(x.transpose(1, 2))  # (B, T, d)
        return self.head(self.enc(h).mean(dim=1))


class FusionNet(nn.Module):
    """Direction 11: byte CNN + sequence CNN + stats MLP, concatenated."""

    def __init__(self, n_classes, n_stat=6):
        super().__init__()
        self.byte_cnn = BytesCNN(n_classes).net
        self.seq_cnn = SeqCNN(2, n_classes).net
        self.stat_mlp = nn.Sequential(nn.Linear(n_stat, 32), nn.ReLU())
        self.head = nn.Linear(128 + 128 + 32, n_classes)

    def forward(self, seq2, raw, stat):
        return self.head(torch.cat(
            [self.byte_cnn(raw.unsqueeze(1)), self.seq_cnn(seq2),
             self.stat_mlp(stat)], dim=1))


# ------------------------------------------------------------- provenance
def build_row_file(src_dir, per_file_dir, n_total, data):
    """Rebuild per-row source filename from per-file npz concatenation order.

    build_iscx_v2.py concatenates per-file npz in pcaps order: sorted
    .pcap first, then sorted .pcapng (same as v1 / audit A). Verifies the
    concatenation equals the merged arrays; aborts if not.
    Label check: for the multi-class file the per-file labels must match
    exactly. For the VPN-binary file the merged label is 0/1 while per-file
    labels are class ids, so the expected binary label is derived from the
    filename instead (stem startswith "vpn_" -> 1, else 0).
    """
    src = Path(src_dir)
    pcaps = sorted(src.rglob("*.pcap")) + sorted(src.rglob("*.pcapng"))
    per_file_dir = Path(per_file_dir)
    row_file, ranges, counts = [], {}, {}
    lo = 0
    arrays = {k: [] for k in ("seq", "iat", "raw", "stat", "label")}
    expected_binary = []
    for p in pcaps:
        npz = per_file_dir / f"{p.stem}.npz"
        if not npz.exists():
            raise SystemExit(f"missing per-file npz for {p.name}")
        d = np.load(npz)
        c = len(d["label"])
        counts[p.name] = c
        ranges[p.name] = [lo, lo + c]
        row_file.extend([p.name] * c)
        expected_binary.extend([1 if p.stem.startswith("vpn_") else 0] * c)
        for k in arrays:
            arrays[k].append(d[k])
        lo += c
    if lo != n_total:
        raise SystemExit(f"per-file flow total {lo} != merged {n_total}")
    for k in ("seq", "iat", "raw", "stat"):
        cat = np.concatenate(arrays[k])
        if not np.array_equal(cat, data[k]):
            raise SystemExit(f"per-file concat != merged for {k}; "
                             f"provenance unreliable, aborting")
    cat_label = np.concatenate(arrays["label"])
    if np.array_equal(cat_label, data["label"]):
        pass  # multi-class file: exact match
    elif np.array_equal(np.array(expected_binary, dtype=np.int64),
                        data["label"]):
        pass  # binary file: filename-derived 0/1 matches
    else:
        raise SystemExit("per-file concat != merged for label; "
                         "provenance unreliable, aborting")
    return np.array(row_file), ranges, counts


# ------------------------------------------------------------- grouped split
def grouped_split(data, row_file, seed, train_cap=1000, include_single=False):
    """Split whole source files into train/test, per class.

    For every class with >= 2 files: shuffle its files with rng(seed) and
    greedily assign files to train until train flows reach ~80% of the
    class total, always leaving >= 1 file for test. Single-file classes
    are excluded unless include_single, in which case their rows are split
    80/20 randomly WITHIN the file (flagged within_session in the output).
    Train rows are then capped at train_cap per class (rng shuffle, seed);
    test rows are uncapped.
    Returns (train_idx, test_idx, info dict).
    """
    labels = data["label"]
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []
    info = {"seed": int(seed), "train_cap": train_cap,
            "assignment": {}, "excluded_single_file_classes": [],
            "within_session_single_file_classes": []}
    for c in sorted(np.unique(labels)):
        rows_c = np.where(labels == c)[0]
        files = sorted(set(row_file[rows_c].tolist()))
        if len(files) < 2:
            info["excluded_single_file_classes"].append(
                {"class": int(c), "n_files": len(files), "files": files,
                 "n_rows": int(len(rows_c))})
            if include_single:
                # within-file random split; NEVER read as generalisation.
                pool = rows_c.copy()
                rng.shuffle(pool)
                n_tr = int(0.8 * len(pool))
                train_rows.append(pool[:n_tr])
                test_rows.append(pool[n_tr:])
                info["within_session_single_file_classes"].append(
                    {"class": int(c), "files": files,
                     "n_train": int(n_tr), "n_test": int(len(pool) - n_tr)})
            continue
        # flow counts per file for this class
        fcounts = {f: int(((row_file[rows_c] == f)).sum()) for f in files}
        total = len(rows_c)
        order = files.copy()
        rng.shuffle(order)
        tr_files, te_files, tr_n = [], [], 0
        for i, f in enumerate(order):
            # keep >= 1 file for test: never take the last remaining file
            if tr_n < 0.8 * total and i < len(order) - 1:
                tr_files.append(f)
                tr_n += fcounts[f]
            else:
                te_files.append(f)
        tr_pool = rows_c[np.isin(row_file[rows_c], tr_files)]
        te_pool = rows_c[np.isin(row_file[rows_c], te_files)]
        info["assignment"][str(int(c))] = {
            "train_files": tr_files, "test_files": te_files,
            "n_train_before_cap": int(len(tr_pool)),
            "n_test": int(len(te_pool))}
        train_rows.append(tr_pool)
        test_rows.append(te_pool)
    train_idx = np.concatenate(train_rows) if train_rows else np.array([], dtype=int)
    test_idx = np.concatenate(test_rows) if test_rows else np.array([], dtype=int)
    # per-class train cap (after grouping)
    capped = []
    for c in sorted(np.unique(labels[train_idx])) if len(train_idx) else []:
        pool = train_idx[labels[train_idx] == c]
        rng.shuffle(pool)
        capped.append(pool[:train_cap])
    train_idx = np.concatenate(capped) if capped else train_idx
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx, info


# ------------------------------------------------------------- data utils
def build_inputs(data, direction, idx):
    seq = torch.tensor(data["seq"][idx], dtype=torch.float32).unsqueeze(1)
    iat = torch.tensor(data["iat"][idx], dtype=torch.float32).unsqueeze(1)
    raw = torch.tensor(data["raw"][idx], dtype=torch.float32)
    stat = torch.tensor(data["stat"][idx], dtype=torch.float32)
    y = torch.tensor(data["label"][idx], dtype=torch.long)
    if direction == 20:  # short flow: first k packets only
        seq, iat = seq[:, :, :SHORT_K], iat[:, :, :SHORT_K]
    seq2 = torch.cat([seq, iat], dim=1)
    return {"seq": seq, "seq2": seq2, "raw": raw, "stat": stat, "y": y}


def make_model(direction, n_classes, n_stat):
    if direction in (1, 20, 25):
        return SeqCNN(1, n_classes), "seq"
    if direction == 7:
        return BytesCNN(n_classes), "raw"
    if direction == 8:
        return SeqCNN(2, n_classes), "seq2"
    if direction == 11:
        return FusionNet(n_classes, n_stat), "fusion"
    if direction == 2:
        return CNNBiLSTM(n_classes), "seq"
    if direction == 3:
        return TinyTransformer(n_classes), "seq"
    raise ValueError(f"unknown direction {direction}")


def confusion_and_prf1(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    per_class = []
    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        per_class.append({"class": int(c), "support": int(cm[c, :].sum()),
                          "precision": round(prec, 4), "recall": round(rec, 4),
                          "f1": round(f1, 4)})
    return cm, per_class


def run_direction(data, direction, train_idx, test_idx, epochs, device, seed):
    n_classes = int(data["label"].max()) + 1
    model, kind = make_model(direction, n_classes, data["stat"].shape[1])
    model.to(device)
    tr, te = build_inputs(data, direction, train_idx), build_inputs(data, direction, test_idx)
    if kind == "fusion":
        def fwd(b):
            return model(b[0], b[1], b[2])
        tr_tensors = (tr["seq2"], tr["raw"], tr["stat"], tr["y"])
        te_tensors = (te["seq2"], te["raw"], te["stat"], te["y"])
    else:
        key = {"seq": "seq", "seq2": "seq2", "raw": "raw"}[kind]
        fwd = lambda b: model(b[0])  # noqa: E731
        tr_tensors = (tr[key], tr["y"])
        te_tensors = (te[key], te["y"])
    loader = DataLoader(TensorDataset(*tr_tensors), batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        tot_loss, tot_n = 0.0, 0
        for *xb, yb in loader:
            xb = [t.to(device) for t in xb]
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(fwd(xb), yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(yb)
            tot_n += len(yb)
        # diagnostic-only test evaluation (no early stopping / selection);
        # the reported score is the LAST epoch.
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(te["y"]), 512):
                chunk = [t[i:i + 512].to(device) for t in te_tensors[:-1]]
                preds.append(fwd(chunk).argmax(1).cpu().numpy())
        preds = np.concatenate(preds)
        acc, mf1, _ = accuracy_and_macro_f1(te["y"].numpy(), preds, n_classes)
        model.train()
        history.append({"epoch": epoch,
                        "train_loss": round(tot_loss / tot_n, 4),
                        "test_accuracy": round(acc, 4),
                        "test_macro_f1": round(mf1, 4)})
        print(f"  dir {direction} seed {seed} epoch {epoch}/{epochs}: "
              f"train_loss={tot_loss / tot_n:.4f} test_acc={acc:.4f} "
              f"test_macro_f1={mf1:.4f}", flush=True)
    cm, per_class = confusion_and_prf1(te["y"].numpy(), preds, n_classes)
    final = history[-1]
    return {"direction": direction, "seed": seed,
            "accuracy": final["test_accuracy"],
            "macro_f1": final["test_macro_f1"], "n_test": int(len(te["y"])),
            "n_train": int(len(tr["y"])), "history": history,
            "confusion_matrix": cm.tolist(), "per_class": per_class,
            "preds": preds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help=".npz from extract_pcap_v2.py")
    ap.add_argument("--src", required=True, help="dir with the source pcaps")
    ap.add_argument("--per-file", required=True, help="dir with per-file npz")
    ap.add_argument("--dirs", type=int, nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--per-class", type=int, default=1000,
                    help="train-side cap per class")
    ap.add_argument("--include-single", action="store_true",
                    help="include single-file classes via within-file split "
                         "(diagnostic only, flagged in output)")
    ap.add_argument("--classes", default=None,
                    help="classes json (id -> name), for readable per-class output")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = dict(np.load(args.data))
    row_file, ranges, counts = build_row_file(args.src, args.per_file,
                                              len(data["label"]), data)
    train_idx, test_idx, info = grouped_split(
        data, row_file, args.seed, train_cap=args.per_class,
        include_single=args.include_single)
    print(f"grouped split seed {args.seed}: train={len(train_idx)} "
          f"test={len(test_idx)} excluded_single="
          f"{info['excluded_single_file_classes']} "
          f"within_session={info['within_session_single_file_classes']}", flush=True)
    # Remap labels to contiguous ids when classes were excluded (e.g. vpn_p2p
    # id 8 removed from an 11-class file): otherwise the model would keep a
    # dead output for the missing class and macro-F1 would include a
    # guaranteed 0. With --include-single nothing is excluded: no remap.
    class_names = None
    if args.classes:
        classes = json.loads(Path(args.classes).read_text())
        class_names = {int(v): k for k, v in classes.items()}
    excluded = [e["class"] for e in info["excluded_single_file_classes"]]
    if excluded and not args.include_single:
        kept = sorted(set(int(c) for c in np.unique(data["label"])) - set(excluded))
        remap = {old: new for new, old in enumerate(kept)}
        data = dict(data)
        data["label"] = np.array([remap.get(int(l), 0) for l in data["label"]],
                                 dtype=np.int64)
        info["label_remap"] = {str(old): new for old, new in remap.items()}
        if class_names:
            info["remapped_class_names"] = {
                str(new): class_names[old] for old, new in remap.items()}
        print(f"label remap (excluded {excluded}): {info['label_remap']}",
              flush=True)
    elif class_names:
        info["class_names"] = {str(k): v for k, v in class_names.items()}
    results = [run_direction(data, d, train_idx, test_idx, args.epochs,
                             device, args.seed) for d in args.dirs]
    preds_path = args.out.replace(".json", "_preds.npz")
    save = {"train_idx": train_idx, "test_idx": test_idx,
            "y_test": data["label"][test_idx]}
    for r in results:
        save[f"pred_{r['direction']}"] = np.asarray(r.pop("preds"))
    np.savez(preds_path, **save)
    print(f"saved predictions to {preds_path}")
    results.sort(key=lambda r: -r["macro_f1"])
    with open(args.out, "w") as f:
        json.dump({"device": device, "seed": args.seed, "epochs": args.epochs,
                   "per_class_cap": args.per_class,
                   "split": info, "results": results}, f, indent=2)
    for r in results:
        print(f"dir {r['direction']:>2} seed {args.seed}: acc={r['accuracy']:.4f} "
              f"macro-F1={r['macro_f1']:.4f} (n_train={r['n_train']} "
              f"n_test={r['n_test']})")


if __name__ == "__main__":
    main()
