#!/usr/bin/env python3
"""G18 R3: Matched-frame timing data (13 G06 pairs) and per-second timeline (33 sibling pairs).

Uses unmodified G06 scripts/run_b.py (file_packets) and pcap_stream.py (Capture).
Verifies every pair against G06 audit/packet_pairs/pair_*.json (count exact, 5 quantiles <= 1e-6 s).
"""
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

WORK_G01 = Path("/private/ch/G18_r2r3_20261121/work/G01")
WORK_G06 = Path("/private/ch/G18_r2r3_20261121/work/G06")
ORIG_G01 = Path("/private/ch/G18_prep/g18_r2r3_inputs_20261121/gpu_returns/G01_final/G01_return")
ORIG_G06 = Path("/private/ch/G18_prep/g18_r2r3_inputs_20261121/gpu_returns/G06/G06_return")
OUT_DIR = Path("/private/ch/G18_r2r3_20261121/work/r3_out")

os.environ["G06_G01_ROOT"] = str(WORK_G01)
sys.path.insert(0, str(WORK_G06 / "scripts"))
import run_b  # noqa: E402
from common import dump, read, sha  # noqa: E402
from pcap_stream import Capture  # noqa: E402

QUANT_KEYS = ["min", "q05", "median", "q95", "max"]
QUANT_PROBS = [0.0, 0.05, 0.5, 0.95, 1.0]
TOL_SECONDS = 1e-6


def verify_untouched_g06():
    for rel in ["scripts/run_b.py", "scripts/common.py", "scripts/pcap_stream.py", "config/pairs.json"]:
        assert sha(WORK_G06 / rel) == sha(ORIG_G06 / rel), f"G06 file modified: {rel}"

    # Verify verbatim matching lines 106-113 from run_b.py
    orig_lines = (ORIG_G06 / "scripts/run_b.py").read_text().splitlines()
    # lines 106-112 in 1-indexed = [105:112]
    expected_block = "\n".join(orig_lines[105:112])
    self_text = Path(__file__).read_text()
    start_tag = "# --- BEGIN VERBATIM COPY from G06 run_b.py lines 106-112 ---\n"
    end_tag = "\n # --- END VERBATIM COPY from G06 run_b.py lines 106-112 ---"
    i = self_text.index(start_tag) + len(start_tag)
    j = self_text.index(end_tag, i)
    assert self_text[i:j] == expected_block, "Verbatim G06 pairing block mismatch!"


def process_one_frame_pair(args):
 index, a, b, rawroot_str, exp_sha_a, exp_sha_b, first_ts_a, expected_g01_count, ref_path_str, out_dir_str = args
 pairid = f"pair_{index + 1:02d}"
 rawroot = Path(rawroot_str)
 out_dir = Path(out_dir_str)
 ref = read(ref_path_str)
 assert ref["pair"] == pairid and ref["file_a"] == a and ref["file_b"] == b

 # Call unmodified run_b.file_packets
 x, ma, _ = run_b.file_packets(rawroot / a, exp_sha_a)
 z, mb, _ = run_b.file_packets(rawroot / b, exp_sha_b)
 assert ma["min_timestamp"] == first_ts_a, f"{pairid}: min_timestamp != first_packet_timestamp"

 # --- BEGIN VERBATIM COPY from G06 run_b.py lines 106-112 ---
 hx,ix,cx=np.unique(x['hash'],return_index=True,return_counts=True);hz,iz,cz=np.unique(z['hash'],return_index=True,return_counts=True)
 common,ia,ib=np.intersect1d(hx,hz,assume_unique=True,return_indices=True);n=int(np.minimum(cx[ia],cz[ib]).sum());matched=np.empty(n,dtype=[('ta','<f8'),('ai','<i8'),('tb','<f8'),('bi','<i8')]);cursor=0
 for p,q in zip(ia,ib):
  m=min(int(cx[p]),int(cz[q]));sl=slice(cursor,cursor+m);matched['ta'][sl]=x['time'][ix[p]:ix[p]+m];matched['tb'][sl]=z['time'][iz[q]:iz[q]+m];matched['ai'][sl]=x['index'][ix[p]:ix[p]+m];matched['bi'][sl]=z['index'][iz[q]:iz[q]+m];cursor+=m
 assert cursor==n
 dt=matched['tb']-matched['ta'];ad=np.abs(dt);counts={'dt_eq_0':int(sum(dt==0)),'abs_dt_le_1us':int(sum(ad<=1e-6)),'abs_dt_le_1ms':int(sum(ad<=1e-3)),'abs_dt_le_1s':int(sum(ad<=1)),'abs_dt_gt_1s':int(sum(ad>1))}
 quant=np.quantile(dt,[0,.05,.5,.95,1]).tolist() if n else [None]*5
 # --- END VERBATIM COPY from G06 run_b.py lines 106-112 ---

 # Sort identically to G06 run_b.py line 113 so output rows are ordered in a's frame time
 matched.sort(order=["ta", "ai", "tb", "bi"], kind="stable")
 t_a_rel = matched["ta"] - first_ts_a
 dt_sorted = matched["tb"] - matched["ta"]

 out_path = out_dir / f"G18_frames_{pairid}.csv.gz"
 with gzip.open(out_path, "wt", encoding="utf-8", newline="", compresslevel=9) as f:
  writer = csv.writer(f)
  writer.writerow(["t_a", "dt"])
  for ta_val, dt_val in zip(t_a_rel.tolist(), dt_sorted.tolist()):
   writer.writerow([repr(ta_val), repr(dt_val)])

 # Independent read-back gate verification from the written CSV.GZ
 with gzip.open(out_path, "rt", encoding="utf-8", newline="") as f:
  rdr = csv.reader(f)
  hdr = next(rdr)
  assert hdr == ["t_a", "dt"], f"{pairid}: bad header {hdr}"
  ta_rb = []
  dt_rb = []
  for row in rdr:
   assert len(row) == 2
   ta_rb.append(float(row[0]))
   dt_rb.append(float(row[1]))

 ta_arr = np.array(ta_rb, dtype=np.float64)
 dt_arr = np.array(dt_rb, dtype=np.float64)
 n_rb = len(dt_arr)
 assert n_rb == n
 assert np.isfinite(ta_arr).all() and np.isfinite(dt_arr).all()
 if n_rb > 0:
  assert ta_arr[0] >= 0.0
  assert np.all(np.diff(ta_arr) >= 0.0), f"{pairid}: t_a not monotonic nondecreasing"

 quant_rb = np.quantile(dt_arr, QUANT_PROBS).tolist() if n_rb else [None] * 5
 ref_n = int(ref["matched_packet_count"])
 ref_quant = ref["dt_quantiles_seconds"]

 diffs = {}
 max_abs_diff = 0.0
 for k, qv in zip(QUANT_KEYS, quant_rb):
  d = abs(qv - float(ref_quant[k]))
  diffs[k] = d
  if d > max_abs_diff:
   max_abs_diff = d

 count_ok = (n_rb == ref_n == expected_g01_count)
 quant_ok = (max_abs_diff <= TOL_SECONDS)
 passed = count_ok and quant_ok

 return {
  "pair": pairid,
  "file_a": a,
  "file_b": b,
  "output_file": out_path.name,
  "output_sha256": sha(out_path),
  "output_bytes": out_path.stat().st_size,
  "matched_packet_count": n_rb,
  "ref_matched_packet_count": ref_n,
  "g01_identical_packet_count": expected_g01_count,
  "count_match": count_ok,
  "dt_quantiles_seconds": dict(zip(QUANT_KEYS, quant_rb)),
  "ref_dt_quantiles_seconds": ref_quant,
  "quantile_abs_diffs_seconds": diffs,
  "max_abs_quantile_diff_seconds": max_abs_diff,
  "quantiles_within_1us": quant_ok,
  "passed": passed,
 }


def timeline_one_file(args):
    fname, raw_path_str, pair_start, exp_sha, exp_count, session_name = args
    reader = Capture(raw_path_str)
    counts = Counter()
    total = 0
    try:
        for ts, _frame, _link in reader:
            sec = int(math.floor(ts - pair_start))
            assert sec >= 0, f"{fname}: negative second_since_pair_start={sec} (ts={ts}, pair_start={pair_start})"
            counts[sec] += 1
            total += 1
        digest = reader.sha.hexdigest()
    finally:
        reader.close()
    assert digest == exp_sha, f"{fname}: SHA-256 mismatch during timeline extraction"
    assert total == exp_count, f"{fname}: packet_count mismatch {total} != {exp_count}"
    min_sec = min(counts)
    max_sec = max(counts)
    return fname, session_name, min_sec, max_sec, total, dict(counts)


def main():
    t0 = time.time()
    verify_untouched_g06()
    print("[R3] Untouched G06 scripts & verbatim pairing block check PASSED.", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = {r["file"]: r for r in read(ORIG_G01 / "audit/v3_cleaning_inventory.json")["files"]}
    with (ORIG_G01 / "audit/sibling_pairs.csv").open(newline="") as f:
        sib_rows = list(csv.DictReader(f))
    rawroot = Path(read(WORK_G01 / "config/paths.local.json")["muse_root"]) / "data/ISCXVPN2016"

    # --- Part A: 13 G06 frame pairs ---
    pairs = read(WORK_G06 / "config/pairs.json")
    assert len(pairs) == 13
    tasks = []
    for idx, (a, b) in enumerate(pairs):
        orig_row = next(r for r in sib_rows if {r["file_a"], r["file_b"]} == {a, b})
        tasks.append((
            idx,
            a,
            b,
            str(rawroot),
            inventory[a]["input_sha256"],
            inventory[b]["input_sha256"],
            inventory[a]["first_packet_timestamp"],
            int(orig_row["identical_packet_count"]),
            str(ORIG_G06 / "audit/packet_pairs" / f"pair_{idx + 1:02d}.json"),
            str(OUT_DIR),
        ))

    t_a = time.time()
    print("[R3] Part A: generating G18_frames_pair_01..13.csv.gz (max_workers=8)...", flush=True)
    pair_results = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(process_one_frame_pair, t) for t in tasks]
        for fut in as_completed(futs):
            res = fut.result()
            pair_results.append(res)
            print(
                f"[R3] {res['pair']} ({res['file_a']} vs {res['file_b']}): "
                f"n={res['matched_packet_count']}, max_q_diff={res['max_abs_quantile_diff_seconds']:.3e}s, "
                f"passed={res['passed']}",
                flush=True,
            )
    pair_results.sort(key=lambda d: d["pair"])
    assert len(pair_results) == 13
    all_pairs_passed = all(r["passed"] for r in pair_results)
    print(f"[R3] Part A finished in {time.time() - t_a:.1f}s; gate passed={all_pairs_passed}.", flush=True)
    if not all_pairs_passed:
        dump(OUT_DIR / "r3_summary.json", {"status": "failed", "pairs": pair_results})
        print("[R3] GATE FAILED on one or more of the 13 pairs! Stopping.", flush=True)
        return 1

    # --- Part B: G18_timeline.csv.gz for the 33 suffix sibling pairs (66 files) ---
    t_b = time.time()
    by_session = defaultdict(list)
    with (ORIG_G01 / "audit/file_to_suffix_session.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            by_session[r["suffix_session"]].append(r["file"])
    suffix_pairs = sorted((s, sorted(fs)) for s, fs in by_session.items() if len(fs) > 1)
    assert len(suffix_pairs) == 33 and all(len(fs) == 2 for _, fs in suffix_pairs)

    # Cross-check against sibling_pairs.csv rows with 'suffix' in candidate_origins
    sib_suffix_sets = {frozenset([r["file_a"], r["file_b"]]) for r in sib_rows if "suffix" in r["candidate_origins"]}
    f2s_suffix_sets = {frozenset(fs) for _, fs in suffix_pairs}
    assert sib_suffix_sets == f2s_suffix_sets, "33 suffix pairs mismatch between file_to_suffix_session.csv and sibling_pairs.csv!"

    tl_tasks = []
    pair_starts = {}
    for sess, (f1, f2) in suffix_pairs:
        pstart = min(inventory[f1]["min_packet_timestamp"], inventory[f2]["min_packet_timestamp"])
        pair_starts[sess] = pstart
        for fn in (f1, f2):
            tl_tasks.append((
                fn,
                str(rawroot / fn),
                pstart,
                inventory[fn]["input_sha256"],
                inventory[fn]["packet_count"],
                sess,
            ))
    assert len(tl_tasks) == 66

    print("[R3] Part B: computing per-second timelines for 33 sibling pairs (66 files, max_workers=12)...", flush=True)
    file_timelines = {}
    with ProcessPoolExecutor(max_workers=12) as pool:
        futs = [pool.submit(timeline_one_file, t) for t in tl_tasks]
        for fut in as_completed(futs):
            fname, sess, min_sec, max_sec, total, counts = fut.result()
            file_timelines[fname] = (sess, min_sec, max_sec, total, counts)
    assert len(file_timelines) == 66

    tl_path = OUT_DIR / "G18_timeline.csv.gz"
    total_tl_rows = 0
    with gzip.open(tl_path, "wt", encoding="utf-8", newline="", compresslevel=9) as f:
        writer = csv.writer(f)
        writer.writerow(["file", "second_since_pair_start", "packets"])
        for fname in sorted(file_timelines):
            sess, min_sec, max_sec, total, counts = file_timelines[fname]
            for sec in range(min_sec, max_sec + 1):
                writer.writerow([fname, sec, counts.get(sec, 0)])
                total_tl_rows += 1

    # Independent read-back verification of G18_timeline.csv.gz
    rb_file_sums = Counter()
    rb_file_secs = defaultdict(list)
    with gzip.open(tl_path, "rt", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        hdr = next(rdr)
        assert hdr == ["file", "second_since_pair_start", "packets"]
        for row in rdr:
            assert len(row) == 3
            fn, sec_s, pkts_s = row
            sec = int(sec_s)
            pkts = int(pkts_s)
            assert sec >= 0 and pkts >= 0
            rb_file_sums[fn] += pkts
            rb_file_secs[fn].append(sec)

    assert len(rb_file_sums) == 66
    for fn, tot in rb_file_sums.items():
        assert tot == inventory[fn]["packet_count"], f"Timeline packet sum mismatch for {fn}"
        secs = rb_file_secs[fn]
        assert secs == list(range(secs[0], secs[-1] + 1)), f"Non-contiguous seconds for {fn}"
    for sess, (f1, f2) in suffix_pairs:
        assert min(rb_file_secs[f1][0], rb_file_secs[f2][0]) == 0, f"Pair {sess} does not start at second 0"

    print(
        f"[R3] Part B complete in {time.time() - t_b:.1f}s: wrote {tl_path} "
        f"({total_tl_rows:,} rows across 66 files / 33 pairs, {tl_path.stat().st_size:,} bytes).",
        flush=True,
    )

    summary = {
        "status": "passed",
        "elapsed_seconds": round(time.time() - t0, 2),
        "gate_13_pairs_checked": 13,
        "gate_13_pairs_passed": sum(r["passed"] for r in pair_results),
        "max_abs_quantile_diff_across_all_13_pairs_seconds": max(
            r["max_abs_quantile_diff_seconds"] for r in pair_results
        ),
        "pairs": pair_results,
        "timeline": {
            "output_file": tl_path.name,
            "output_sha256": sha(tl_path),
            "output_bytes": tl_path.stat().st_size,
            "sibling_pair_count": 33,
            "file_count": 66,
            "row_count": total_tl_rows,
            "total_packets_summed": sum(rb_file_sums.values()),
            "dense_per_file_span": True,
            "all_file_packet_sums_match_inventory": True,
            "all_pairs_start_at_second_0": True,
        },
    }
    dump(OUT_DIR / "r3_summary.json", summary)
    print("[R3] ALL GATES AND VERIFICATIONS PASSED.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
