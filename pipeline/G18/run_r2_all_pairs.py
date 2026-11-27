#!/usr/bin/env python3
"""G18 R2: All-pairs (9,730 pairs across 140 captures) sibling audit driver.

Uses G01 sibling_audit.one_file (with only the authorized 3-line v3_files/*.npz edit)
and copies sibling_audit.main lines 140-156 verbatim for pairwise evaluation.
"""
import csv
import gzip
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

WORK_G01 = Path("/private/ch/G18_r2r3_20261121/work/G01")
ORIG_G01 = Path("/private/ch/G18_prep/g18_r2r3_inputs_20261121/gpu_returns/G01_final/G01_return")
OUT_DIR = Path("/private/ch/G18_r2r3_20261121/work/r2_out")

sys.path.insert(0, str(WORK_G01 / "scripts"))
import sibling_audit  # noqa: E402
from sibling_audit import PRIVATE, W, one_file, split_connection  # noqa: E402
from extract_v3 import dump  # noqa: E402

GATE_COLUMNS = [
    "time_relation",
    "gap_seconds",
    "common_tcp_five_tuples",
    "split_tcp_connections",
    "uncertain_tcp_boundaries",
    "common_udp_five_tuples",
    "identical_packet_count",
    "identical_packet_fraction_a",
    "identical_packet_fraction_b",
    "identical_packet_fraction_min",
    "verdict",
    "suspected_duplicate",
]

EMPTY_FEATURE_COLUMNS = [
    "identical_feature_flow_count",
    "same_connection_matched_flows_a",
    "same_connection_matched_flows_b",
    "same_connection_matching_flow_pairs",
    "same_sequence_matched_flows_a",
    "same_sequence_matched_flows_b",
    "same_sequence_matching_flow_pairs",
    "same_sequence_raw_matched_flows_a",
    "same_sequence_raw_matched_flows_b",
    "same_sequence_raw_matching_flow_pairs",
]

# Populated once in parent before forking pair-evaluation workers.
EVIDENCE_CACHE = {}
RECORDS = {}
ORIG_PAIRS = {}


def evidence(name):
    return EVIDENCE_CACHE[name]


def rss_mb():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


def eval_pair_chunk(pair_chunk):
    records = RECORDS
    orig_pairs = ORIG_PAIRS
    out_rows = []
    for n1, n2 in pair_chunk:
        # --- BEGIN VERBATIM COPY from G01 sibling_audit.py lines 140-156 ---
        r1,r2=records[n1],records[n2]
        a,b=(n1,n2) if (r1['min_packet_timestamp'],n1)<=(r2['min_packet_timestamp'],n2) else (n2,n1)
        ra,rb=records[a],records[b]
        overlap=max(ra['min_packet_timestamp'],rb['min_packet_timestamp'])<=min(ra['max_packet_timestamp'],rb['max_packet_timestamp'])
        gap=0. if overlap else max(0.,rb['min_packet_timestamp']-ra['max_packet_timestamp'])
        relation='overlap' if overlap else ('adjacent' if gap<=60 else 'separate')
        ea,eb=evidence(a),evidence(b);ct=set(ea['tcp'])&set(eb['tcp']);cu=ea['udp']&eb['udp'];split=uncertain=0
        if relation=='adjacent':
            for key in ct:
                yes,u=split_connection(ea['tcp'][key],eb['tcp'][key]);split+=int(yes);uncertain+=u
        shared=sum((ea['packet_hashes']&eb['packet_hashes']).values());na=ra['packet_count'];nb=rb['packet_count']
        frac=shared/min(na,nb) if min(na,nb) else 0.;duplicate=min(na,nb)>=100 and frac>=.90
        samefeat=sum((ea['features']&eb['features']).values());common=len(ct)+len(cu)
        if split or relation=='adjacent' and common:verdict='confirmed'
        elif relation in ['adjacent','overlap'] and common==0:verdict='possible'
        elif relation=='separate' and common==0:verdict='no_evidence'
        else:verdict='shared_connection_insufficient'
        # --- END VERBATIM COPY from G01 sibling_audit.py lines 140-156 ---

        assert samefeat == 0, f"Unexpected non-zero samefeat for {a}, {b}"
        in_orig = (n1, n2) in orig_pairs
        same_cls = ra["class"] == rb["class"]
        cls_val = ra["class"] if same_cls else f"{ra['class']}|{rb['class']}"
        orig_val = "+".join(sorted(orig_pairs[(n1, n2)])) if in_orig else "all_pairs"
        row = {
            "file_a": a,
            "file_b": b,
            "class": cls_val,
            "candidate_origins": orig_val,
            "time_relation": relation,
            "gap_seconds": gap,
            "common_tcp_five_tuples": len(ct),
            "split_tcp_connections": split,
            "uncertain_tcp_boundaries": uncertain,
            "common_udp_five_tuples": len(cu),
            "identical_packet_count": shared,
            "identical_packet_fraction_a": shared / na if na else 0.0,
            "identical_packet_fraction_b": shared / nb if nb else 0.0,
            "identical_packet_fraction_min": frac,
            "identical_feature_flow_count": "",
            "verdict": verdict,
            "suspected_duplicate": duplicate,
            "same_connection_matched_flows_a": "",
            "same_connection_matched_flows_b": "",
            "same_connection_matching_flow_pairs": "",
            "same_sequence_matched_flows_a": "",
            "same_sequence_matched_flows_b": "",
            "same_sequence_matching_flow_pairs": "",
            "same_sequence_raw_matched_flows_a": "",
            "same_sequence_raw_matched_flows_b": "",
            "same_sequence_raw_matching_flow_pairs": "",
            "same_class": same_cls,
            "in_original_130": in_orig,
        }
        out_rows.append(row)
    return out_rows


def verify_verbatim_block():
    """Assert that lines 140-156 in sibling_audit.py appear byte-for-byte in this driver."""
    orig_lines = (ORIG_G01 / "scripts/sibling_audit.py").read_text().splitlines()
    expected_block = "\n".join(orig_lines[139:156])
    self_text = Path(__file__).read_text()
    start_tag = "# --- BEGIN VERBATIM COPY from G01 sibling_audit.py lines 140-156 ---\n"
    end_tag = "\n        # --- END VERBATIM COPY from G01 sibling_audit.py lines 140-156 ---"
    i = self_text.index(start_tag) + len(start_tag)
    j = self_text.index(end_tag, i)
    actual_block = self_text[i:j]
    assert actual_block == expected_block, "Verbatim block mismatch against original sibling_audit.py!"


def main():
    t0 = time.time()
    assert W == WORK_G01.resolve(), f"Unexpected W={W}"
    verify_verbatim_block()
    print("[R2] Verbatim pairwise block check PASSED.", flush=True)

    # Verify frozen RULES.md registration & v3 cleaning inventory
    registration = json.loads((W / "audit/rules_registration.json").read_text())
    assert hashlib.sha256((W / "RULES.md").read_bytes()).hexdigest() == registration["sha256"]
    inv = json.loads((W / "audit/v3_cleaning_inventory.json").read_text())
    assert inv["complete"] is True
    records = {r["file"]: r for r in inv["files"]}
    assert len(records) == 140, f"Expected 140 files, got {len(records)}"

    # Load original sibling_pairs.csv reference
    with (ORIG_G01 / "audit/sibling_pairs.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        orig_fieldnames = list(reader.fieldnames)
        orig_rows = list(reader)
    assert len(orig_rows) == 130, f"Expected 130 reference rows, got {len(orig_rows)}"

    # Reconstruct original 130 candidate origins identically to sibling_audit.py lines 123-129
    suffix = defaultdict(list)
    byclass = defaultdict(list)
    orig_pairs = defaultdict(set)
    for name, r in records.items():
        suffix[r["session"]].append(name)
        byclass[r["class"]].append(name)
    for names in suffix.values():
        for a, b in itertools.combinations(sorted(names), 2):
            orig_pairs[(a, b)].add("suffix")
    for names in byclass.values():
        names.sort(key=lambda n: (records[n]["first_packet_timestamp"], n))
        for a, b in zip(names, names[1:]):
            orig_pairs[tuple(sorted([a, b]))].add("adjacent_first_timestamp")
    assert len(orig_pairs) == 130, f"Reconstructed candidate pairs = {len(orig_pairs)} != 130"

    # Generate fresh local random salt (32 bytes, 0600); never included in return package
    saltpath = W / "data/sibling_salt.bin"
    PRIVATE.mkdir(parents=True, exist_ok=True)
    os.chmod(PRIVATE, 0o700)
    if not saltpath.exists():
        saltpath.write_bytes(os.urandom(32))
        saltpath.chmod(0o600)
    muse = json.loads((W / "config/paths.local.json").read_text())["muse_root"]

    # Phase 1: Per-file extraction using sibling_audit.one_file (12 workers <= 16 limit)
    t_p1 = time.time()
    print("[R2] Phase 1: extracting per-file evidence for 140 captures (max_workers=12)...", flush=True)
    failed = []
    file_summaries = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(one_file, (r, muse, str(saltpath))): r for r in records.values()}
        for future in as_completed(futures):
            r = futures[future]
            try:
                x = future.result()
            except Exception as e:
                failed.append({"file": r["file"], "error": str(e)})
                print(f"[R2] ERROR in one_file({r['file']}): {e}", flush=True)
                continue
            file_summaries.append(x)
            if len(file_summaries) % 20 == 0 or len(file_summaries) == 140:
                print(
                    f"[R2] Phase 1 progress: {len(file_summaries)}/140 "
                    f"(last={r['file']}, pkts={x['packet_count']}, elapsed={time.time()-t_p1:.1f}s)",
                    flush=True,
                )

    # Delete local random salt immediately after Phase 1 finishes (Hard Rule 0.4)
    if saltpath.exists():
        saltpath.unlink()
        print("[R2] Local random salt deleted immediately after Phase 1.", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_summaries.sort(key=lambda d: d["file"])
    dump(OUT_DIR / "r2_file_summaries.json", file_summaries)
    if failed:
        dump(OUT_DIR / "r2_failed_files.json", failed)
        print(f"[R2] FATAL: {len(failed)} file(s) failed in Phase 1. Stopping.", flush=True)
        return 2
    assert len(file_summaries) == 140
    print(f"[R2] Phase 1 complete in {time.time()-t_p1:.1f}s.", flush=True)

    # Phase 2: Preload all 140 evidence pickles once into memory
    t_load = time.time()
    global EVIDENCE_CACHE, RECORDS, ORIG_PAIRS
    RECORDS = records
    ORIG_PAIRS = orig_pairs
    for name in sorted(records):
        with gzip.open(PRIVATE / f"{Path(name).stem}.pickle.gz", "rb") as f:
            EVIDENCE_CACHE[name] = pickle.load(f)
    parent_rss = rss_mb()
    print(
        f"[R2] Preloaded 140 evidence caches in {time.time()-t_load:.1f}s; parent RSS={parent_rss:.1f} MiB.",
        flush=True,
    )

    all_pairs = list(itertools.combinations(sorted(records), 2))
    assert len(all_pairs) == 9730, f"Expected 9730 pairs, got {len(all_pairs)}"

    # Pick worker count conservatively so total RAM stays far below 120 GiB and processes <= 16
    if parent_rss > 0:
        max_by_ram = max(1, int(65536.0 / max(parent_rss, 512.0)))
    else:
        max_by_ram = 8
    n_workers = min(12, max_by_ram)
    print(f"[R2] Phase 2: evaluating {len(all_pairs)} pairs with {n_workers} forked workers...", flush=True)
    t_p2 = time.time()

    # Create balanced interleaved chunks so large files are spread evenly across workers
    # Preserve deterministic order by pairing each chunk item with original index or re-sorting
    chunks = [[] for _ in range(n_workers)]
    for idx, pair in enumerate(all_pairs):
        chunks[idx % n_workers].append(pair)

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers) as pool:
        chunk_results = pool.map(eval_pair_chunk, chunks)

    # Reassemble in exact itertools.combinations(sorted(records), 2) order
    rows = [None] * len(all_pairs)
    for w_idx, res_list in enumerate(chunk_results):
        for local_i, row in enumerate(res_list):
            rows[local_i * n_workers + w_idx] = row
    assert all(r is not None for r in rows)
    print(f"[R2] Phase 2 complete in {time.time()-t_p2:.1f}s.", flush=True)

    # Write G18_all_pairs.csv.gz
    fieldnames = orig_fieldnames + ["same_class", "in_original_130"]
    out_csv_gz = OUT_DIR / "G18_all_pairs.csv.gz"
    with gzip.open(out_csv_gz, "wt", encoding="utf-8", newline="", compresslevel=9) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[R2] Wrote {out_csv_gz} ({out_csv_gz.stat().st_size:,} bytes).", flush=True)

    # Phase 3: Strict Gate verification by reading back G18_all_pairs.csv.gz
    with gzip.open(out_csv_gz, "rt", encoding="utf-8", newline="") as f:
        rb_reader = csv.DictReader(f)
        rb_header = list(rb_reader.fieldnames)
        rb_rows = list(rb_reader)

    assert rb_header == fieldnames, f"Header mismatch: {rb_header}"
    assert len(rb_rows) == 9730, f"Row count mismatch: {len(rb_rows)}"

    # Check empty feature columns on ALL 9,730 rows
    non_empty_feature_cells = 0
    for r in rb_rows:
        for col in EMPTY_FEATURE_COLUMNS:
            if r[col] != "":
                non_empty_feature_cells += 1
    assert non_empty_feature_cells == 0, f"Found {non_empty_feature_cells} non-empty feature cells!"

    # Match original 130 pairs
    orig_by_pair = {(r["file_a"], r["file_b"]): r for r in orig_rows}
    new_orig_rows = [r for r in rb_rows if r["in_original_130"] == "True"]
    new_by_pair = {(r["file_a"], r["file_b"]): r for r in new_orig_rows}

    mismatches = []
    if set(orig_by_pair.keys()) != set(new_by_pair.keys()):
        mismatches.append({
            "type": "pair_set_mismatch",
            "only_in_orig": sorted(set(orig_by_pair) - set(new_by_pair)),
            "only_in_new": sorted(set(new_by_pair) - set(orig_by_pair)),
        })
    else:
        for pair_key in sorted(orig_by_pair):
            o = orig_by_pair[pair_key]
            n = new_by_pair[pair_key]
            for col in ["class", "candidate_origins"] + GATE_COLUMNS:
                if o[col] != n[col]:
                    mismatches.append({
                        "pair": list(pair_key),
                        "column": col,
                        "orig": o[col],
                        "new": n[col],
                    })

    # Compute cross-class findings (factual only)
    cross_class_hits = []
    for r in rb_rows:
        if r["same_class"] == "False":
            ct = int(r["common_tcp_five_tuples"])
            cu = int(r["common_udp_five_tuples"])
            ipkt = int(r["identical_packet_count"])
            if (ct + cu) >= 1 or ipkt >= 1:
                cross_class_hits.append({
                    "file_a": r["file_a"],
                    "file_b": r["file_b"],
                    "class_a": records[r["file_a"]]["class"],
                    "class_b": records[r["file_b"]]["class"],
                    "time_relation": r["time_relation"],
                    "gap_seconds": float(r["gap_seconds"]),
                    "common_tcp_five_tuples": ct,
                    "split_tcp_connections": int(r["split_tcp_connections"]),
                    "uncertain_tcp_boundaries": int(r["uncertain_tcp_boundaries"]),
                    "common_udp_five_tuples": cu,
                    "identical_packet_count": ipkt,
                    "identical_packet_fraction_a": float(r["identical_packet_fraction_a"]),
                    "identical_packet_fraction_b": float(r["identical_packet_fraction_b"]),
                    "identical_packet_fraction_min": float(r["identical_packet_fraction_min"]),
                    "verdict": r["verdict"],
                    "suspected_duplicate": r["suspected_duplicate"] == "True",
                })

    # Sort cross_class_hits deterministically: by verdict/identical_packet_count/common_tuples/file_a/file_b
    cross_class_hits.sort(
        key=lambda x: (
            -(x["identical_packet_count"]),
            -(x["common_tcp_five_tuples"] + x["common_udp_five_tuples"]),
            x["file_a"],
            x["file_b"],
        )
    )

    same_class_rows = [r for r in rb_rows if r["same_class"] == "True"]
    cross_class_rows = [r for r in rb_rows if r["same_class"] == "False"]

    summary = {
        "status": "passed" if not mismatches else "failed",
        "elapsed_seconds": round(time.time() - t0, 2),
        "file_count": len(records),
        "total_pairs": len(rb_rows),
        "same_class_pairs": len(same_class_rows),
        "cross_class_pairs": len(cross_class_rows),
        "in_original_130_count": len(new_orig_rows),
        "non_empty_feature_cells": non_empty_feature_cells,
        "gate_130_pairs_checked": len(orig_rows),
        "gate_130_exact_match": len(mismatches) == 0,
        "gate_130_mismatch_count": len(mismatches),
        "gate_130_mismatches": mismatches[:50],
        "verdict_counts_all": dict(Counter(r["verdict"] for r in rb_rows)),
        "verdict_counts_same_class": dict(Counter(r["verdict"] for r in same_class_rows)),
        "verdict_counts_cross_class": dict(Counter(r["verdict"] for r in cross_class_rows)),
        "suspected_duplicate_all": sum(r["suspected_duplicate"] == "True" for r in rb_rows),
        "suspected_duplicate_same_class": sum(r["suspected_duplicate"] == "True" for r in same_class_rows),
        "suspected_duplicate_cross_class": sum(r["suspected_duplicate"] == "True" for r in cross_class_rows),
        "cross_class_shared_tuple_or_packet_count": len(cross_class_hits),
        "cross_class_shared_tuple_or_packet_pairs": cross_class_hits,
        "parent_rss_mib_after_preload": round(parent_rss, 1),
        "pair_eval_workers": n_workers,
        "salt_deleted": not saltpath.exists(),
        "output_sha256": hashlib.sha256(out_csv_gz.read_bytes()).hexdigest(),
    }
    dump(OUT_DIR / "r2_summary.json", summary)

    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "cross_class_shared_tuple_or_packet_pairs"},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if mismatches:
        print("[R2] GATE FAILED! Stopping as required by task rules.", flush=True)
        return 1
    print("[R2] GATE PASSED (130/130 exact match across all 12 gate columns + class + candidate_origins).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
