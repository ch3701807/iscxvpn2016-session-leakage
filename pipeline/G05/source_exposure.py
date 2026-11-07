"""Frozen B+2 definitions, private connection indexing and exact fit-only exposure."""
import bisect
import csv
import datetime
import gzip
import hashlib
import hmac
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from extract_v3 import PORTS, dump, ipv4_transport
from pcap_stream import Capture
from protocol_control import stop_is_active, verify_frozen_rules
from run_experiment import load

W = Path(__file__).resolve().parents[1]
PRIVATE = W / 'data/near_duplicate_private'
DEFINITIONS = ['same_connection', 'same_sequence', 'same_sequence_raw']


def connection_index(spec):
    record, muse = spec
    name = record['file']
    out = PRIVATE / (Path(name).stem + '.npz')
    marker = W / 'markers/near_duplicate_files' / (name + '.json')
    if out.exists() and marker.exists():
        return json.loads(marker.read_text())
    salt = (W / 'data/sibling_salt.bin').read_bytes()
    flows = {}
    addresses = {}
    reader = Capture(Path(muse) / 'data/ISCXVPN2016' / name)
    for t, raw, link in reader:
        packet = ipv4_transport(raw, link)
        if packet is None:
            continue
        src, dst, sp, dp, proto, length, payload, broadcast, _, _, _ = packet
        for address in (src, dst):
            if address not in addresses:
                addresses[address] = '.'.join(map(str, address))
        ep1, ep2 = (addresses[src], sp), (addresses[dst], dp)
        key = (min(ep1, ep2), max(ep1, ep2), proto)
        chain = flows.setdefault(key, [])
        if not chain or t - chain[-1]['last'] > 120:
            chain.append({'last': t, 'initiator': src, 'packets': [], 'drop': False})
        flow = chain[-1]
        flow['last'] = t
        flow['drop'] |= (sp in PORTS or dp in PORTS or 224 <= dst[0] <= 239 or
                         dst == (255, 255, 255, 255) or broadcast)
        bisect.insort_right(flow['packets'], (t, 1 if src == flow['initiator'] else -1,
                                            min(length, 1500)))
        if len(flow['packets']) > 50:
            flow['packets'].pop()
    assert reader.sha.hexdigest() == record['input_sha256'], 'Changed input capture'
    reader.close()
    connection_ids, clean, seqs, iats = [], [], [], []
    for key in sorted(flows):
        ident = hmac.new(salt, repr(key).encode(), hashlib.sha256).hexdigest()
        for flow in flows[key]:
            packets = flow['packets']
            if len(packets) < 5:
                continue
            times = np.array([p[0] for p in packets], dtype=np.float64)
            seq = np.zeros(50, dtype=np.float32)
            seq[:len(packets)] = [p[1] * p[2] for p in packets]
            iat = np.zeros(50, dtype=np.float32)
            iat[1:len(packets)] = np.diff(times)
            connection_ids.append(ident)
            clean.append(not flow['drop'])
            seqs.append(seq)
            iats.append(iat)
    with np.load(W / 'data/v3_files' / (Path(name).stem + '.npz'), allow_pickle=False) as reference:
        n = len(reference['flow_id'])
        assert n == len(connection_ids)
        assert np.array_equal(np.asarray(seqs, dtype=np.float32).reshape(n, 50), reference['seq'])
        assert np.array_equal(np.asarray(iats, dtype=np.float32).reshape(n, 50), reference['iat'])
        assert np.array_equal(np.asarray(clean, dtype=bool), reference['keep_clean'])
        flow_ids = reference['flow_id'].copy()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, flow_id=flow_ids, connection_hmac=np.asarray(connection_ids, dtype='U64'))
    result = {'file': name, 'flows': n, 'clean_flows': sum(clean),
              'input_sha256': record['input_sha256'], 'v3_sequence_iat_and_filter_exact': True,
              'existing_hmac_salt_reused': True,
              'private_index_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}
    dump(marker, result)
    return result


def all_keys(records):
    keys = {}
    by_file = {}
    for record in records:
        name = record['file']
        stem = Path(name).stem
        with np.load(PRIVATE / (stem + '.npz'), allow_pickle=False) as private, \
             np.load(W / 'data/v3_files' / (stem + '.npz'), allow_pickle=False) as data:
            assert np.array_equal(private['flow_id'], data['flow_id'])
            file_keys = {definition: [] for definition in DEFINITIONS}
            for i, fid in enumerate(data['flow_id']):
                eligible = np.count_nonzero(data['seq'][i]) >= 10
                sequence = data['seq'][i].tobytes(order='C') if eligible else None
                strict = sequence + data['raw_masked'][i].tobytes(order='C') if eligible else None
                k = [str(private['connection_hmac'][i]), sequence, strict]
                keys[str(fid)] = k
                if data['keep_clean'][i]:
                    for definition, value in zip(DEFINITIONS, k):
                        if value is not None:
                            file_keys[definition].append(value)
            by_file[name] = {definition: Counter(values) for definition, values in file_keys.items()}
    return keys, by_file


def augment_pairs(by_file):
    target = W / 'audit/sibling_pairs.csv'
    backup = W / 'audit/sibling_pairs.before_Bplus2.csv'
    if not backup.exists():
        backup.write_bytes(target.read_bytes())
    with backup.open(newline='') as f:
        reader = csv.DictReader(f)
        original_fields = list(reader.fieldnames)
        original = list(reader)
    rows = []
    extra = [definition + '_' + suffix for definition in DEFINITIONS
             for suffix in ['matched_flows_a', 'matched_flows_b', 'matching_flow_pairs']]
    for row in original:
        result = dict(row)
        for definition in DEFINITIONS:
            a = by_file[row['file_a']][definition]
            b = by_file[row['file_b']][definition]
            common = a.keys() & b.keys()
            result[definition + '_matched_flows_a'] = sum(a[k] for k in common)
            result[definition + '_matched_flows_b'] = sum(b[k] for k in common)
            result[definition + '_matching_flow_pairs'] = sum(a[k] * b[k] for k in common)
        assert all(result[k] == row[k] for k in original_fields)
        rows.append(result)
    temp = target.with_suffix('.csv.tmp')
    with temp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=original_fields + extra)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(target)
    dump(W / 'audit/near_duplicate_pair_summary.json', {
        'post_stage_a_addition': True, 'pairs': len(rows), 'original_columns_unchanged': True,
        'original_csv_sha256': hashlib.sha256(backup.read_bytes()).hexdigest(),
        'augmented_csv_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'scope': 'all cleaned application classes, including excluded single-session classes',
        'definitions': {definition: {
            'pairs_with_matches': sum(int(r[definition + '_matching_flow_pairs']) > 0 for r in rows),
            'sum_matched_flows_a': sum(int(r[definition + '_matched_flows_a']) for r in rows),
            'sum_matched_flows_b': sum(int(r[definition + '_matched_flows_b']) for r in rows),
            'sum_matching_flow_pairs': sum(int(r[definition + '_matching_flow_pairs']) for r in rows),
        } for definition in DEFINITIONS},
        'warning': 'Sums across file pairs can count a flow more than once.',
    })


def measure(condition, seed, keys):
    prefix = 'A' if condition == 'Pstar' else 'B'
    job_id = f'{prefix}_baseline_{condition}_multi_RF_seq_iat_{seed}'
    result = json.loads((W / 'results/v3' / (job_id + '.json')).read_text())
    data, classes, groups = load(result['job'])
    n = len(data['label'])
    labels = data['label']
    flowids = list(map(str, data['flow_id']))
    position = {fid: i for i, fid in enumerate(flowids)}
    assert len(position) == n
    correct = np.zeros(n, dtype=bool)
    pred = np.full(n, -1, dtype=np.int64)
    pred_fold = np.full(n, -1, dtype=np.int64)
    with gzip.open(W / 'results/v3' / (job_id + '_predictions.csv.gz'), 'rt') as f:
        seen = set()
        for row in csv.DictReader(f):
            fid = row['flow_id']
            assert fid not in seen
            seen.add(fid)
            i = position[fid]
            assert int(row['true_label']) == labels[i]
            assert row['file'] == data['file'][i]
            pred[i] = int(row['pred_label'])
            correct[i] = pred[i] == labels[i]
            pred_fold[i] = int(row['fold'])
        assert len(seen) == n
    same = np.zeros((n, len(DEFINITIONS)), dtype=bool)
    other = np.zeros_like(same)
    seen_test = np.zeros(n, dtype=np.int64)
    with np.load(W / 'results/v3' / (job_id + '_splits.npz'), allow_pickle=False) as splits:
        for fold in range(5):
            fit, validation, test = [splits[f'fold{fold}_{part}'] for part in ['fit', 'validation', 'test']]
            assert not (set(fit) & set(validation) or set(fit) & set(test) or set(validation) & set(test))
            assert np.all(pred_fold[test] == fold)
            seen_test[test] += 1
            for definition_index in range(len(DEFINITIONS)):
                fit_labels = defaultdict(set)
                for i in fit:
                    value = keys[flowids[i]][definition_index]
                    if value is not None:
                        fit_labels[value].add(int(labels[i]))
                for i in test:
                    value = keys[flowids[i]][definition_index]
                    matches = fit_labels.get(value, set()) if value is not None else set()
                    same[i, definition_index] = int(labels[i]) in matches
                    other[i, definition_index] = any(c != labels[i] for c in matches)
    assert np.all(seen_test == 1)
    rows = []
    for fold in [None, 0, 1, 2, 3, 4]:
        for class_index in [None] + list(range(len(classes))):
            mask = np.ones(n, dtype=bool)
            if fold is not None:
                mask &= pred_fold == fold
            if class_index is not None:
                mask &= labels == class_index
            indices = np.flatnonzero(mask)
            denominator = len(indices)
            for k, definition in enumerate(DEFINITIONS):
                a, b = same[indices, k], other[indices, k]
                exposed = a | b
                both = int(np.count_nonzero(a & b))
                only_same = int(np.count_nonzero(a & ~b))
                only_other = int(np.count_nonzero(~a & b))
                neither = int(np.count_nonzero(~exposed))
                assert both + only_same + only_other + neither == denominator
                count = int(np.count_nonzero(exposed))
                exposed_correct = int(np.count_nonzero(correct[indices] & exposed))
                unexposed_correct = int(np.count_nonzero(correct[indices] & ~exposed))
                rows.append({
                    'condition': condition, 'seed': seed, 'fold': fold,
                    'class': classes[class_index] if class_index is not None else '__all__',
                    'definition': definition, 'n_test': denominator,
                    'same_class_only': only_same, 'other_class_only': only_other,
                    'both_classes': both, 'neither': neither,
                    'same_class_exposed': only_same + both, 'other_class_exposed': only_other + both,
                    'any_exposed': count, 'any_exposure_rate': count / denominator if denominator else None,
                    'same_class_exposure_rate': (only_same + both) / denominator if denominator else None,
                    'other_class_exposure_rate': (only_other + both) / denominator if denominator else None,
                    'exposed_correct': exposed_correct, 'exposed_accuracy': exposed_correct / count if count else None,
                    'unexposed_correct': unexposed_correct,
                    'unexposed_accuracy': unexposed_correct / neither if neither else None,
                })
    destination = W / 'audit/near_duplicate_exposure'
    destination.mkdir(parents=True, exist_ok=True)
    flags = [definition + '_' + flag for definition in DEFINITIONS
             for flag in ['same_class', 'other_class', 'any']]
    with gzip.open(destination / f'{condition}_{seed}_flows.csv.gz', 'wt', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['flow_id', 'file', 'session', 'fold', 'true_label', 'pred_label', 'correct'] + flags)
        for i, fid in enumerate(flowids):
            values = []
            for k in range(len(DEFINITIONS)):
                values.extend([int(same[i, k]), int(other[i, k]), int(same[i, k] or other[i, k])])
            writer.writerow([fid, data['file'][i], groups[i], int(pred_fold[i]), int(labels[i]),
                             int(pred[i]), int(correct[i])] + values)
    dump(destination / f'{condition}_{seed}_summary.json', rows)
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row['fold'] is None:
            grouped[(row['condition'], row['class'], row['definition'])].append(row)
    output = []
    for (condition, cname, definition), values in sorted(grouped.items()):
        entry = {'condition': condition, 'class': cname, 'definition': definition,
                 'seeds': sorted(r['seed'] for r in values), 'per_seed': values}
        for metric in ['any_exposure_rate', 'same_class_exposure_rate', 'other_class_exposure_rate',
                       'exposed_accuracy', 'unexposed_accuracy']:
            a = np.array([r[metric] for r in values if r[metric] is not None], dtype=float)
            entry[metric] = {'n': len(a), 'mean': float(a.mean()) if len(a) else None,
                             'std': float(a.std(ddof=1)) if len(a) > 1 else None}
        output.append(entry)
    dump(W / 'audit/near_duplicate_exposure_summary.json', output)


def main():
    verify_frozen_rules()
    assert not stop_is_active()
    assert (W / 'data/sibling_salt.bin').exists(), 'Must reuse existing salt'
    inventory = json.loads((W / 'audit/v3_cleaning_inventory.json').read_text())
    muse = json.loads((W / 'config/paths.local.json').read_text())['muse_root']
    failures, checked = [], []
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(connection_index, (r, muse)): r['file'] for r in inventory['files']}
        for future in as_completed(futures):
            try:
                record = future.result()
                checked.append(record)
                print('CONNECTION INDEX DONE', record['file'], record['flows'], flush=True)
            except Exception as error:
                failures.append({'file': futures[future], 'error_type': type(error).__name__, 'error': str(error)})
    if failures:
        dump(W / 'audit/near_duplicate_failed_files.json', failures)
        return 2
    dump(W / 'audit/near_duplicate_connection_validation.json', checked)
    keys, by_file = all_keys(inventory['files'])
    augment_pairs(by_file)
    rows = []
    for condition in ['Pstar', 'no_session', 'leak', 'evidence']:
        for seed in [42, 1, 7]:
            rows.extend(measure(condition, seed, keys))
            print('EXPOSURE DONE', condition, seed, flush=True)
    aggregate(rows)
    dump(W / 'markers/BPLUS2_COMPLETED.json', {
        'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'post_stage_a_addition': True, 'file_count': len(checked),
        'conditions': 4, 'seeds_per_condition': 3, 'definitions': DEFINITIONS,
        'fit_only': True, 'source_rf_predictions_reused': True,
        'rules_addendum_sha256': json.loads((W / 'markers/RULES_ADDENDUM_FROZEN.json').read_text())['sha256'],
    })
    return 0


if __name__ == '__main__':
    sys.exit(main())
