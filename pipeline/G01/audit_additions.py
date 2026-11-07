"""Independent vectorized exposure checks and preservation of original results."""
import csv
import gzip
import hashlib
import json
import os
import tarfile
from pathlib import Path
import numpy as np
from extract_v3 import dump
from protocol_control import stop_is_active, verify_frozen_rules
from run_experiment import load

W = Path(__file__).resolve().parents[1]
DEFINITIONS = ['same_connection', 'same_sequence', 'same_sequence_raw']


def row_tokens(array):
    a = np.ascontiguousarray(array)
    return a.view(np.dtype((np.void, a.shape[1] * a.dtype.itemsize))).reshape(-1)


def main():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    verify_frozen_rules()
    assert not stop_is_active()
    reg = json.loads((W / 'audit/continuation3_registration.json').read_text())
    assert hashlib.sha256((W / 'config/queue_plan.json').read_bytes()).hexdigest() == reg['plan_sha256_after_append']
    release = json.loads((W / 'markers/PROTOCOL_STOP_RELEASED.json').read_text())
    assert hashlib.sha256((W / 'markers/STOPPED.json').read_bytes()).hexdigest() == release['released_stop_sha256']
    assert hashlib.sha256((W / 'G01_A_return.tar.gz').read_bytes()).hexdigest() == '4eafb0ec98686040bb200ecaccff7cffb4f4d9119ea1362a65de46a1c075a7ef'
    with tarfile.open(W / 'G01_B_stopped_return.tar.gz') as archive:
        manifest = archive.extractfile('G01_return/SHA256SUMS').read().decode().splitlines()
    expected = {line.split('  ', 1)[1]: line.split('  ', 1)[0] for line in manifest}
    preserved = 0
    for name, checksum in expected.items():
        if name.startswith('results/v3/'):
            assert hashlib.sha256((W / name).read_bytes()).hexdigest() == checksum, name
            preserved += 1
    assert preserved == 393 * 3

    private_ids = {}
    for path in (W / 'data/near_duplicate_private').glob('*.npz'):
        with np.load(path, allow_pickle=False) as data:
            for fid, value in zip(data['flow_id'], data['connection_hmac']):
                assert str(fid) not in private_ids
                private_ids[str(fid)] = str(value)
    checks = []
    for condition in ['Pstar', 'no_session', 'leak', 'evidence']:
        for seed in [42, 1, 7]:
            prefix = 'A' if condition == 'Pstar' else 'B'
            ident = f'{prefix}_baseline_{condition}_multi_RF_seq_iat_{seed}'
            result = json.loads((W / 'results/v3' / (ident + '.json')).read_text())
            data, classes, exposure_groups = load(result['job'])
            n = len(data['label'])
            ids = list(map(str, data['flow_id']))
            position = {fid: i for i, fid in enumerate(ids)}
            y = data['label']
            eligible = np.count_nonzero(data['seq'], axis=1) >= 10
            tokens = [np.array([private_ids[fid] for fid in ids]), row_tokens(data['seq']),
                      row_tokens(np.concatenate([data['seq'], data['raw']], axis=1))]
            calculated = np.zeros((n, 3, 2), dtype=bool)
            with np.load(W / 'results/v3' / (ident + '_splits.npz'), allow_pickle=False) as splits:
                for fold in range(5):
                    fit, test = splits[f'fold{fold}_fit'], splits[f'fold{fold}_test']
                    for k, values in enumerate(tokens):
                        fit_pool = fit if k == 0 else fit[eligible[fit]]
                        test_pool = test if k == 0 else test[eligible[test]]
                        for label in range(len(classes)):
                            query = test_pool[y[test_pool] == label]
                            calculated[query, k, 0] = np.isin(values[query], values[fit_pool[y[fit_pool] == label]])
                            calculated[query, k, 1] = np.isin(values[query], values[fit_pool[y[fit_pool] != label]])
            rf = {}
            with gzip.open(W / 'results/v3' / (ident + '_predictions.csv.gz'), 'rt') as f:
                for row in csv.DictReader(f):
                    rf[row['flow_id']] = row
            observed = set()
            correct = np.zeros(n, dtype=bool)
            fold_ids = np.full(n, -1, dtype=int)
            with gzip.open(W / 'audit/near_duplicate_exposure' / f'{condition}_{seed}_flows.csv.gz', 'rt') as f:
                for row in csv.DictReader(f):
                    fid = row['flow_id']
                    assert fid not in observed
                    observed.add(fid)
                    i = position[fid]
                    for field in ['file', 'fold', 'true_label', 'pred_label']:
                        assert row[field] == rf[fid][field], field
                    # Auxiliary exposure tables store the condition's actual grouping unit.
                    # RF prediction CSVs intentionally retain suffix sessions for file-group ablations.
                    assert row['session'] == str(exposure_groups[i])
                    correct[i] = row['true_label'] == row['pred_label']
                    fold_ids[i] = int(row['fold'])
                    assert int(row['correct']) == correct[i]
                    for k, definition in enumerate(DEFINITIONS):
                        assert int(row[definition + '_same_class']) == calculated[i, k, 0]
                        assert int(row[definition + '_other_class']) == calculated[i, k, 1]
                        assert int(row[definition + '_any']) == calculated[i, k].any()
            assert len(observed) == n == 7371
            assert np.all(~calculated[:, 2, :] | calculated[:, 1, :])
            assert abs(correct.mean() - result['metrics']['accuracy']) < 1e-12
            summary = json.loads((W / 'audit/near_duplicate_exposure' / f'{condition}_{seed}_summary.json').read_text())
            for row in summary:
                selected = np.ones(n, dtype=bool)
                if row['fold'] is not None:
                    selected &= fold_ids == row['fold']
                if row['class'] != '__all__':
                    selected &= y == classes.index(row['class'])
                k = DEFINITIONS.index(row['definition'])
                same, other = calculated[selected, k, 0], calculated[selected, k, 1]
                any_match = same | other
                truth = {'n_test': int(selected.sum()), 'same_class_only': int((same & ~other).sum()),
                         'other_class_only': int((other & ~same).sum()), 'both_classes': int((same & other).sum()),
                         'neither': int((~any_match).sum()), 'any_exposed': int(any_match.sum()),
                         'same_class_exposed': int(same.sum()), 'other_class_exposed': int(other.sum()),
                         'exposed_correct': int((correct[selected] & any_match).sum()),
                         'unexposed_correct': int((correct[selected] & ~any_match).sum())}
                assert all(row[field] == value for field, value in truth.items())
                for rate, numerator, denominator in [('any_exposure_rate','any_exposed','n_test'),
                                                      ('same_class_exposure_rate','same_class_exposed','n_test'),
                                                      ('other_class_exposure_rate','other_class_exposed','n_test'),
                                                      ('exposed_accuracy','exposed_correct','any_exposed'),
                                                      ('unexposed_accuracy','unexposed_correct','neither')]:
                    expected_rate = truth[numerator] / truth[denominator] if truth[denominator] else None
                    assert row[rate] == expected_rate
            checks.append({'condition': condition, 'seed': seed, 'rows': n, 'independent_np_isin_flags_match': True,
                           'all_fold_class_counts_and_accuracies_match': True})
            print('INDEPENDENT EXPOSURE CHECK', condition, seed, flush=True)

    with (W / 'audit/sibling_pairs.before_Bplus2.csv').open() as f:
        original = list(csv.DictReader(f))
    with (W / 'audit/sibling_pairs.csv').open() as f:
        augmented = list(csv.DictReader(f))
    assert len(original) == len(augmented) == 130
    for old, new in zip(original, augmented):
        assert all(new[k] == value for k, value in old.items())
        for suffix in ['matched_flows_a', 'matched_flows_b', 'matching_flow_pairs']:
            assert int(new['same_sequence_raw_' + suffix]) <= int(new['same_sequence_' + suffix])
    metadata = json.loads((W / 'audit/vpn_encapsulation_files.json').read_text())
    inventory = json.loads((W / 'audit/v3_cleaning_inventory.json').read_text())['files']
    source = {r['file']: r for r in inventory}
    assert len(metadata) == len(source) == 140
    for row in metadata:
        r = source[row['file']]
        assert row['packet_count'] == sum(row['linktype_packet_counts'].values()) == r['packet_count']
        assert row['input_sha256'] == r['input_sha256']
        for stage, source_key, count_key in [('before_clean','sni_before_clean','before_flows'),
                                            ('after_clean','sni_after_clean','after_flows')]:
            tls = row['tls_flow_statistics'][stage]
            assert [tls['clienthello_flows'], tls['sni_flows']] == r[source_key][:2]
            assert tls['flow_count'] == r[count_key]
    ustc = json.loads((W / 'audit/ustc_summary.json').read_text())
    files = json.loads((W / 'audit/ustc_file_inventory.json').read_text())
    assert len(files) == ustc['capture_file_count'] == 24
    for row in ustc['classes']:
        records = [r for r in files if r['class'] == row['class']]
        assert row['suffix_sessions_with_unclean_flows'] == len({r['suffix_session_id'] for r in records if r['v3_before_clean_flows']})
        assert row['suffix_sessions_with_clean_flows'] == len({r['suffix_session_id'] for r in records if r['v3_clean_flows']})
        assert row['cleaning_rule_conflict'] == (row['removed_fraction'] > .5)
    for row in files:
        assert row['time_span_seconds'] == row['max_packet_timestamp'] - row['min_packet_timestamp']
    dump(W / 'audit/continuation4_final_validation.json', {
        'passed': True, 'original_results_files_preserved': preserved,
        'original_queue_plan_unchanged': True, 'original_stage_a_archive_unchanged': True,
        'historical_stop_preserved': True, 'frozen_rules_unchanged': True,
        'exposure_checks': checks, 'sibling_original_columns_preserved': True,
        'vpn_metadata_file_checks': 140, 'ustc_file_checks': 24,
    })
    print('All independent addition and preservation checks passed.', flush=True)


if __name__ == '__main__':
    main()
