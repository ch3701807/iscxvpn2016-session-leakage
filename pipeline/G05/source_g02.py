"""One G02 run: reuse exact G01 splits; change only epoch cap and early stopping."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
import time
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from common import W, EarlyStopping, dump, load_data, now, protocol_guard, read, sha, transform
warnings.simplefilter('always')


def metrics(y, p, names):
    labels = np.arange(len(names))
    precision, recall, fscore, support = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {'macro_f1': float(f1_score(y, p, labels=labels, average='macro', zero_division=0)),
            'accuracy': float(accuracy_score(y, p)), 'confusion_matrix': confusion_matrix(y, p, labels=labels).tolist(),
            'per_class': [{'class': name, 'precision': float(precision[i]), 'recall': float(recall[i]),
                           'f1': float(fscore[i]), 'support': int(support[i])} for i, name in enumerate(names)]}


def deep(data, fit, val, test, job, nclasses, fold):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    sys.path.insert(0, str(W / 'scripts/v2'))
    import screen_v2 as v2
    from determinism import configure
    seed = job['seed'] + fold * 10007
    configure(seed)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU worker cannot see its authorized CUDA device')
    if not len(val):
        raise RuntimeError('No validation rows; selection cannot proceed')
    scaled, normalization = transform(data, fit, True)
    direction = int(job['model'])
    model, kind = v2.make_model(direction, nclasses, 6)
    model.to('cuda')

    def tensors(idx):
        x = v2.build_inputs(scaled, direction, idx)
        return (x['seq2'], x['raw'], x['stat'], x['y']) if kind == 'fusion' else (x[kind], x['y'])

    train = tensors(fit)
    valid = tensors(val)
    testing = tensors(test)
    loader = DataLoader(TensorDataset(*train), batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    def fwd(xs):
        return model(*xs) if kind == 'fusion' else model(xs[0])

    def infer(values):
        model.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, len(values[-1]), 512):
                outputs.append(torch.softmax(fwd([x[i:i + 512].to('cuda') for x in values[:-1]]), 1).cpu().numpy())
        return np.concatenate(outputs)

    stopper = EarlyStopping(job['patience'])
    best_weights = None
    history = []
    reason = 'epoch_cap'
    started = time.monotonic()
    for epoch in range(1, job['max_epochs'] + 1):
        model.train()
        total, count = 0., 0
        for *xs, y in loader:
            xs = [x.to('cuda') for x in xs]
            y = y.to('cuda')
            opt.zero_grad()
            loss = loss_fn(fwd(xs), y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(y)
            count += len(y)
        pv = infer(valid)
        mf = float(f1_score(data['label'][val], pv.argmax(1), labels=np.arange(nclasses),
                            average='macro', zero_division=0))
        assert np.isfinite(mf) and np.isfinite(total)
        history.append({'epoch': epoch, 'train_loss': total / count, 'validation_macro_f1': mf,
                        'validation_accuracy': float(accuracy_score(data['label'][val], pv.argmax(1)))})
        improved, stop = stopper.observe(epoch, mf)
        if improved:
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        status = {'time': now(), 'job': job['id'], 'fold': fold, 'epoch': epoch,
                  'train_loss': total / count, 'validation_macro_f1': mf,
                  'best_epoch': stopper.best_epoch, 'best_validation_macro_f1': stopper.best,
                  'epochs_without_strict_improvement': epoch - stopper.best_epoch}
        print(json.dumps(status), flush=True)
        dump(W / 'progress' / (job['id'] + '.json'), status)
        if stop:
            reason = 'patience_30'
            break
    model.load_state_dict(best_weights)
    probabilities = infer(testing)
    extra = {'selected_epoch': stopper.best_epoch, 'stop_epoch': epoch, 'epoch_cap': job['max_epochs'],
             'patience': job['patience'], 'stop_reason': reason, 'stopped_at_epoch_300': epoch == 300,
             'epochs_without_strict_improvement': epoch - stopper.best_epoch,
             'selected_validation_macro_f1': stopper.best, 'validation_curve': history,
             'normalization': normalization, 'model_seed': seed, 'elapsed_seconds': time.monotonic() - started,
             'test_inference_count': 1,
             'deterministic_settings': {'cudnn.deterministic': True, 'cudnn.benchmark': False,
                                      'deterministic_algorithms': True, 'warn_only': True,
                                      'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG']}}
    del model, best_weights, opt, loader
    torch.cuda.empty_cache()
    return probabilities, extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['2', '3'], 'Only physical GPUs 2 or 3 are permitted'
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    protocol_guard()
    job = next(j for j in read(W / 'config/queue_plan.json')['jobs'] if j['id'] == args.job)
    protocol_hash = sha(W / 'config/protocol.json')
    split_file = W / job['split_file']
    assert sha(split_file) == job['split_sha256']
    marker = W / 'markers/runs' / (job['id'] + '.json')
    result_file = W / 'results' / (job['id'] + '.json')
    if marker.exists():
        saved = read(marker)
        assert saved['protocol_sha256'] == protocol_hash and sha(result_file) == saved['result_sha256']
        print(json.dumps({'job': job['id'], 'already_completed': True}), flush=True)
        return
    data, classes, groups = load_data(job['condition'])
    n = len(data['label'])
    probabilities = np.full((n, len(classes)), np.nan, dtype=np.float32)
    fold_numbers = np.full(n, -1, dtype=np.int8)
    histories, resumed = [], []
    checkpoint_dir = W / 'checkpoints' / job['id']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with np.load(split_file, allow_pickle=False) as splits:
        for fold in range(5):
            fit, val, test = [splits[f'fold{fold}_{part}'] for part in ['fit', 'validation', 'test']]
            assert np.all(fold_numbers[test] == -1)
            meta_path = checkpoint_dir / f'fold{fold}.json'
            prob_path = checkpoint_dir / f'fold{fold}.npz'
            done_path = checkpoint_dir / f'fold{fold}.done.json'
            if done_path.exists():
                done = read(done_path)
                assert done['protocol_sha256'] == protocol_hash and done['split_sha256'] == job['split_sha256']
                assert sha(meta_path) == done['metadata_sha256'] and sha(prob_path) == done['probabilities_sha256']
                extra = read(meta_path)
                with np.load(prob_path, allow_pickle=False) as cached:
                    assert np.array_equal(cached['test'], test)
                    p = cached['probabilities']
                resumed.append(fold)
            else:
                p, extra = deep(data, fit, val, test, job, len(classes), fold)
                extra = {'fold': fold, 'n_fit': len(fit), 'n_validation': len(val), 'n_test': len(test), **extra}
                temp = checkpoint_dir / f'fold{fold}.tmp.npz'
                np.savez_compressed(temp, test=test, probabilities=p)
                temp.replace(prob_path)
                dump(meta_path, extra)
                dump(done_path, {'time': now(), 'protocol_sha256': protocol_hash,
                                 'split_sha256': job['split_sha256'], 'metadata_sha256': sha(meta_path),
                                 'probabilities_sha256': sha(prob_path)})
            assert p.shape == (len(test), len(classes)) and np.isfinite(p).all()
            assert np.allclose(p.sum(1), 1.)
            probabilities[test] = p
            fold_numbers[test] = fold
            histories.append(extra)
    assert np.all(fold_numbers >= 0) and np.isfinite(probabilities).all()
    predictions = probabilities.argmax(1)
    result = {'job': job, 'status': 'completed', 'post_G01_Bplus1_addition': True, 'classes': classes,
              'n_flows': n, 'metrics': metrics(data['label'], predictions, classes), 'fold_details': histories,
              'protocol_sha256': protocol_hash, 'split_sha256': job['split_sha256'],
              'metric_aggregation': 'merged five-fold OOF predictions', 'tie_break_epoch': 'first maximum',
              'training_pool_reused_verbatim': True, 'resumed_completed_folds': resumed,
              'grouping': {'Pstar': 'suffix_session', 'no_session': 'file', 'leak': 'flow_stratified',
                           'evidence': 'evidence_session'}[job['condition']]}
    pred_path = W / 'results' / (job['id'] + '_predictions.csv.gz')
    temp_pred = pred_path.with_suffix('.tmp')
    with gzip.open(temp_pred, 'wt', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['flow_id', 'session', 'file', 'fold', 'true_label', 'pred_label'] +
                        [f'prob_{i}' for i in range(len(classes))])
        for i in range(n):
            session = groups[i] if job['condition'] == 'evidence' else data['session'][i]
            writer.writerow([data['flow_id'][i], session, data['file'][i], int(fold_numbers[i]),
                             int(data['label'][i]), int(predictions[i])] + [float(x) for x in probabilities[i]])
    temp_pred.replace(pred_path)
    dump(result_file, result)
    protocol_guard()
    dump(marker, {'job': job['id'], 'time': now(), 'status': 'completed', 'protocol_sha256': protocol_hash,
                  'result_sha256': sha(result_file), 'predictions_sha256': sha(pred_path), 'n_flows': n})
    print(json.dumps({'job': job['id'], 'completed': True, 'metrics': result['metrics']}), flush=True)


if __name__ == '__main__':
    main()
