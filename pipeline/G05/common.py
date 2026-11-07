"""G05 adapters. Data, transforms, models and inner split RNG retain G01 behavior."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parents[1]

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temp.replace(path)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def g01_root():
    return (W / read(W/'config/paths.json')['g01_root']).resolve()

def array_identity(a):
    return {'dtype':str(a.dtype),'shape':list(a.shape),'sha256':hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}

def load_data(condition):
    with np.load(g01_root()/'data/iscx_v3.npz', allow_pickle=False) as z:
        data = {k:z[k] for k in z.files}
    data['file'] = data.pop('source_file')
    classes = read(W/'config/final_classes.json')['main_classes']
    if condition in ['Pstar_8','E3_8']:
        classes = [c for c in classes if c != 'vpn_streaming']
    old = read(W/'config/class_map.json'); ids = [old[c] for c in classes]
    mask = np.isin(data['label'],ids)
    if condition not in ['no_clean','interaction']: mask &= data['keep_clean']
    data = {k:v[mask] for k,v in data.items()}
    remap = {v:i for i,v in enumerate(ids)}
    data['label'] = np.array([remap[int(v)] for v in data['label']], dtype=np.int64)
    if condition != 'no_sni': data['raw'] = data['raw_masked']
    groups = data['session'].copy()
    if condition in ['no_session','interaction']: groups = data['file'].copy()
    if condition == 'evidence':
        mapping = read(g01_root()/'audit/evidence_sessions.json')['file_to_session']
        groups = np.array([mapping[f] for f in data['file']])
    if condition in ['E2','E3_8']:
        variant = 'E2' if condition == 'E2' else 'E3'
        with (W/'config'/f'{variant}_file_to_group.csv').open(newline='') as f:
            mapping = {r['file']:r['group'] for r in csv.DictReader(f)}
        groups = np.array([mapping[f] for f in data['file']])
    return data, classes, groups

# Use the original implementation, with no arithmetic changes.
from source_g01 import transform

class EarlyStopping:
    def __init__(self, patience=30):
        self.patience=patience; self.best=-1.; self.best_epoch=None
    def observe(self, epoch, score):
        improved = score > self.best
        if improved: self.best,self.best_epoch=score,epoch
        return improved, epoch-self.best_epoch >= self.patience

def stop_is_active():
    return (W/'markers/STOPPED.json').exists()

def protocol_guard():
    assert not stop_is_active(), 'Protocol stop active'
    frozen = read(W/'config/frozen_inputs.json')
    for task, entries in frozen.items():
        root = W if task == 'g05_frozen' else W.parent/task
        for entry in entries:
            if sha(root/entry['path']) != entry['sha256']:
                dump(W/'markers/STOPPED.json',{'time':now(),'stage':'frozen_input_guard','scope':task,'path':entry['path'],'training_started':True})
                raise RuntimeError('Frozen input changed: '+task+'/'+entry['path'])

def check_splits(data, groups, path, condition):
    n=len(groups); y=data['label']; nclasses=len(np.unique(y)); seen=np.zeros(n,dtype=int); details=[]
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)=={f'fold{f}_{p}' for f in range(5) for p in ['fit','validation','test']}
        for fold in range(5):
            parts={p:z[f'fold{fold}_{p}'] for p in ['fit','validation','test']}
            fit,val,test=[parts[p] for p in ['fit','validation','test']]
            for a in parts.values():
                assert a.ndim==1 and a.dtype.kind in 'iu' and len(a)>0
                assert a.min()>=0 and a.max()<n and len(set(a))==len(a)
            assert not (set(fit)&set(val) or set(fit)&set(test) or set(val)&set(test))
            assert not set(groups[fit])&set(groups[val])
            if condition!='leak': assert not (set(groups[fit])|set(groups[val]))&set(groups[test])
            train=np.setdiff1d(np.arange(n),test)
            counts=[]
            for c in range(nclasses):
                row={'class_index':c}
                for p,idx in list(parts.items())+[('outer_train',train)]:
                    selected=idx[y[idx]==c];row[p+'_flows']=len(selected);row[p+'_groups']=len(np.unique(groups[selected]))
                assert row['fit_groups']>=1
                assert row['fit_flows']<=1000
                assert row['validation_groups']==(1 if row['outer_train_groups']>1 else 0)
                counts.append(row)
            seen[test]+=1
            details.append({'fold':fold,'per_class':counts,'arrays':{k:array_identity(a) for k,a in parts.items()}})
    assert np.all(seen==1)
    return details
