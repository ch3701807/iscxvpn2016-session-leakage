"""G06 CPU-only utilities; all historical inputs are read-only."""
import csv, datetime, gzip, hashlib, json, os, re
from pathlib import Path
import numpy as np

W=Path(__file__).resolve().parents[1]
G01=Path(os.environ.get('G06_G01_ROOT',W.parent/'G01')).resolve()
G05=Path(os.environ.get('G06_G05_ROOT',W.parent/'G05')).resolve()
HANDOFF=Path(os.environ.get('G06_HANDOFF',W.parent.parent/'paper/claude/BlockSys')).resolve()
CONDITIONS=['leak','no_session','Pstar','evidence','E2']
SEEDS=[1,7,42]

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def cpu_only():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert os.environ.get('PYTHONNOUSERSITE')=='1'
 for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:assert os.environ.get(k)=='1'
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:16])
def guard():
 cpu_only()
 f=read(W/'config/frozen.json')
 for scope,items in f.items():
  root={'G06':W,'G01':G01,'G05':G05}[scope]
  for x in items:assert sha(root/x['path'])==x['sha256'],'Frozen input changed: '+scope+'/'+x['path']
def load_data(condition):
 with np.load(G01/'data/iscx_v3.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 d['file']=d.pop('source_file');classes=read(W/'config/final_classes.json')['main_classes'];old=read(W/'config/class_map.json');ids=[old[c] for c in classes]
 mask=np.isin(d['label'],ids)&d['keep_clean'];d={k:v[mask] for k,v in d.items()};remap={v:i for i,v in enumerate(ids)}
 d['label']=np.array([remap[int(v)] for v in d['label']],dtype=np.int64);d['raw']=d['raw_masked']
 groups=d['file'].copy() if condition=='no_session' else d['session'].copy()
 if condition=='evidence':m=read(G01/'audit/evidence_sessions.json')['file_to_session'];groups=np.array([m[f] for f in d['file']])
 if condition=='E2':
  with (W/'config/E2_file_to_group.csv').open(newline='') as f:m={r['file']:r['group'] for r in csv.DictReader(f)}
  groups=np.array([m[f] for f in d['file']])
 assert len(d['label'])==7371 and len(classes)==9
 return d,classes,groups
def source_paths(condition,seed):
 if condition=='leak':root=G01;base=f'B_baseline_leak_multi_RF_seq_iat_{seed}';split=root/'results/v3'/f'{base}_splits.npz'
 else:root=G05;base=f'G05_{condition}_multi_RF_seq_iat_{seed}';split=root/'data/splits'/f'G05_{condition}_{seed}_splits.npz'
 return split,root/'results/v3'/f'{base}.json',root/'results/v3'/f'{base}_predictions.csv.gz'
CAP=1000
def fit_full(y,test,seed,fold):
 rng=np.random.default_rng(seed+fold*10007+500009)
 train=np.setdiff1d(np.arange(len(y)),test)
 parts=[]
 for c in np.unique(y[train]):
  p=train[y[train]==c].copy();rng.shuffle(p);parts.append(p[:CAP])
 f=np.concatenate(parts);rng.shuffle(f)
 return f.astype('<i8')
def read_predictions(p):
 with gzip.open(p,'rt',newline='') as f:
  r=csv.DictReader(f);header=r.fieldnames;rows=list(r)
 return header,rows,np.array([[float(r[f'prob_{i}']) for i in range(9)] for r in rows],dtype=np.float32)
def scrub(s):
 for root in [str(W),str(W.parent.parent),str(Path.home())]:s=s.replace(root,'${LOCAL_ROOT}')
 s=re.sub(r'/home/[^/\s:]+','${HOME}',s)
 return s
