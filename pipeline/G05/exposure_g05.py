"""Original B+2 definitions, changing only input path expressions for G05 naming."""
import argparse
import ast
from collections import Counter,defaultdict
import csv
import difflib
import gzip
import json
from pathlib import Path
import numpy as np
from common import W,dump,load_data,now,protocol_guard,read,sha

def functions_scope():
    src=W/'scripts/source_exposure.py';original=src.read_text()
    changes={
      "W / 'results/v3' / (job_id + '.json')":"W / 'results/v3' / f'G05_{condition}_multi_RF_seq_iat_{seed}.json'",
      "W / 'results/v3' / (job_id + '_predictions.csv.gz')":"W / 'results/v3' / f'G05_{condition}_multi_RF_seq_iat_{seed}_predictions.csv.gz'",
      "W / 'results/v3' / (job_id + '_splits.npz')":"W / 'data/splits' / f'G05_{condition}_{seed}_splits.npz'"}
    patched=original
    for old,new in changes.items():assert patched.count(old)==1;patched=patched.replace(old,new)
    reversed_text=patched
    for old,new in changes.items():reversed_text=reversed_text.replace(new,old)
    assert reversed_text==original
    (W/'audit/exposure_path_only.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),fromfile='G01/Bplus2.py',tofile='G05/input_paths.py')))
    funcs=[n for n in ast.parse(patched).body if isinstance(n,ast.FunctionDef) and n.name in ['all_keys','measure','aggregate']]
    assert len(funcs)==3
    scope={'Path':Path,'np':np,'Counter':Counter,'defaultdict':defaultdict,'csv':csv,'gzip':gzip,'json':json,'dump':dump,
        'DEFINITIONS':['same_connection','same_sequence','same_sequence_raw'],'load':lambda j:load_data(j['condition']),
        'W':W.parent/'G01','PRIVATE':W.parent/'G01/data/near_duplicate_private'}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<G05 B+2 input path adapter>','exec'),scope)
    return scope,changes

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--condition');ap.add_argument('--seed',type=int);ap.add_argument('--aggregate',action='store_true');args=ap.parse_args()
    assert __import__('os').environ.get('CUDA_VISIBLE_DEVICES','')=='';protocol_guard()
    scope,changes=functions_scope()
    if args.aggregate:
        rows=[];completed=[];failed=[]
        for c in ['Pstar','no_session','evidence']:
            for seed in [1,7,42]:
                m=W/'markers/exposure'/f'{c}_{seed}.json'
                if m.exists():
                    saved=read(m);p=W/'audit/near_duplicate_exposure'/f'{c}_{seed}_summary.json';assert sha(p)==saved['summary_sha256'];rows+=read(p);completed.append({'condition':c,'seed':seed})
                else:failed.append({'condition':c,'seed':seed})
        scope['W']=W;scope['aggregate'](rows)
        dump(W/'audit/exposure_reuse.json',{'time':now(),'source_sha256':sha(W/'scripts/source_exposure.py'),'only_function_edits_are_input_path_expressions':changes,'definitions_unchanged':True,'salt_not_copied':True,'private_index_reused_readonly':True,'completed':completed,'failed':failed})
        dump(W/'markers/EXPOSURE_COMPLETED.json',{'time':now(),'completed':len(completed),'failed':failed,'summary_sha256':sha(W/'audit/near_duplicate_exposure_summary.json')})
        return
    c,seed=args.condition,args.seed;assert c in ['Pstar','no_session','evidence'] and seed in [1,7,42]
    job=next(j for j in read(W/'config/queue_plan.json')['jobs'] if j['condition']==c and j['model']=='RF_seq_iat' and j['seed']==seed)
    split=W/job['split_file'];assert split.is_file() and sha(split)==job['split_sha256']
    assert (W/'markers/runs'/(job['id']+'.json')).exists(),'Required RF run failed'
    inventory=read(W.parent/'G01/audit/v3_cleaning_inventory.json')
    keys,_=scope['all_keys'](inventory['files']);scope['W']=W
    rows=scope['measure'](c,seed,keys);protocol_guard()
    summary=W/'audit/near_duplicate_exposure'/f'{c}_{seed}_summary.json';flags=summary.with_name(f'{c}_{seed}_flows.csv.gz')
    dump(W/'markers/exposure'/f'{c}_{seed}.json',{'time':now(),'condition':c,'seed':seed,'split_sha256':job['split_sha256'],'summary_sha256':sha(summary),'flags_sha256':sha(flags)})
    print(json.dumps({'condition':c,'seed':seed,'summary_rows':len(rows),'completed':True}),flush=True)

if __name__=='__main__':main()
