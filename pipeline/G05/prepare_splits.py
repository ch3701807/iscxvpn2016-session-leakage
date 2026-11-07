"""Pretrain 30-cell exact digest/coverage gate. No estimator is fitted here."""
import csv
import gzip
import hashlib
import importlib.metadata
import os
import numpy as np
from common import W,check_splits,dump,load_data,now,read,sha
from fixed_splitter import folds
import source_g01

def digest(parts):
    h=hashlib.sha256()
    for fit,val,test in parts:
        for a in (fit,val,test):h.update(np.asarray(a,dtype='<i8').tobytes())
    return h.hexdigest()

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    assert not (W/'markers/PRECHECK_PASSED.json').exists()
    assert not list((W/'results/v3').glob('*'))
    assert read(W/'markers/SPLIT_RULES_FROZEN.json')['sha256']==sha(W/'RULES_G05.md')
    version=importlib.metadata.version('scikit-learn');assert version=='1.1.3',version
    dump(W/'config/paths.json',{'g01_root':'../G01','handoff_directory':'../../paper/claude/BlockSys'})
    expected=read(W/'config/expected_split_hashes.json')
    dump(W/'markers/EXPECTED_HASHES_FROZEN.json',{'time':now(),'sha256':sha(W/'config/expected_split_hashes.json'),'models_trained':0})
    source_g01.W=W.parent/'G01'
    checks=[];counts=[];errors=[]
    for condition,spec in expected.items():
        data,classes,groups=load_data(condition);actual_groups=len(np.unique(groups))
        if condition not in ['E2','Pstar_8','E3_8']:
            d0,c0,g0=source_g01.load({'condition':condition,'task':'multi'})
            assert classes==c0 and np.array_equal(groups,g0)
            assert set(data)==set(d0) and all(np.array_equal(data[k],d0[k]) for k in data)
        with gzip.open(W/'audit'/f'{condition}_row_manifest.csv.gz','wt',newline='') as f:
            writer=csv.writer(f);writer.writerow(['flow_id','file','session','true_label'])
            writer.writerows((data['flow_id'][i],data['file'][i],groups[i],int(data['label'][i])) for i in range(len(groups)))
        for seed in [1,7,42]:
            result=list(folds(data,groups,{'condition':condition,'seed':seed}))
            triples=[(fit,val,test) for _,fit,val,test in result]
            arrays={f'fold{fold}_{part}':a for fold,fit,val,test in result for part,a in [('fit',fit),('validation',val),('test',test)]}
            path=W/'data/splits'/f'G05_{condition}_{seed}_splits.npz';np.savez_compressed(path,**arrays)
            actual=digest(triples);want=spec['hashes'][str(seed)]
            entry={'condition':condition,'seed':seed,'groups_expected':spec['groups'],'groups_actual':actual_groups,'groups_match':actual_groups==spec['groups'],
                'digest_expected':want,'digest_actual':actual,'digest_match':actual==want,'split_file':path.relative_to(W).as_posix(),'split_sha256':sha(path),
                'n_flows':len(groups),'classes':classes,'outer_random_state':seed,'splitter':'user_supplied_FixedSGKF','inner_rng':'seed+fold*10007'}
            try:
                details=check_splits(data,groups,path,condition)
                entry.update(coverage_passed=True,folds=details)
                for d in details:
                    for c,row in enumerate(d['per_class']):counts.append({'condition':condition,'seed':seed,'fold':d['fold'],'class':classes[c],**row})
            except AssertionError as exc:entry.update(coverage_passed=False,coverage_error=str(exc))
            if not (entry['groups_match'] and entry['digest_match'] and entry['coverage_passed']):errors.append(entry)
            checks.append(entry)
            print(__import__('json').dumps({k:entry[k] for k in ['condition','seed','digest_match','groups_match','coverage_passed','digest_actual']}),flush=True)
    for c in ['no_sni','no_norm']:
        for seed in [1,7,42]:
            assert (W/'data/splits'/f'G05_{c}_{seed}_splits.npz').read_bytes()==(W/'data/splits'/f'G05_Pstar_{seed}_splits.npz').read_bytes()
    out={'task':'G05','time':now(),'models_trained':0,'sklearn_version':version,'rules_sha256':sha(W/'RULES_G05.md'),
        'expected_table_sha256':sha(W/'config/expected_split_hashes.json'),'all_checks_passed':not errors,'cells':30,'checks':checks,'mismatches':errors,
        'Pstar_no_sni_no_norm_splits_byte_identical':True,'digest_encoding':'fold0..4: fit,validation,test; concatenate little-endian int64 bytes'}
    dump(W/'split_check.json',out)
    if counts:
        with (W/'audit/fold_class_coverage.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(counts[0]));writer.writeheader();writer.writerows(counts)
    if errors:
        dump(W/'markers/STOPPED.json',{'time':now(),'stage':'pretraining_split_gate','mismatches':len(errors),'training_started':False})
        lines=['# G05 前置检查未通过','','没有启动训练。全部已算数组保存于 data/splits/。','','| 条件 | seed | 预期 SHA-256 | 实际 SHA-256 | 组数预期/实际 | 覆盖 |','|---|---:|---|---|---|---|']
        for r in errors:lines.append(f"| {r['condition']} | {r['seed']} | {r['digest_expected']} | {r['digest_actual']} | {r['groups_expected']}/{r['groups_actual']} | {r['coverage_passed']} |")
        (W/'RESULT.md').write_text('\n'.join(lines)+'\n')
        raise SystemExit(3)
    dump(W/'markers/PRECHECK_PASSED.json',{'time':now(),'cells':30,'all_digest_groups_coverage_checks_passed':True,'split_check_sha256':sha(W/'split_check.json'),'training_started':False})

if __name__=='__main__':main()
