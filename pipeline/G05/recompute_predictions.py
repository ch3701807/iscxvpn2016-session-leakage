"""Offline stage/full verification using numpy, without GPU, network or raw features."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with gzip.open(p,'rt',newline='') as f:return list(csv.DictReader(f))

def split_digest(z):
    h=hashlib.sha256()
    for fold in range(5):
        for part in ['fit','validation','test']:h.update(np.asarray(z[f'fold{fold}_{part}'],dtype='<i8').tobytes())
    return h.hexdigest()

def verify_exposure(root,conditions):
    checked=[];missing=[]
    for c in conditions:
        for seed in [1,7,42]:
            path=root/'audit/near_duplicate_exposure'/f'{c}_{seed}_flows.csv.gz'
            if not path.exists():
                assert (root/'markers/stage_failures'/f'exposure_{c}_{seed}.json').exists()
                missing.append({'condition':c,'seed':seed});continue
            flags=rows(path);preds=rows(root/'results/v3'/f'G05_{c}_multi_RF_seq_iat_{seed}_predictions.csv.gz')
            assert len(flags)==len(preds)
            marker=read(root/'markers/exposure'/f'{c}_{seed}.json')
            summaries=path.with_name(f'{c}_{seed}_summary.json')
            assert sha(path)==marker['flags_sha256'] and sha(summaries)==marker['summary_sha256']
            for a,b in zip(flags,preds):
                assert all(a[k]==b[k] for k in ['flow_id','file','session','fold','true_label','pred_label'])
                assert int(a['correct'])==int(a['true_label']==a['pred_label'])
            classes=read(root/'results/v3'/f'G05_{c}_multi_RF_seq_iat_{seed}.json')['classes']
            summary=read(summaries)
            for entry in summary:
                selected=[r for r in flags if (entry['fold'] is None or int(r['fold'])==entry['fold']) and (entry['class']=='__all__' or classes[int(r['true_label'])]==entry['class'])]
                n=len(selected);assert n==entry['n_test'];definition=entry['definition']
                a=np.array([int(r[definition+'_same_class']) for r in selected],dtype=bool);b=np.array([int(r[definition+'_other_class']) for r in selected],dtype=bool)
                correct=np.array([int(r['correct']) for r in selected],dtype=bool);exposed=a|b
                assert all(bool(int(r[definition+'_any']))==bool(v) for r,v in zip(selected,exposed))
                counts={'same_class_only':int((a&~b).sum()),'other_class_only':int((~a&b).sum()),'both_classes':int((a&b).sum()),'neither':int((~exposed).sum()),'same_class_exposed':int(a.sum()),'other_class_exposed':int(b.sum()),'any_exposed':int(exposed.sum()),'exposed_correct':int((correct&exposed).sum()),'unexposed_correct':int((correct&~exposed).sum())}
                assert all(entry[k]==v for k,v in counts.items())
                for k,num,den in [('any_exposure_rate',counts['any_exposed'],n),('same_class_exposure_rate',counts['same_class_exposed'],n),('other_class_exposure_rate',counts['other_class_exposed'],n),('exposed_accuracy',counts['exposed_correct'],counts['any_exposed']),('unexposed_accuracy',counts['unexposed_correct'],counts['neither'])]:assert entry[k]==(num/den if den else None)
            checked.append({'condition':c,'seed':seed,'rows':len(flags),'summary_rows':len(summary)})
    aggregate=root/'audit/near_duplicate_exposure_summary.json'
    if aggregate.exists():
        for entry in read(aggregate):
            for metric in ['any_exposure_rate','same_class_exposure_rate','other_class_exposure_rate','exposed_accuracy','unexposed_accuracy']:
                v=[r[metric] for r in entry['per_seed'] if r[metric] is not None]
                assert entry[metric]=={'n':len(v),'mean':float(np.mean(v)) if v else None,'std':float(np.std(v,ddof=1)) if len(v)>1 else None}
    return {'completed':len(checked),'failed':missing,'checks':checked,'summary_counts_rates_accuracy_verified':True}

def verify(root,priorities=None):
    root=Path(root)
    manifest=root/'SHA256SUMS'
    if manifest.exists():
        entries={}
        for line in manifest.read_text().splitlines():
            h,name=line.split('  ',1);assert name not in entries and not Path(name).is_absolute() and '..' not in Path(name).parts;entries[name]=h
        actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p!=manifest}
        assert actual==set(entries),'Manifest does not cover the exact file collection'
        for name,h in entries.items():assert sha(root/name)==h,name
    if priorities is None:priorities=read(root/'audit/package_scope.json')['priorities']
    for entry in read(root/'config/frozen_inputs.json')['g05_frozen']:assert sha(root/entry['path'])==entry['sha256'],'Frozen scientific input differs: '+entry['path']
    assert read(root/'audit/environment.json')['scikit-learn']=='1.1.3'
    checks=read(root/'split_check.json');expected=read(root/'config/expected_split_hashes.json');assert checks['all_checks_passed'] and len(checks['checks'])==30
    rosters={};splits={};count_rows=[]
    for entry in checks['checks']:
        condition=entry['condition'];seed=entry['seed'];classes=entry['classes']
        if condition not in rosters:rosters[condition]=rows(root/'audit'/f'{condition}_row_manifest.csv.gz')
        roster=rosters[condition];n=len(roster);groups=np.array([r['session'] for r in roster]);truth=np.array([int(r['true_label']) for r in roster])
        assert n==entry['n_flows'] and len(set(groups))==entry['groups_actual']==expected[condition]['groups']
        path=root/entry['split_file'];assert sha(path)==entry['split_sha256']
        folded=np.full(n,-1,dtype=int);covered=np.zeros(n,dtype=int)
        with np.load(path,allow_pickle=False) as z:
            assert split_digest(z)==expected[condition]['hashes'][str(seed)]==entry['digest_actual']==entry['digest_expected']
            assert set(z.files)=={f'fold{f}_{p}' for f in range(5) for p in ['fit','validation','test']}
            sizes=[]
            for fold in range(5):
                parts={p:z[f'fold{fold}_{p}'] for p in ['fit','validation','test']};fit,val,test=[parts[p] for p in ['fit','validation','test']]
                for a in parts.values():assert a.ndim==1 and a.dtype.kind in 'iu' and len(a)>0 and a.min()>=0 and a.max()<n and len(set(a))==len(a)
                assert not(set(fit)&set(val) or set(fit)&set(test) or set(val)&set(test))
                assert not(set(groups[fit])&set(groups[val]) or set(groups[fit])&set(groups[test]) or set(groups[val])&set(groups[test]))
                train=np.setdiff1d(np.arange(n),test)
                for c,name in enumerate(classes):
                    row={'condition':condition,'seed':seed,'fold':fold,'class':name,'class_index':c}
                    for p,idx in list(parts.items())+[('outer_train',train)]:
                        a=idx[truth[idx]==c];row[p+'_flows']=len(a);row[p+'_groups']=len(np.unique(groups[a]))
                    assert row['outer_train_groups']>=1 and row['fit_groups']>=1 and row['fit_flows']<=1000
                    assert row['validation_groups']==(1 if row['outer_train_groups']>1 else 0)
                    count_rows.append({k:str(v) for k,v in row.items()})
                covered[test]+=1;folded[test]=fold;sizes.append({p:len(a) for p,a in parts.items()})
        assert np.all(covered==1);splits[condition,seed]=(folded,sizes)
    with (root/'audit/fold_class_coverage.csv').open(newline='') as f:actual_counts=list(csv.DictReader(f))
    sort=lambda r:(r['condition'],r['seed'],r['fold'],r['class'])
    assert sorted(count_rows,key=sort)==sorted(actual_counts,key=sort)
    jobs=[j for j in read(root/'config/queue_plan.json')['jobs'] if j['priority'] in priorities];verified=[];failed=[]
    for job in jobs:
        rp=root/'results/v3'/(job['id']+'.json');pp=root/'results/v3'/(job['id']+'_predictions.csv.gz')
        if not rp.exists():
            assert (root/'markers/runs'/(job['id']+'.failed.json')).exists();failed.append(job['id']);continue
        result=read(rp);marker=read(root/'markers/runs'/(job['id']+'.json'))
        assert result['status']=='completed' and result['job']==job
        assert sha(rp)==marker['result_sha256'] and sha(pp)==marker['predictions_sha256']
        assert result['protocol_sha256']==marker['protocol_sha256']==sha(root/'config/protocol.json')
        assert result['split_sha256']==job['split_sha256']
        classes=result['classes'];nclasses=len(classes);roster=rosters[job['condition']];folded,sizes=splits[job['condition'],job['seed']]
        predictions=rows(pp);assert len(predictions)==len(roster)==result['n_flows'];ids=set();cm=np.zeros((nclasses,nclasses),dtype=np.int64)
        for i,(row,original) in enumerate(zip(predictions,roster)):
            assert all(row[k]==original[k] for k in ['flow_id','file','session','true_label'])
            assert row['flow_id'] not in ids;ids.add(row['flow_id']);assert int(row['fold'])==folded[i]
            p=np.array([float(row[f'prob_{c}']) for c in range(nclasses)])
            assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1 and abs(p.sum()-1)<1e-5
            pred=int(row['pred_label']);assert pred==p.argmax();cm[int(row['true_label']),pred]+=1
        tp=np.diag(cm).astype(float);den=cm.sum(0)+cm.sum(1);f=np.divide(2*tp,den,out=np.zeros(nclasses),where=den!=0)
        macro=float(f.mean());acc=float(tp.sum()/cm.sum());m=result['metrics']
        assert abs(macro-m['macro_f1'])<2e-15 and acc==m['accuracy'] and cm.tolist()==m['confusion_matrix']
        precision=np.divide(tp,cm.sum(0),out=np.zeros(nclasses),where=cm.sum(0)!=0);recall=np.divide(tp,cm.sum(1),out=np.zeros(nclasses),where=cm.sum(1)!=0)
        for c,d in enumerate(m['per_class']):assert d['class']==classes[c] and d['support']==int(cm[c].sum()) and abs(d['precision']-precision[c])<2e-15 and abs(d['recall']-recall[c])<2e-15 and abs(d['f1']-f[c])<2e-15
        assert len(result['fold_details'])==5
        for fold,d in enumerate(result['fold_details']):
            assert d['fold']==fold and d['model_seed']==job['seed']+fold*10007
            for part,key in [('fit','n_fit'),('validation','n_validation'),('test','n_test')]:assert d[key]==sizes[fold][part]
            if job['kind']=='baseline':assert d['validation_curve']==[] and d['selected_epoch'] is None and d['stop_epoch'] is None;continue
            curve=d['validation_curve'];scores=[v['validation_macro_f1'] for v in curve]
            assert [v['epoch'] for v in curve]==list(range(1,len(curve)+1)) and np.isfinite(scores).all()
            assert d['selected_epoch']==scores.index(max(scores))+1 and d['selected_validation_macro_f1']==max(scores) and d['stop_epoch']==len(scores)
            assert d['test_inference_count']==1 and d['normalization']['normalization']==(job['condition']!='no_norm')
            if job['patience'] is None:assert len(scores)==job['max_epochs'] and d['stop_reason']=='fixed_epoch_budget'
            else:
                best=-1.;best_epoch=0;stop=None
                for epoch,score in enumerate(scores,1):
                    if score>best:best,best_epoch=score,epoch
                    if epoch-best_epoch>=30:stop=epoch;break
                assert len(scores)==(stop if stop else 300) and d['stop_reason']==('patience_30' if stop else 'epoch_cap')
            assert d['deterministic_settings']=={'cudnn.deterministic':True,'cudnn.benchmark':False,'deterministic_algorithms':True,'warn_only':True,'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        verified.append({'job':job['id'],'rows':len(predictions),'macro_f1':m['macro_f1'],'accuracy':m['accuracy']})
    assert len(verified)+len(failed)==len(jobs)
    exposure=verify_exposure(root,['Pstar','no_session','evidence']) if 1 in priorities else {'included':False}
    return {'task':'G05','priorities':priorities,'completed_runs_verified':len(verified),'failed_runs':failed,'total_prediction_rows':sum(r['rows'] for r in verified),
        'all_30_split_digests_and_coverages_verified':True,'sklearn_version':'1.1.3','every_flow_predicted_once':True,'every_class_has_fit_group':True,
        'selection_and_first_early_stop_verified':True,'frozen_G05_scientific_inputs_verified':True,'exposure':exposure,'runs':verified,'all_verified':True}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--priority',choices=['1','2','3','4','all']);a=ap.parse_args()
    priorities=([1,2,3,4] if a.priority=='all' else [int(a.priority)]) if a.priority else None
    print(json.dumps(verify(Path(a.root),priorities),ensure_ascii=False))
