"""One repaired-split G05 run; invoke unchanged G01 RF or G02 deep function."""
import argparse
import csv
import gzip
import json
import os
import warnings
import numpy as np
from common import W, check_splits, dump, load_data, now, protocol_guard, read, sha
from source_g01 import traditional, deep as fixed_deep
from source_g02 import deep as early_deep, metrics
warnings.simplefilter('always')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);args=ap.parse_args()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:16])
    protocol_guard()
    job=next(j for j in read(W/'config/queue_plan.json')['jobs'] if j['id']==args.job)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in (['2','3'] if job['kind']=='deep' else [''])
    protocol_hash=sha(W/'config/protocol.json')
    split_file=W/job['split_file'];assert sha(split_file)==job['split_sha256']
    result_file=W/'results/v3'/(job['id']+'.json')
    pred_path=W/'results/v3'/(job['id']+'_predictions.csv.gz')
    marker=W/'markers/runs'/(job['id']+'.json')
    if marker.exists():
        saved=read(marker)
        assert saved['protocol_sha256']==protocol_hash and sha(result_file)==saved['result_sha256'] and sha(pred_path)==saved['predictions_sha256']
        print(json.dumps({'job':job['id'],'already_completed':True}),flush=True);return
    data,classes,groups=load_data(job['condition'])
    split_audit=check_splits(data,groups,split_file,job['condition'])
    n=len(groups);prob=np.full((n,len(classes)),np.nan,dtype=np.float32);fold_numbers=np.full(n,-1,dtype=np.int8)
    histories=[];resumed=[]
    checkpoint_dir=W/'checkpoints'/job['id'];checkpoint_dir.mkdir(parents=True,exist_ok=True)
    with np.load(split_file,allow_pickle=False) as splits:
        for fold in range(5):
            fit,val,test=[splits[f'fold{fold}_{part}'] for part in ['fit','validation','test']]
            assert np.all(fold_numbers[test]==-1)
            meta_path=checkpoint_dir/f'fold{fold}.json';prob_path=checkpoint_dir/f'fold{fold}.npz';done_path=checkpoint_dir/f'fold{fold}.done.json'
            if done_path.exists():
                done=read(done_path)
                assert done['protocol_sha256']==protocol_hash and done['split_sha256']==job['split_sha256']
                assert sha(meta_path)==done['metadata_sha256'] and sha(prob_path)==done['probabilities_sha256']
                extra=read(meta_path)
                with np.load(prob_path,allow_pickle=False) as cached:
                    assert np.array_equal(cached['test'],test);p=cached['probabilities']
                resumed.append(fold)
            else:
                if job['kind']=='deep':
                    if job['patience'] is None:
                        p,extra=fixed_deep(data,fit,val,test,job,len(classes),fold)
                        extra.update(stop_epoch=job['max_epochs'],stop_reason='fixed_epoch_budget',patience=None,test_inference_count=1,stopped_at_epoch_300=False)
                    else:
                        p,extra=early_deep(data,fit,val,test,job,len(classes),fold)
                else:
                    p,extra=traditional(data,fit,test,job,len(classes),fold)
                    extra.update(selected_epoch=None,stop_epoch=None,validation_curve=[],model_seed=job['seed']+fold*10007,
                                 test_inference_count=1,original_G01_estimator_used=True,normalization={'normalization':False})
                extra={'fold':fold,'n_fit':len(fit),'n_validation':len(val),'n_test':len(test),**extra}
                temp=checkpoint_dir/f'fold{fold}.tmp.npz';np.savez_compressed(temp,test=test,probabilities=p);temp.replace(prob_path)
                dump(meta_path,extra)
                dump(done_path,{'time':now(),'protocol_sha256':protocol_hash,'split_sha256':job['split_sha256'],
                    'metadata_sha256':sha(meta_path),'probabilities_sha256':sha(prob_path)})
                print(json.dumps({'job':job['id'],'fold':fold,'completed':True,'selected_epoch':extra['selected_epoch'],'stop_epoch':extra['stop_epoch']}),flush=True)
            assert p.shape==(len(test),len(classes)) and np.isfinite(p).all() and np.allclose(p.sum(1),1.)
            prob[test]=p;fold_numbers[test]=fold;histories.append(extra)
    assert np.all(fold_numbers>=0) and np.isfinite(prob).all()
    predictions=prob.argmax(1)
    result={'job':job,'status':'completed','post_G01_G02_G03_G04_second_external_review_addition':True,'rules_fixed_before_training':True,'classes':classes,
      'n_flows':n,'metrics':metrics(data['label'],predictions,classes),'fold_details':histories,'protocol_sha256':protocol_hash,
      'split_sha256':job['split_sha256'],'metric_aggregation':'merged five-fold OOF predictions','tie_break_epoch':'first maximum',
      'shared_G05_split_arrays_reused_verbatim':True,'resumed_completed_folds':resumed,'grouping':job['condition'],'split_audit':split_audit}
    temp_pred=pred_path.with_suffix('.tmp')
    with gzip.open(temp_pred,'wt',newline='') as f:
        writer=csv.writer(f);writer.writerow(['flow_id','session','file','fold','true_label','pred_label']+[f'prob_{i}' for i in range(len(classes))])
        for i in range(n):writer.writerow([data['flow_id'][i],groups[i],data['file'][i],int(fold_numbers[i]),int(data['label'][i]),int(predictions[i])]+[float(x) for x in prob[i]])
    temp_pred.replace(pred_path);dump(result_file,result);protocol_guard()
    dump(marker,{'job':job['id'],'time':now(),'status':'completed','protocol_sha256':protocol_hash,
        'result_sha256':sha(result_file),'predictions_sha256':sha(pred_path),'n_flows':n})
    print(json.dumps({'job':job['id'],'completed':True,'macro_f1':result['metrics']['macro_f1'],'accuracy':result['metrics']['accuracy']}),flush=True)

if __name__=='__main__':main()
