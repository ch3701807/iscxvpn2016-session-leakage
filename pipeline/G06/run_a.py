"""Original RF implementation; only fit indices change in the 15 official runs."""
import argparse
from common import *
from source_rf import traditional,metrics

def execute(condition,seed,reproduce=False):
 guard();assert read(W/'split_check.json')['all_passed']
 if not reproduce:assert read(W/'audit/environment_reproduction.json')['A_allowed']
 jobid='G06_environment_reproduction' if reproduce else f'G06_{condition}_multi_RF_seq_iat_fitfull_{seed}'
 marker=W/'markers/runs'/f'{jobid}.json';output=W/'results'/f'{jobid}.json';predpath=W/'results'/f'{jobid}_predictions.csv.gz'
 if marker.exists():
  m=read(marker);assert sha(output)==m['json_sha256'] and sha(predpath)==m['predictions_sha256'];return
 d,classes,groups=load_data(condition);n=len(groups);y=d['label'];reference=W/'references'/f'{condition}_{seed}'
 source=reference/'original_splits.npz' if reproduce else W/'data/splits'/f'G06_{condition}_{seed}_fitfull_splits.npz'
 splitsha=sha(source);frozen_sha=sha(W/'config/frozen.json');prob=np.full((n,9),np.nan,dtype=np.float32);foldnum=np.full(n,-1,dtype=np.int8);details=[]
 checkpoint=W/'checkpoints'/jobid;checkpoint.mkdir(exist_ok=True)
 with np.load(source,allow_pickle=False) as z,np.load(reference/'original_splits.npz',allow_pickle=False) as orig:
  for fold in range(5):
   fit=z[f'fold{fold}_fit' if reproduce else f'fold{fold}_fit_full'];test=z[f'fold{fold}_test'];assert np.array_equal(test,orig[f'fold{fold}_test'])
   cached=checkpoint/f'fold{fold}.npz';meta=checkpoint/f'fold{fold}.json';done=checkpoint/f'fold{fold}.done.json'
   if done.exists():
    m=read(done);assert m['split_sha256']==splitsha and m['frozen_sha256']==frozen_sha and sha(cached)==m['npz_sha256'] and sha(meta)==m['json_sha256']
    with np.load(cached,allow_pickle=False) as v:assert np.array_equal(v['test'],test);p=v['probabilities']
    info=read(meta)
   else:
    p,_=traditional(d,fit,test,{'model':'RF_seq_iat','seed':seed,'id':jobid},9,fold)
    info={'fold':fold,'original_fit_count':len(orig[f'fold{fold}_fit']),'original_validation_count':len(orig[f'fold{fold}_validation']),'fit_count':len(fit),'test_count':len(test),'model_seed':seed+fold*10007,'all_9_classes':bool(np.array_equal(np.unique(y[fit]),np.arange(9))),'test_unchanged':True,'n_estimators':300,'n_jobs':12}
    temp=cached.with_name(cached.stem+'.tmp.npz');np.savez_compressed(temp,test=test,probabilities=p);temp.replace(cached);dump(meta,info)
    dump(done,{'time':now(),'split_sha256':splitsha,'frozen_sha256':frozen_sha,'npz_sha256':sha(cached),'json_sha256':sha(meta)})
   prob[test]=p;foldnum[test]=fold;details.append(info);print(json.dumps({'job':jobid,'fold':fold,'completed':True}),flush=True)
 assert np.isfinite(prob).all() and np.all(foldnum>=0) and np.allclose(prob.sum(1),1)
 header,rows,refprob=read_predictions(reference/'original_predictions.csv.gz')
 assert header==['flow_id','session','file','fold','true_label','pred_label']+[f'prob_{i}' for i in range(9)]
 for i,r in enumerate(rows):assert r['flow_id']==d['flow_id'][i] and r['file']==d['file'][i] and r['session']==groups[i] and int(r['true_label'])==y[i] and int(r['fold'])==foldnum[i]
 pred=prob.argmax(1);result={'job':{'id':jobid,'condition':condition,'seed':seed,'model':'RF_seq_iat','fit_policy':'original_fit' if reproduce else 'fit_full'},'status':'completed','classes':classes,'n_flows':n,'metrics':metrics(y,pred,classes),'fold_details':details,'split_sha256':splitsha,'frozen_manifest_sha256':frozen_sha,'CPU_only':True,'post_G05_third_external_review_addition':True,'rules_fixed_before_models':True}
 refresult=read(reference/'original.json');result['reference']={'task':'G01' if condition=='leak' else 'G05','json_sha256':sha(reference/'original.json'),'macro_f1':refresult['metrics']['macro_f1'],'accuracy':refresult['metrics']['accuracy']}
 result['paired_difference']={k:result['metrics'][k]-refresult['metrics'][k] for k in ['macro_f1','accuracy']}
 temp=predpath.with_suffix('.tmp')
 with gzip.open(temp,'wt',newline='') as f:
  writer=csv.writer(f);writer.writerow(header)
  for i in range(n):writer.writerow([d['flow_id'][i],groups[i],d['file'][i],int(foldnum[i]),int(y[i]),int(pred[i])]+[float(x) for x in prob[i]])
 temp.replace(predpath);dump(output,result)
 if reproduce:
  different=np.any(prob.view(np.uint32)!=refprob.view(np.uint32),axis=1);delta=result['paired_difference']['macro_f1']
  comp={'time':now(),'original_condition':'Pstar','seed':1,'original_fit_and_test_reused':True,'probability_dtype':'float32','bitwise_equal':bool(not different.any()),'different_flow_count':int(different.sum()),'different_probability_cell_count':int(np.count_nonzero(prob.view(np.uint32)!=refprob.view(np.uint32))),'max_abs_probability_difference':float(np.max(np.abs(prob-refprob))),'original_macro_f1':refresult['metrics']['macro_f1'],'reproduced_macro_f1':result['metrics']['macro_f1'],'macro_f1_difference':delta,'macro_f1_difference_percentage_points':delta*100,'allowed_absolute_difference':0.001,'A_allowed':abs(delta)<=0.001,'original_predictions_sha256':sha(reference/'original_predictions.csv.gz'),'reproduced_predictions_sha256':sha(predpath)}
  dump(W/'audit/environment_reproduction.json',comp)
  if not comp['A_allowed']:dump(W/'markers/A_STOPPED.json',{'stage':'environment_reproduction','comparison':comp,'B_continues':True})
 guard();dump(marker,{'time':now(),'status':'completed','job':jobid,'json_sha256':sha(output),'predictions_sha256':sha(predpath)})
 print(json.dumps({'job':jobid,'completed':True,'macro_f1':result['metrics']['macro_f1'],'accuracy':result['metrics']['accuracy']}),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=CONDITIONS,default='Pstar');p.add_argument('--seed',type=int,choices=SEEDS,default=1);p.add_argument('--reproduce',action='store_true');a=p.parse_args();execute(a.condition,a.seed,a.reproduce)
if __name__=='__main__':main()
