"""Freeze rules and inputs, then compute every requested fit-full array gate."""
import ast,csv,importlib.metadata,json,os,platform,shutil,sys
from pathlib import Path
import numpy as np
from common import *

def main():
 cpu_only();assert importlib.metadata.version('scikit-learn')=='1.1.3'
 for name in ['class_map.json','final_classes.json','E2_file_to_group.csv']:shutil.copyfile(G05/'config'/name,W/'config'/name)
 src=(G01/'scripts/run_experiment.py').read_text();tree=ast.parse(src)
 funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['traditional','metrics']]
 out='"""Exact G01 function source, extracted without arithmetic changes."""\nimport numpy as np\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.dummy import DummyClassifier\nfrom sklearn.metrics import accuracy_score,f1_score,confusion_matrix,precision_recall_fscore_support\n\n'+'\n\n'.join(ast.get_source_segment(src,n) for n in funcs)+'\n'
 (W/'scripts/source_rf.py').write_text(out);shutil.copyfile(G01/'scripts/pcap_stream.py',W/'scripts/pcap_stream.py')
 inputs={'G01':['scripts/run_experiment.py','scripts/pcap_stream.py','data/iscx_v3.npz','audit/sibling_pairs.csv','audit/v3_cleaning_inventory.json','audit/evidence_sessions.json'],
 'G05':['scripts/common.py','scripts/run_g05.py','scripts/source_g01.py','config/class_map.json','config/final_classes.json','config/E2_file_to_group.csv']}
 for c in CONDITIONS:
  for seed in SEEDS:
   for p in source_paths(c,seed):
    root=G01 if c=='leak' else G05;scope='G01' if c=='leak' else 'G05';inputs[scope].append(p.relative_to(root).as_posix())
 frozen={scope:[{'path':p,'sha256':sha(root/p)} for p in sorted(set(paths))] for scope,paths in inputs.items() for root in [G01 if scope=='G01' else G05]}
 frozen['G06']=[{'path':p.relative_to(W).as_posix(),'sha256':sha(p)} for p in sorted([W/'RULES_G06.md',*W.joinpath('scripts').glob('*.py'),*W.joinpath('config').glob('*.json'),W/'config/E2_file_to_group.csv']) if p.name!='frozen.json']
 dump(W/'config/frozen.json',frozen);dump(W/'markers/RULES_FROZEN.json',{'time':now(),'rules_sha256':sha(W/'RULES_G06.md'),'frozen_manifest_sha256':sha(W/'config/frozen.json'),'models_not_started':True})
 env={'python':platform.python_version(),'packages':{k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','joblib','threadpoolctl']},'GPU_used':False,'PYTHONNOUSERSITE':'1','numerical_threads':1,'RF_n_jobs':12,'cpu_affinity_core_count':len(os.sched_getaffinity(0))}
 dump(W/'audit/environment.json',env)
 expected=read(W/'config/expected_hashes.json');cells=[];bad=[]
 for c in CONDITIONS:
  data,classes,groups=load_data(c);y=data['label']
  with gzip.open(W/'audit'/f'{c}_row_manifest.csv.gz','wt',newline='') as f:
   writer=csv.writer(f);writer.writerow(['row','flow_id','session','file','true_label'])
   for i in range(len(y)):writer.writerow([i,data['flow_id'][i],groups[i],data['file'][i],int(y[i])])
  for seed in SEEDS:
   split,ref,pred=source_paths(c,seed);tag=f'{c}_{seed}';arrays={};h=hashlib.sha256();details=[];seen=np.zeros(len(y),dtype=int)
   copied=W/'references'/tag;copied.mkdir(exist_ok=True)
   for p,n in [(split,'original_splits.npz'),(ref,'original.json'),(pred,'original_predictions.csv.gz')]:shutil.copyfile(p,copied/n)
   with np.load(split,allow_pickle=False) as z:
    for fold in range(5):
     test=z[f'fold{fold}_test'].copy();fit=fit_full(y,test,seed,fold);oldfit=z[f'fold{fold}_fit'];val=z[f'fold{fold}_validation'];seen[test]+=1
     arrays[f'fold{fold}_fit_full']=fit;arrays[f'fold{fold}_test']=test
     for a in [fit,test]:h.update(np.asarray(a,dtype='<i8').tobytes())
     ok=set(np.unique(y[fit]))==set(range(9)) and not np.intersect1d(fit,test).size and len(np.unique(fit))==len(fit)
     perclass=[{'class':classes[k],'old_fit':int(sum(y[oldfit]==k)),'old_validation':int(sum(y[val]==k)),'fit_full':int(sum(y[fit]==k)),'test':int(sum(y[test]==k))} for k in range(9)]
     ok=ok and all(x['fit_full']<=1000 for x in perclass)
     details.append({'fold':fold,'old_fit_count':len(oldfit),'old_validation_count':len(val),'fit_full_count':len(fit),'test_count':len(test),'all_9_classes_and_disjoint':bool(ok),'per_class':perclass,'test_source_array_equal':bool(np.array_equal(test,z[f'fold{fold}_test']))})
   dest=W/'data/splits'/f'G06_{c}_{seed}_fitfull_splits.npz';np.savez_compressed(dest,**arrays)
   digest=h.hexdigest();ok=digest==expected[tag] and all(x['all_9_classes_and_disjoint'] for x in details) and bool(np.all(seen==1))
   cell={'condition':c,'seed':seed,'n_flows':len(y),'expected_sha256':expected[tag],'actual_sha256':digest,'passed':bool(ok),'split_file':dest.relative_to(W).as_posix(),'npz_sha256':sha(dest),'source_scope':'G01' if c=='leak' else 'G05','source_split':split.relative_to(G01 if c=='leak' else G05).as_posix(),'source_split_sha256':sha(split),'reference_json_sha256':sha(ref),'reference_predictions_sha256':sha(pred),'folds':details};cells.append(cell)
   if not ok:bad.append(tag)
 summary={'time':now(),'all_passed':not bad,'failed_cells':bad,'cells':cells,'digest_order':'fold0..4: fit_full then test, each <i8','rules_fixed_before_models':True};dump(W/'split_check.json',summary)
 guard()
 if bad:dump(W/'markers/A_STOPPED.json',{'stage':'split_precheck','failed_cells':bad,'B_continues':True})
 else:dump(W/'markers/SPLIT_CHECK_PASSED.json',{'time':now(),'cells':15})
 print(json.dumps({'all_15_split_checks_passed':not bad,'failed_cells':bad}),flush=True)
if __name__=='__main__':main()
