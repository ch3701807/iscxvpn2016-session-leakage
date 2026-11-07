"""One fixed-protocol CV experiment. Each seed merges all five held-out folds."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,csv,gzip,hashlib,json,random,sys,time,warnings
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold,StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
W=Path(__file__).resolve().parents[1]
warnings.simplefilter('always')

def dump(p,data):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)

def load(j):
    data=dict(np.load(W/'data/iscx_v3.npz',allow_pickle=False))
    data['file']=data.pop('source_file')
    final=json.loads((W/'config/final_classes.json').read_text())
    muse=Path(json.loads((W/'config/paths.local.json').read_text())['muse_root'])
    orig=json.loads((muse/'data/iscx_classes_v2.json').read_text())
    selected=np.ones(len(data['label']),dtype=bool)
    if j['condition'] not in ['no_clean','interaction']:selected&=data['keep_clean']
    if j['task']=='multi':
        names=final['main_classes'];ids=[orig[n] for n in names];selected&=np.isin(data['label'],ids)
    else:names=['non_vpn','vpn'];ids=[0,1]
    data={k:v[selected] for k,v in data.items()}
    source_labels=data['label'].copy()
    if j['task']=='multi':data['label']=np.array([{old:new for new,old in enumerate(ids)}[int(x)] for x in source_labels],dtype=np.int64)
    else:data['label']=data['vpn_label'].copy()
    if j['condition']!='no_sni':data['raw']=data['raw_masked']
    groups=data['session'].copy()
    if j['condition'] in ['no_session','interaction']:groups=data['file'].copy()
    if j['condition']=='evidence':
        mapping=json.loads((W/'audit/evidence_sessions.json').read_text())
        if not mapping['new_edges']:
            return None,names,None
        groups=np.array([mapping['file_to_session'][f] for f in data['file']])
    return data,names,groups

def folds(data,groups,j):
    y=data['label'];seed=j['seed']
    splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed) if j['condition']=='leak' else StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    outer=splitter.split(np.zeros(len(y)),y) if j['condition']=='leak' else splitter.split(np.zeros(len(y)),y,groups)
    for fold,(train,test) in enumerate(outer):
        rng=np.random.default_rng(seed+fold*10007)
        valgroups=set()
        for c in sorted(np.unique(y[train])):
            options=np.unique(groups[train[y[train]==c]])
            if len(options)>1:valgroups.add(str(rng.choice(options)))
        val=train[np.isin(groups[train],list(valgroups))]
        pool=train[~np.isin(groups[train],list(valgroups))]
        capped=[]
        for c in sorted(np.unique(y[pool])):
            p=pool[y[pool]==c].copy();rng.shuffle(p);capped.append(p[:1000])
        fit=np.concatenate(capped);rng.shuffle(fit)
        if j['condition']!='leak':assert not(set(groups[train])&set(groups[test]))
        assert not(set(groups[fit])&set(groups[val]))
        assert not(set(fit)&set(test))
        yield fold,fit,val,test

def metrics(y,p,names):
    labels=np.arange(len(names));p0,r0,f0,s0=precision_recall_fscore_support(y,p,labels=labels,zero_division=0)
    return {'macro_f1':float(f1_score(y,p,labels=labels,average='macro',zero_division=0)),'accuracy':float(accuracy_score(y,p)),'confusion_matrix':confusion_matrix(y,p,labels=labels).tolist(),'per_class':[{'class':name,'precision':float(p0[i]),'recall':float(r0[i]),'f1':float(f0[i]),'support':int(s0[i])} for i,name in enumerate(names)]}

def transform(data,fit,normalized):
    if not normalized:return data,{'normalization':False}
    out=dict(data);out['seq']=np.sign(data['seq'])*np.log1p(np.abs(data['seq']))/np.log1p(1500.)
    logiat=np.log1p(data['iat']);mu=float(logiat[fit].mean(dtype=np.float64));sd=float(logiat[fit].std(dtype=np.float64)) or 1.
    out['iat']=((logiat-mu)/sd).astype(np.float32)
    stat=data['stat'].copy();stat[:,3:5]=np.log1p(stat[:,3:5]);sm=stat[fit].mean(0,dtype=np.float64);ss=stat[fit].std(0,dtype=np.float64);ss[ss==0]=1.
    out['stat']=((stat-sm)/ss).astype(np.float32);out['seq']=out['seq'].astype(np.float32)
    return out,{'normalization':True,'iat_log_mean':mu,'iat_log_std':sd,'iat_fit_axes':'all training rows and 50 positions, including padding','stat_mean':sm.tolist(),'stat_std':ss.tolist(),'stat_log_columns':[3,4],'fit_row_count':len(fit),'raw_already_divided_by_255':True}

def traditional(data,fit,test,j,nclasses,fold):
    model=j['model'];seed=j['seed']+fold*10007
    if model in ['RF_seq_iat','LR_seq_iat']:X=np.concatenate([data['seq'],data['iat']],axis=1).astype(np.float64)
    else:X=data['stat'].astype(np.float64)
    if model.startswith('RF'):est=RandomForestClassifier(n_estimators=300,n_jobs=12,random_state=seed)
    elif model.startswith('LR'):est=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,C=1.0,n_jobs=12,random_state=seed))
    else:est=DummyClassifier(strategy='most_frequent' if model=='majority' else 'stratified',random_state=seed)
    est.fit(X[fit],data['label'][fit]);probs=est.predict_proba(X[test]);classes=est.classes_
    aligned=np.zeros((len(test),nclasses),dtype=np.float32);aligned[:,classes]=probs
    return aligned,{}

def deep(data,fit,val,test,j,nclasses,fold):
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    sys.path.insert(0,str(W/'scripts/v2'))
    import screen_v2 as v2
    from determinism import configure
    seed=j['seed']+fold*10007;configure(seed)
    if not torch.cuda.is_available():raise RuntimeError('GPU worker cannot see authorized CUDA device')
    if not len(val):raise RuntimeError('No validation sessions in fold: cannot select epoch under prescribed rule')
    scaled,normalization=transform(data,fit,j['condition']!='no_norm')
    direction=int(j['model']);model,kind=v2.make_model(direction,nclasses,6);model.to('cuda')
    def tensors(idx):
        x=v2.build_inputs(scaled,direction,idx)
        return (x['seq2'],x['raw'],x['stat'],x['y']) if kind=='fusion' else (x[kind],x['y'])
    train=tensors(fit);valid=tensors(val);testing=tensors(test)
    loader=DataLoader(TensorDataset(*train),batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
    opt=torch.optim.Adam(model.parameters(),lr=1e-3);loss_fn=torch.nn.CrossEntropyLoss()
    def fwd(xs):return model(*xs) if kind=='fusion' else model(xs[0])
    def infer(tensors):
        model.eval();outputs=[]
        with torch.no_grad():
            for i in range(0,len(tensors[-1]),512):outputs.append(torch.softmax(fwd([x[i:i+512].to('cuda') for x in tensors[:-1]]),1).cpu().numpy())
        return np.concatenate(outputs)
    best=-1.;best_epoch=None;best_weights=None;history=[]
    for epoch in range(1,int(j.get('max_epochs',30))+1):
        model.train();total=0.;count=0
        for *xs,y in loader:
            xs=[x.to('cuda') for x in xs];y=y.to('cuda');opt.zero_grad();loss=loss_fn(fwd(xs),y);loss.backward();opt.step();total+=float(loss.item())*len(y);count+=len(y)
        pv=infer(valid);mf=float(f1_score(data['label'][val],pv.argmax(1),labels=np.arange(nclasses),average='macro',zero_division=0))
        history.append({'epoch':epoch,'train_loss':total/count,'validation_macro_f1':mf,'validation_accuracy':float(accuracy_score(data['label'][val],pv.argmax(1)))})
        if mf>best:best=mf;best_epoch=epoch;best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(json.dumps({'job':j['id'],'fold':fold,'epoch':epoch,'train_loss':total/count,'validation_macro_f1':mf}),flush=True)
    model.load_state_dict(best_weights);probs=infer(testing)
    return probs,{'selected_epoch':best_epoch,'epoch_cap':int(j.get('max_epochs',30)),'selected_validation_macro_f1':best,'validation_curve':history,'normalization':normalization,'model_seed':seed,'deterministic_settings':{'cudnn.deterministic':True,'cudnn.benchmark':False,'deterministic_algorithms':True,'warn_only':True,'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG']}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);args=ap.parse_args()
    plan=json.loads((W/'config/queue_plan.json').read_text());j=next(x for x in plan['jobs'] if x['id']==args.job)
    if j.get('post_stage_a_addition'):
        from protocol_control import stop_is_active,verify_frozen_rules
        verify_frozen_rules();assert not stop_is_active()
        j={**j,'execution_started':True,'implementation_status':'ready'}
    out=W/'results/v3'/j['id'];out.parent.mkdir(parents=True,exist_ok=True)
    marker=W/'markers/runs'/f'{j["id"]}.json'
    if marker.exists() and Path(str(out)+'.json').exists():print('Already completed',j['id']);return
    data,names,groups=load(j)
    if data is None:
        dump(marker,{'job':j,'status':'not_needed','reason':'No new confirmed evidence edges; suffix and evidence grouping identical'});dump(Path(str(out)+'.json'),{'job':j,'status':'not_needed'});return
    N=len(data['label']);prob=np.full((N,len(names)),np.nan,dtype=np.float32);foldnum=np.full(N,-1,dtype=np.int8);hist=[];splits={}
    for fold,fit,val,test in folds(data,groups,j):
        if j.get('reference_job'):
            with np.load(W/'results/v3'/f'{j["reference_job"]}_splits.npz',allow_pickle=False) as reference:
                for part,indices in [('fit',fit),('validation',val),('test',test)]:
                    assert np.array_equal(indices,reference[f'fold{fold}_{part}']),f'Sensitivity split mismatch: {fold}/{part}'
        assert np.all(foldnum[test]==-1)
        p,extra=(deep(data,fit,val,test,j,len(names),fold) if j['kind']=='deep' else traditional(data,fit,test,j,len(names),fold))
        prob[test]=p;foldnum[test]=fold
        hist.append({'fold':fold,'n_fit':len(fit),'n_validation':len(val),'n_test':len(test),**extra})
        splits[f'fold{fold}_fit']=fit;splits[f'fold{fold}_validation']=val;splits[f'fold{fold}_test']=test
    assert np.all(foldnum>=0) and np.isfinite(prob).all()
    assert np.allclose(prob.sum(1),1,atol=1e-5)
    pred=prob.argmax(1);result={'job':j,'status':'completed','classes':names,'n_flows':N,'metrics':metrics(data['label'],pred,names),'fold_details':hist,'metric_aggregation':'merged five-fold OOF predictions','training_cap_per_class':1000,'validation_split_before_training_cap':True,'tie_break_epoch':'first maximum validation macro-F1','grouping':'flow-stratified outer folds with session-grouped validation' if j['condition']=='leak' else ('file' if j['condition'] in ['no_session','interaction'] else j['condition'])}
    np.savez_compressed(Path(str(out)+'_splits.npz'),**splits)
    with gzip.open(Path(str(out)+'_predictions.csv.gz'),'wt',newline='') as f:
        writer=csv.writer(f);writer.writerow(['flow_id','session','file','fold','true_label','pred_label']+[f'prob_{i}' for i in range(len(names))])
        for i in range(N):writer.writerow([data['flow_id'][i],groups[i] if j['condition']=='evidence' else data['session'][i],data['file'][i],int(foldnum[i]),int(data['label'][i]),int(pred[i])]+[float(x) for x in prob[i]])
    dump(Path(str(out)+'.json'),result);dump(marker,{'job':j['id'],'status':'completed','metrics':result['metrics'],'predictions_rows':N})
    print(json.dumps({'job':j['id'],'macro_f1':result['metrics']['macro_f1'],'accuracy':result['metrics']['accuracy']}),flush=True)
if __name__=='__main__':main()
