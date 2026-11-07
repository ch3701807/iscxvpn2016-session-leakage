"""Exact G01 function source, extracted without arithmetic changes."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,precision_recall_fscore_support

def metrics(y,p,names):
    labels=np.arange(len(names));p0,r0,f0,s0=precision_recall_fscore_support(y,p,labels=labels,zero_division=0)
    return {'macro_f1':float(f1_score(y,p,labels=labels,average='macro',zero_division=0)),'accuracy':float(accuracy_score(y,p)),'confusion_matrix':confusion_matrix(y,p,labels=labels).tolist(),'per_class':[{'class':name,'precision':float(p0[i]),'recall':float(r0[i]),'f1':float(f0[i]),'support':int(s0[i])} for i,name in enumerate(names)]}

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
