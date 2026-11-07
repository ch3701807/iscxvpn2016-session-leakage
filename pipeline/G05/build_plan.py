"""Register the four authorized priorities and matched-budget historical references."""
import importlib.metadata
import os
import re
import subprocess
import sys
from common import W,dump,now,read,sha

def main():
    assert (W/'markers/PRECHECK_PASSED.json').exists()
    assert not (W/'markers/PREPARED.json').exists()
    checks={(r['condition'],r['seed']):r for r in read(W/'split_check.json')['checks']}
    jobs=[];refs=[];paths={x:set() for x in ['G01','G02','G03','G04']}
    def add(priority,condition,models,budget='baseline'):
        for seed in [1,7,42]:
            for model in models:
                kind='deep' if model.isdigit() else 'baseline'
                b=budget if kind=='deep' else 'baseline'
                suffix='_es' if b=='es' else ('_100epoch' if b=='100epoch' else '')
                jid=f'G05_{condition}_multi_{model}_{seed}'+suffix
                c=checks[condition,seed]
                job={'id':jid,'priority':priority,'condition':condition,'task':'multi','model':model,'seed':seed,'kind':kind,'budget':b,
                    'max_epochs':300 if b=='es' else (100 if b=='100epoch' else (30 if kind=='deep' else None)),
                    'patience':30 if b=='es' else None,'split_file':c['split_file'],'split_sha256':c['split_sha256'],'split_array_digest':c['digest_actual']}
                # Use G04 only if it contains the same model AND the same budget.
                repair=(condition=='evidence' and seed==1) or (condition=='E2' and seed==1) or (condition=='no_clean' and seed==1) or (condition in ['Pstar_8','E3_8'] and seed==7)
                rpath=None;task=None
                if repair:
                    name=f'B_baseline_{condition}_multi_{model}_{seed}' if kind=='baseline' else f'G04_{condition}_multi_{model}_{seed}'
                    candidate=W.parent/'G04/results/v3'/(name+'.json')
                    if candidate.exists():
                        old=read(candidate);oj=old['job']
                        if kind=='baseline' or (oj['max_epochs']==job['max_epochs'] and oj['patience']==job['patience']):task='G04';rpath=candidate
                if rpath is None:
                    prefix='A' if condition=='Pstar' else 'B'
                    if kind=='deep' and b=='es':
                        task='G03' if condition=='E2' else 'G02'
                        rpath=W.parent/task/('results/v3' if task=='G03' else 'results')/f'{task}_{condition}_multi_{model}_{seed}.json'
                    elif condition in ['E2','Pstar_8','E3_8']:
                        task='G03';rpath=W.parent/task/'results/v3'/f'B_baseline_{condition}_multi_{model}_{seed}.json'
                    else:
                        task='G01'
                        name=(f'{prefix}_baseline_{condition}_multi_{model}_{seed}' if kind=='baseline' else
                              (f'Bplus1_deep_Pstar_100epoch_multi_{model}_{seed}' if b=='100epoch' else f'{prefix}_deep_{condition}_multi_{model}_{seed}'))
                        rpath=W.parent/task/'results/v3'/(name+'.json')
                assert rpath.exists(), 'Missing matched reference '+jid
                old=read(rpath);assert old['status']=='completed'
                job.update(reference_task=task,reference_file=rpath.relative_to(W.parent/task).as_posix(),reference_sha256=sha(rpath))
                refs.append({'job':jid,'task':task,'path':job['reference_file'],'sha256':sha(rpath),'metrics':old['metrics'],'budget':b})
                paths[task].add(job['reference_file']);jobs.append(job)
    add(1,'Pstar',['RF_seq_iat','8','11'],'es')
    add(1,'no_session',['RF_seq_iat','8','11'],'es')
    add(1,'evidence',['RF_seq_iat','8','11'],'es')
    add(2,'Pstar',['majority','stratified_random','LR_stat','LR_seq_iat','RF_stat'])
    for budget in ['30epoch','100epoch']:add(2,'Pstar',['1','2','3','20','7','8','11'],budget)
    for condition in ['no_clean','interaction']:add(3,condition,['RF_seq_iat','11','7'],'30epoch')
    add(3,'no_sni',['11','7'],'30epoch');add(3,'no_norm',['11'],'30epoch')
    add(4,'E2',['RF_seq_iat','8','11'],'es')
    for c in ['Pstar_8','E3_8']:add(4,c,['RF_seq_iat'])
    assert len(jobs)==126 and len({j['id'] for j in jobs})==126
    assert [sum(j['priority']==p for j in jobs) for p in range(1,5)]==[27,57,27,15]
    # Original leak is untouched; include only existing scores/provenance for the ladder.
    leak=[]
    for seed in [1,7,42]:
        for model in ['RF_seq_iat','8','11']:
            task='G01' if model=='RF_seq_iat' else 'G02'
            p=W.parent/task/('results/v3' if task=='G01' else 'results')/(f'B_baseline_leak_multi_{model}_{seed}.json' if task=='G01' else f'G02_leak_multi_{model}_{seed}.json')
            old=read(p);assert old['status']=='completed';rel=p.relative_to(W.parent/task).as_posix();paths[task].add(rel)
            leak.append({'condition':'leak','model':model,'seed':seed,'task':task,'path':rel,'sha256':sha(p),'metrics':old['metrics'],'not_rerun':True})
    dump(W/'config/queue_plan.json',{'task':'G05','strict_priority_barriers':True,'jobs':jobs})
    dump(W/'audit/reference_results.json',refs);dump(W/'audit/original_leak_reference.json',leak)
    dump(W/'audit/reference_source_paths.json',{task:sorted(p) for task,p in paths.items()})
    import torch
    versions={p:importlib.metadata.version(p) for p in ['torch','numpy','scikit-learn','scipy']}
    versions.update(python=sys.version.split()[0],torch_cuda_build=torch.version.cuda,environment_reused_readonly='G01/env',PYTHONNOUSERSITE=os.environ.get('PYTHONNOUSERSITE'),PYTHONDONTWRITEBYTECODE=os.environ.get('PYTHONDONTWRITEBYTECODE'))
    assert versions['scikit-learn']=='1.1.3'
    dump(W/'audit/environment.json',versions)
    freeze=subprocess.check_output([sys.executable,'-B','-m','pip','freeze'],text=True,stderr=subprocess.DEVNULL)
    freeze=freeze.replace(str(W.parent.parent),'${WORKSPACE_ROOT}').replace(str(W.parents[3]),'${OWNER_HOME}')
    freeze=re.sub(r'/home/[^/\s:]+','${HOME}',freeze)
    freeze=re.sub(r'(?i)(https?://)[^\s/@]+:[^\s/@]+@',r'\1[REDACTED]@',freeze)
    (W/'audit/pip_freeze.txt').write_text(freeze)
    dump(W/'config/protocol.json',{'task':'G05','post_G01_G04_and_second_external_review':True,'rules_fixed_before_training':True,'rules_sha256':sha(W/'RULES_G05.md'),
        'splitter':'user-provided FixedSGKF','outer_random_state':'seed','inner_rng':'seed+fold*10007','model_seed':'seed+fold*10007',
        'sklearn_version':'1.1.3','split_checks_sha256':sha(W/'split_check.json'),'n_splits':5,'seeds':[1,7,42],'cap_per_class':1000,
        'RF_LR_Dummy':'unmodified G01 traditional function','fixed_deep':'unmodified G01 deep function','early_deep':'unmodified G02 deep function',
        'priorities':[{'priority':p,'runs':sum(j['priority']==p for j in jobs)} for p in range(1,5)],
        'barrier':'finish models, exposure if P1, verify/report/package, Azure hash acceptance, then next priority','allowed_GPU':[2,3]})
    print('Registered 126 runs: priorities 27/57/27/15, all references matched')

if __name__=='__main__':main()
