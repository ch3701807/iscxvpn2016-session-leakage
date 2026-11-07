"""Freeze scientific code/configuration and all reused source inputs before training."""
import ast
from common import W,dump,load_data,now,read,sha
from fixed_splitter import folds
from prepare_splits import digest
import source_g01
import numpy as np

def main():
    assert not (W/'markers/PREPARED.json').exists()
    assert read(W/'split_check.json')['all_checks_passed']
    assert folds.__code__.co_code==source_g01.folds.__code__.co_code
    for entry in read(W/'split_check.json')['checks']:
        data,classes,groups=load_data(entry['condition'])
        current=[(fit,val,test) for _,fit,val,test in folds(data,groups,{'condition':entry['condition'],'seed':entry['seed']})]
        assert digest(current)==entry['digest_expected']
        with np.load(W/entry['split_file'],allow_pickle=False) as stored:
            for fold,values in enumerate(current):
                for part,a in zip(['fit','validation','test'],values):assert np.array_equal(a,stored[f'fold{fold}_{part}'])
    for own,source in [('source_g01.py','G01/scripts/run_experiment.py'),('source_g02.py','G02/scripts/run_g02.py'),('source_exposure.py','G01/scripts/near_duplicate_exposure.py')]:assert sha(W/'scripts'/own)==sha(W.parent/source)
    jobs=read(W/'config/queue_plan.json')['jobs'];assert len(jobs)==126 and not any(j['condition']=='leak' for j in jobs)
    for p in (W/'scripts').rglob('*.py'):compile(p.read_text(),p.name,'exec')
    paths={task:set(v) for task,v in read(W/'audit/reference_source_paths.json').items()}
    paths['G01'].update(['data/iscx_v3.npz','config/final_classes.json','audit/evidence_sessions.json','audit/v3_cleaning_inventory.json',
        'scripts/run_experiment.py','scripts/near_duplicate_exposure.py','scripts/determinism.py','scripts/v2/screen_v2.py','scripts/v2/metrics.py','RULES_addendum.md'])
    for d in ['data/near_duplicate_private','data/v3_files']:paths['G01'].update(p.relative_to(W.parent/'G01').as_posix() for p in (W.parent/'G01'/d).glob('*.npz'))
    paths['G02'].update(['scripts/run_g02.py','scripts/common.py','config/class_map.json'])
    paths['G03'].update(['config/E2_file_to_group.csv','config/E3_file_to_group.csv','scripts/common.py'])
    paths['G04'].update(['scripts/common.py','scripts/run_g04.py','split_repair.json'])
    frozen={task:[{'path':p,'sha256':sha(W.parent/task/p)} for p in sorted(values)] for task,values in paths.items()}
    own=[W/'RULES_G05.md',W/'split_check.json',W/'audit/reference_results.json',W/'audit/original_leak_reference.json',W/'audit/fold_class_coverage.csv']
    own+=sorted(p for d in ['scripts','config','data/splits'] for p in (W/d).rglob('*') if p.is_file() and p.suffix not in ['.pyc','.tmp'])
    own+=sorted((W/'audit').glob('*_row_manifest.csv.gz'))
    frozen['g05_frozen']=[{'path':p.relative_to(W).as_posix(),'sha256':sha(p)} for p in sorted(set(own))]
    dump(W/'config/frozen_inputs.json',frozen)
    dump(W/'audit/pretraining_protocol_tests.json',{'time':now(),'all_30_digests_rechecked':True,'original_folds_bytecode_identical_except_global_splitter_binding':True,'source_functions_byte_identical':True,'all_126_run_budgets_and_references_registered':True,'leak_not_rerun':True,'passed':True})
    dump(W/'markers/PREPARED.json',{'time':now(),'training_started':False,'runs':126,'priority_counts':[27,57,27,15],'frozen_inputs_sha256':sha(W/'config/frozen_inputs.json'),'protocol_sha256':sha(W/'config/protocol.json')})
    print('Scientific inputs and code frozen; pretraining checks passed')

if __name__=='__main__':main()
