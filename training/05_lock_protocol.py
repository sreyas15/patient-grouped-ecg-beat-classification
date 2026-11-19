"""R2 Stage 2b: freeze the approved protocol. Run ONCE, before any training. Refuses to overwrite an existing lock."""
import os, sys, json, shutil, datetime, platform
import numpy as np, torch, sklearn, imblearn, scipy, wfdb, pywt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2common as rc

R2 = rc.R2; LP = os.path.join(R2, 'locked_protocol.json')
if os.path.exists(LP): sys.exit('Protocol already locked: ' + LP)
# 1. freeze the approved fold assignment (content must equal the proposal that the authors approved)
prop = json.load(open(os.path.join(R2, 'results', 'splits', 'proposed_folds.json')))
locked = dict(prop, status='LOCKED (approved by authors, decision D5)', locked_at=datetime.datetime.now().isoformat(timespec='seconds'))
fl = os.path.join(R2, 'results', 'splits', 'folds_LOCKED.json'); json.dump(locked, open(fl, 'w'), indent=1)
assert json.load(open(fl))['folds'] == prop['folds']
# 2. freeze the protocol text
shutil.copy(os.path.join(R2, 'experimental_protocol_DRAFT.md'), os.path.join(R2, 'experimental_protocol_LOCKED.md'))
# 3. hash everything the results depend on
E = 'experiments/'; U = rc.ECGTF
files = ['experimental_protocol_LOCKED.md', 'results/splits/folds_LOCKED.json', 'data/mitbih_beats_r2.npz', 'data/incart_beats_r2.npz',
         E + '04_extract_beats.py', E + 'r2common.py', E + '06_train.py', E + '07_score.py', E + '08_evaluate.py', E + 'verify_lock.py',
         os.path.join(U, 'models.py'), os.path.join(U, 'configs', 'data_configs.py'), os.path.join(U, 'configs', 'hparams.py')]
frozen = {f: rc.sha256(f if os.path.isabs(f) else os.path.join(R2, f)) for f in files}
fs = rc.fold_of_subject()
lock = dict(locked_at=locked['locked_at'], status='LOCKED', protocol='experimental_protocol_LOCKED.md',
            primary_model='B', comparators=['A', 'C'], prespecified_comparisons=['B-A', 'B-C'], exploratory=['C-A', 'any ranking of models'],
            internal_evaluation='MIT-BIH, nested subject-grouped 5-fold CV, pooled out-of-fold predictions (internal; records previously exposed in earlier analyses)',
            external_evaluation='INCART, ensemble of the five locked fold models, primary macro-F1 over N,S,V,F; as-given annotations primary, peak-recentred sensitivity',
            config=rc.CONFIG, folds={k: v for k, v in locked['folds'].items()}, n_subjects=len(fs),
            fold_seeds={str(k): rc.CONFIG['fold_seed_base'] + k for k in range(rc.K)}, bootstrap=dict(resamples=2000, seed=20261102, unit='subject/patient'),
            decisions=['D1 identical 186-sample input', 'D2 SMOTE 20,000/class (compute; not optimised)', 'D3 MLII where available, V5 for 102/104',
                       'D4 INCART = ensemble of 5 fold models', 'D5 folds frozen, never redrawn', 'D6 INCART external, macro-F1 N,S,V,F', 'D7 as-given primary, recentred sensitivity'],
            environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                             numpy=np.__version__, sklearn=sklearn.__version__, imblearn=imblearn.__version__, scipy=scipy.__version__, wfdb=wfdb.__version__, pywt=pywt.__version__),
            frozen_files=frozen)
json.dump(lock, open(LP, 'w'), indent=1)
open(os.path.join(R2, 'locked_protocol.sha256'), 'w').write(rc.sha256(LP) + '  locked_protocol.json\n')
print(json.dumps({k: v for k, v in lock.items() if k not in ('frozen_files', 'config')}, indent=1)); print('frozen files:', len(frozen)); print('lock sha256:', rc.sha256(LP))
