"""R2 Stage 4: ONE-TIME scoring of locked fold models on (a) their MIT-BIH outer test fold and (b) INCART (as-given and peak-recentred).
Preconditions: lock verifies; all 15 runs (A,B,C x 5 folds) have selected.json. Refuses to run twice (results/scores/SCORED.json).
No metric is computed here; only logits are written. Metrics: 08_evaluate.py."""
import os, sys, json, time, datetime, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2common as rc
from verify_lock import verify

lock = verify()
OUT = os.path.join(rc.R2, 'results', 'scores'); os.makedirs(OUT, exist_ok=True)
flag = os.path.join(OUT, 'SCORED.json')
if os.path.exists(flag): sys.exit('Scoring already performed once (%s). Refusing to re-score.' % flag)
runs = {(m, k): os.path.join(rc.R2, 'results', 'runs', f'{m}_fold{k}') for m in 'ABC' for k in range(rc.K)}
missing = [f'{m}{k}' for (m, k), p in runs.items() if not os.path.exists(os.path.join(p, 'selected.json'))]
if missing: sys.exit(f'Not all fold models are locked/trained: missing {missing}')
d = rc.load_mit(); inc = rc.load_incart(); log = []
for (m, k), p in runs.items():
    sel = json.load(open(os.path.join(p, 'selected.json')))
    model = rc.build(m).to('cuda'); model.load_state_dict(torch.load(os.path.join(p, sel['checkpoint']))); model.eval()
    _, _, te = rc.partition(d['subject'], k)
    np.savez_compressed(os.path.join(OUT, f'mit_{m}_fold{k}.npz'), logits=rc.logits(model, d['X'][te]).numpy(), y=d['y'][te],
                        subject=d['subject'][te], record=d['record'][te], beat_index=np.where(te)[0], T=sel['temperature'])
    np.savez_compressed(os.path.join(OUT, f'incart_{m}_fold{k}.npz'), logits=rc.logits(model, inc['X']).numpy(),
                        logits_recentred=rc.logits(model, inc['X_recentred']).numpy(), T=sel['temperature'])
    log.append(dict(model=m, fold=k, checkpoint=sel['checkpoint'], checkpoint_sha256=rc.sha256(os.path.join(p, sel['checkpoint'])), T=sel['temperature'],
                    n_test=int(te.sum())))
    print('scored', m, k, flush=True)
np.savez_compressed(os.path.join(OUT, 'incart_labels.npz'), y=inc['y'], patient=inc['patient'], record=inc['record'])
json.dump(dict(scored_at=datetime.datetime.now().isoformat(timespec='seconds'), lock_sha256=lock['lock_file_sha256'], runs=log), open(flag, 'w'), indent=1)
print('done')
