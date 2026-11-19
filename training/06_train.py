"""R2 Stage 3: train one model on one outer fold under the LOCKED protocol.
Uses ONLY the training partition (SMOTE) and the validation partition (lr, checkpoint, temperature). Test-partition beats are never
indexed here: the test mask is computed only to assert disjointness and is not used to slice data.
Usage: python 06_train.py <A|B|C> <fold 0-4>
Outputs: results/runs/<M>_fold<k>/ (config, per-lr history + checkpoint, selected.json, val logits). Refuses to overwrite a completed run."""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2common as rc
from verify_lock import verify

m, k = sys.argv[1], int(sys.argv[2]); assert m in 'ABC' and 0 <= k < rc.K
lock = verify()   # aborts if any locked file changed
out = os.path.join(rc.R2, 'results', 'runs', f'{m}_fold{k}'); os.makedirs(out, exist_ok=True)
if os.path.exists(os.path.join(out, 'selected.json')): print('already complete:', out); sys.exit(0)

d = rc.load_mit(); tr, va, te = rc.partition(d['subject'], k)
seed = rc.CONFIG['fold_seed_base'] + k
Xs, ys = rc.smote_balance(d['X'][tr], d['y'][tr], seed, rc.CONFIG['smote_target'], rc.CONFIG['smote_k'])
Xva, yva = d['X'][va], d['y'][va]
info = dict(model=m, fold=k, seed=seed, lock_sha256=lock['lock_file_sha256'], config=rc.CONFIG, n_params=rc.nparams(rc.build(m)),
            train_subjects=sorted(set(d['subject'][tr])), val_subjects=sorted(set(d['subject'][va])),
            n_train_beats_real=int(tr.sum()), train_real_class_counts=np.bincount(d['y'][tr], minlength=5).tolist(),
            train_after_smote_class_counts=np.bincount(ys, minlength=5).tolist(), n_val=int(va.sum()), val_class_counts=np.bincount(yva, minlength=5).tolist(),
            smote_applied_to='training partition only', test_partition_used='no')
json.dump(info, open(os.path.join(out, 'config.json'), 'w'), indent=1)
results = []
for lr in rc.CONFIG['lr_grid']:
    t0 = time.time()
    (f, ep, state), hist, vc = rc.train_one(m, Xs, ys, Xva, yva, lr, rc.CONFIG['epochs'][m], seed)
    torch.save(state, os.path.join(out, f'ckpt_lr{lr:g}.pt'))
    json.dump(hist, open(os.path.join(out, f'history_lr{lr:g}.json'), 'w'), indent=1)
    results.append(dict(lr=lr, best_val_macro_f1=f, best_epoch=ep, val_classes_used=[rc.CLS[c] for c in vc], sec=round(time.time() - t0, 1)))
    print(m, k, results[-1], flush=True)
best = max(results, key=lambda r: r['best_val_macro_f1'])   # ties -> first (smallest lr) in grid order
model = rc.build(m).to('cuda'); model.load_state_dict(torch.load(os.path.join(out, f"ckpt_lr{best['lr']:g}.pt"))); lv = rc.logits(model, Xva)
T = rc.fit_T(lv, yva)
np.savez_compressed(os.path.join(out, 'val_logits.npz'), logits=lv.numpy(), y=yva, subject=d['subject'][va], record=d['record'][va])
sel = dict(model=m, fold=k, selected_lr=best['lr'], selected_epoch=best['best_epoch'], val_macro_f1=best['best_val_macro_f1'],
           val_classes_used=best['val_classes_used'], temperature=T, checkpoint=f"ckpt_lr{best['lr']:g}.pt", grid=results)
json.dump(sel, open(os.path.join(out, 'selected.json'), 'w'), indent=1)
print('SELECTED', json.dumps({k_: v for k_, v in sel.items() if k_ != 'grid'}), flush=True)
