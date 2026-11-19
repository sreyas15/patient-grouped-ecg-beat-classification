"""R2 per-fold DESCRIPTIVE diagnostics. Approved addendum, written and hashed BEFORE any test scoring. NOT part of the frozen primary evaluation.

STRICTLY DESCRIPTIVE. It computes counts and ratios only. It performs no hypothesis test, no model/checkpoint/threshold/calibration/hyperparameter choice,
changes no protocol or primary-evaluation definition, and writes nothing that 07_score.py / 08_evaluate.py read.
Inputs (read-only): the 15 MIT-BIH test-logit files results/scores/mit_{A,B,C}_fold{0-4}.npz (logits, labels, subjects, T) and the 15 VALIDATION logit
files results/runs/*/val_logits.npz (for the validation-vs-test contrast). It never opens an INCART file.
Per fold and model: 5x5 confusion counts and row-normalised matrix; per-class precision, recall, F1, support; actual vs predicted class distributions;
per-subject actual/predicted/true-positive counts; and a focus report on the F class of B fold 3 (and A, C on the same fold for context).
Predictions = argmax of temperature-scaled softmax (the argmax equals the argmax of the raw logits; T does not change it).

REFUSES TO RUN unless ALL of:  --approved is given;  the protocol lock verifies;  results/scores/SCORED.json exists (test logits were produced);
results/evaluation/mit_internal_summary.json exists (the primary evaluation was run first, so this cannot influence it).
Outputs: a NEW directory (exclusive-create, nothing is overwritten). Usage:  python -P -B experiments/10_per_fold_diagnostics.py --approved
Optional path overrides exist only so the script can be tested on synthetic data: --scores-dir --eval-dir --runs-dir --out-dir."""
import os, sys, json, csv, hashlib, argparse, datetime
import numpy as np
from scipy.special import softmax

HERE = os.path.dirname(os.path.abspath(__file__)); R2 = os.path.dirname(HERE); sys.path.insert(0, HERE)
CLS = ['N', 'S', 'V', 'F', 'Q']; K = 5; MODELS = 'ABC'
ap = argparse.ArgumentParser()
ap.add_argument('--approved', action='store_true', help='explicit user approval for running the descriptive diagnostics')
ap.add_argument('--scores-dir', default=os.path.join(R2, 'results', 'scores')); ap.add_argument('--eval-dir', default=os.path.join(R2, 'results', 'evaluation'))
ap.add_argument('--runs-dir', default=os.path.join(R2, 'results', 'runs')); ap.add_argument('--out-dir', default=os.path.join(R2, 'results', 'evaluation_supplementary', 'per_fold_diagnostics'))
a = ap.parse_args()
if not a.approved: sys.exit('Refusing to run: --approved not given (explicit user approval is required, after scoring and the primary evaluation).')
import a1common as a1                       # addendum A1: scoped lock check (does not hash the external-dataset file) + access guard
a1.install_no_external_guard()              # this descriptive stage must never open or list any external-dataset path
lock = a1.verify_lock(include_external=False)
if not os.path.exists(os.path.join(a.scores_dir, 'SCORED.json')): sys.exit('Refusing to run: SCORED.json not found (no test scoring has been performed).')
if not os.path.exists(os.path.join(a.eval_dir, 'mit_internal_summary.json')): sys.exit('Refusing to run: primary evaluation output (mit_internal_summary.json) not found. Run the primary evaluation first.')
if os.path.exists(a.out_dir): sys.exit(f'Refusing to run: output directory already exists ({a.out_dir}); nothing is overwritten.')

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def ratio(num, den): return None if den == 0 else float(num) / float(den)
def cm_of(y, pred): return np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
def per_class(cm):
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    return [dict(cls=CLS[c], support=int(cm[c].sum()), predicted=int(cm[:, c].sum()), tp=int(tp[c]), fp=int(fp[c]), fn=int(fn[c]),
                 precision=ratio(tp[c], tp[c] + fp[c]), recall=ratio(tp[c], tp[c] + fn[c]), f1=ratio(2 * tp[c], 2 * tp[c] + fp[c] + fn[c]),
                 predicted_over_actual=ratio(cm[:, c].sum(), cm[c].sum())) for c in range(K)]
def dist(cm):
    act, prd, n = cm.sum(1), cm.sum(0), cm.sum()
    return dict(actual_counts=[int(v) for v in act], predicted_counts=[int(v) for v in prd], actual_fraction=[ratio(v, n) for v in act], predicted_fraction=[ratio(v, n) for v in prd])

os.makedirs(a.out_dir)   # exclusive: fails if it exists
rec = []; conf_rows = []; cls_rows = []; subj_rows = []; inputs = {}; fold_data = {}
for m in MODELS:
    for k in range(K):
        p = os.path.join(a.scores_dir, f'mit_{m}_fold{k}.npz'); inputs[os.path.basename(p)] = sha(p); q = np.load(p)
        y = q['y'].astype(np.int64); pred = softmax(q['logits'] / float(q['T']), 1).argmax(1); subj = q['subject']; cm = cm_of(y, pred); pc = per_class(cm)
        present = [c for c in range(K) if cm[c].sum() > 0]; f1p = [pc[c]['f1'] for c in present if pc[c]['f1'] is not None]
        rowsum = cm.sum(1, keepdims=True)
        r = dict(model=m, fold=k, n=int(len(y)), accuracy=float(np.trace(cm) / cm.sum()), temperature=float(q['T']), classes_with_support=[CLS[c] for c in present],
                 macro_f1_over_classes_with_support=(float(np.mean(f1p)) if f1p else None), confusion_counts=cm.tolist(),
                 confusion_rownorm=[[ratio(cm[i, j], rowsum[i, 0]) for j in range(K)] for i in range(K)], per_class=pc, distribution=dist(cm))
        rec.append(r)
        for i in range(K):
            for j in range(K): conf_rows.append(dict(model=m, fold=k, true=CLS[i], pred=CLS[j], count=int(cm[i, j])))
        for row in pc: cls_rows.append(dict(model=m, fold=k, **row))
        for s in sorted(set(subj.tolist())):
            sel = subj == s
            for c in range(K):
                subj_rows.append(dict(model=m, fold=k, subject=s, cls=CLS[c], n_beats_subject=int(sel.sum()), actual=int((y[sel] == c).sum()), predicted=int((pred[sel] == c).sum()), true_positive=int(((y[sel] == c) & (pred[sel] == c)).sum())))
        fold_data[(m, k)] = (y, pred, subj, cm)

# validation-vs-test contrast (VALIDATION logits only), all 15 runs
val = []
for m in MODELS:
    for k in range(K):
        z = np.load(os.path.join(a.runs_dir, f'{m}_fold{k}', 'val_logits.npz')); yv = z['y'].astype(np.int64); pv = z['logits'].argmax(1)
        val.append(dict(model=m, fold=k, validation_actual_counts=np.bincount(yv, minlength=K).tolist(), validation_predicted_counts=np.bincount(pv, minlength=K).tolist()))
vmap = {(v['model'], v['fold']): v for v in val}

# focus: F class, fold 3 (B primary; A, C for context) -- descriptive only
focus = {}
for m in MODELS:
    y, pred, subj, cm = fold_data[(m, 3)]; F = CLS.index('F'); fpmask = (pred == F) & (y != F)
    by_sub = []
    for s in sorted(set(subj.tolist())):
        sel = subj == s; by_sub.append(dict(subject=s, n_beats=int(sel.sum()), actual_F=int(((y == F) & sel).sum()), predicted_F=int(((pred == F) & sel).sum()), false_positive_F=int((fpmask & sel).sum())))
    by_sub.sort(key=lambda d: -d['false_positive_F'])
    fp_total = int(fpmask.sum())
    focus[m] = dict(test_fold=3, n=int(len(y)), actual_F=int(cm[F].sum()), predicted_F=int(cm[:, F].sum()), true_positive_F=int(cm[F, F]), false_positive_F=fp_total,
                    precision_F=ratio(cm[F, F], cm[:, F].sum()), recall_F=ratio(cm[F, F], cm[F].sum()), f1_F=ratio(2 * cm[F, F], cm[F].sum() + cm[:, F].sum()),
                    predicted_F_over_actual_F=ratio(cm[:, F].sum(), cm[F].sum()), false_positive_F_by_true_class={CLS[c]: int(cm[c, F]) for c in range(K) if c != F},
                    top_subjects_by_false_positive_F=[dict(d, share_of_false_positive_F=ratio(d['false_positive_F'], fp_total)) for d in by_sub[:10]],
                    validation_fold4_actual_F=vmap[(m, 3)]['validation_actual_counts'][F], validation_fold4_predicted_F=vmap[(m, 3)]['validation_predicted_counts'][F],
                    note='descriptive only; validation numbers are those of the model for outer fold 3 on its validation partition (fold 4 subjects)')

def w_csv(name, rows):
    with open(os.path.join(a.out_dir, name), 'x', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
def w_json(name, obj):
    with open(os.path.join(a.out_dir, name), 'x') as f: json.dump(obj, f, indent=1)
w_json('per_fold_diagnostics.json', dict(descriptive_only=True, records=rec, validation_contrast=val))
w_csv('per_fold_confusion_counts.csv', conf_rows); w_csv('per_fold_per_class.csv', cls_rows); w_csv('per_subject_class_counts.csv', subj_rows)
w_json('F_class_focus_fold3.json', focus)
w_json('DIAGNOSTICS_META.json', dict(created=datetime.datetime.now().isoformat(timespec='seconds'), script_sha256=sha(os.path.abspath(__file__)), lock_sha256=lock['lock_file_sha256'],
                                      input_test_logit_files_sha256=inputs, scores_flag_sha256=sha(os.path.join(a.scores_dir, 'SCORED.json')), primary_evaluation_summary_sha256=sha(os.path.join(a.eval_dir, 'mit_internal_summary.json')),
                                      statement='descriptive only; no test, threshold, selection, calibration or protocol change; INCART files never opened'))
print('descriptive diagnostics written to', a.out_dir)
