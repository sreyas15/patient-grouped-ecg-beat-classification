"""R2 POST-HOC DESCRIPTIVE analysis (requested by the authors after fold 0; NOT part of the locked protocol, NOT a new endpoint).
Question: are low macro-F1 values driven disproportionately by the rare S, F and Q classes?
Reads only the one-time scores (results/scores) and the locked evaluation outputs; changes no model, threshold or class set.
Outputs: results/evaluation/posthoc_*.csv|json. All class-subset numbers are a DECOMPOSITION of the primary metric, not alternative results.
"""
import os, sys, json, numpy as np, pandas as pd
from scipy.special import softmax
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2common as rc
SC = os.path.join(rc.R2, 'results', 'scores'); OUT = os.path.join(rc.R2, 'results', 'evaluation')
assert os.path.exists(os.path.join(OUT, 'mit_internal_summary.json')), 'run the locked 08_evaluate.py first'

def f1s(y, pred, K=5):
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K); tp = np.diag(cm).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return cm, 2 * tp / (cm.sum(0) + cm.sum(1)), tp / cm.sum(1), tp / cm.sum(0)
def load_mit(m):
    parts = [np.load(os.path.join(SC, f'mit_{m}_fold{k}.npz')) for k in range(rc.K)]
    o = np.argsort(np.concatenate([q['beat_index'] for q in parts]))
    cat = lambda key: np.concatenate([q[key] for q in parts])[o]
    T = np.concatenate([np.full(len(q['y']), float(q['T'])) for q in parts])[o]
    return softmax(cat('logits') / T[:, None], 1), cat('y'), cat('subject')
def load_inc(m):
    lab = np.load(os.path.join(SC, 'incart_labels.npz'))
    p = np.mean([softmax((q := np.load(os.path.join(SC, f'incart_{m}_fold{k}.npz')))['logits'] / float(q['T']), 1) for k in range(rc.K)], 0)
    return p, lab['y'], lab['patient'].astype(str)

res = {}
for dset, loader, primary in (('mit_internal', load_mit, [0, 1, 2, 3, 4]), ('incart_external', load_inc, [0, 1, 2, 3])):
    rows_dec, rows_subj, rows_flow = [], [], []
    for m in 'ABC':
        p, y, g = loader(m); pred = p.argmax(1); cm, f1, rec, prec = f1s(y, pred)
        prim = float(np.mean(np.nan_to_num(f1[primary])))
        # 1. decomposition: each class's contribution to the primary macro-F1 and its shortfall from 1
        for c in primary:
            rows_dec.append(dict(model=m, cls=rc.CLS[c], f1=f1[c], recall=rec[c], precision=prec[c], contribution_to_macro=f1[c] / len(primary),
                                 shortfall_share=(1 - f1[c]) / max(1e-12, sum(1 - np.nan_to_num(f1[j]) for j in primary)),
                                 support_beats=int((y == c).sum()), support_subjects=int(len(np.unique(g[y == c]))),
                                 top_subject_share=float(pd.Series(g[y == c]).value_counts(normalize=True).iloc[0]) if (y == c).any() else np.nan,
                                 primary_macro_f1=prim))
        # 2. per-subject recall heterogeneity (subjects with >= 10 beats of the class)
        for c in range(5):
            for s in np.unique(g[y == c]):
                msk = (g == s) & (y == c)
                if msk.sum() >= 10: rows_subj.append(dict(model=m, cls=rc.CLS[c], subject=s, n=int(msk.sum()), recall=float((pred[msk] == c).mean())))
        # 3. error flows: where true-class beats go; and what share of each predicted class is actually N (prevalence effect on precision)
        for c in range(5):
            if (y == c).any():
                rows_flow.append(dict(model=m, true_cls=rc.CLS[c], **{f'to_{rc.CLS[j]}': float(cm[c, j] / cm[c].sum()) for j in range(5)}))
        for j in range(5):
            if cm[:, j].sum(): rows_flow.append(dict(model=m, predicted_cls=rc.CLS[j], n_predicted=int(cm[:, j].sum()), share_true_N=float(cm[0, j] / cm[:, j].sum())))
        # 4. descriptive subsets (decomposition only; NOT alternative endpoints)
        res.setdefault(dset, {})[m] = dict(primary_macro_f1=prim,
            decomposition_macro_f1_over={'N,V': float(np.nanmean(f1[[0, 2]])), 'N,S,V': float(np.nanmean(f1[[0, 1, 2]])),
                                         'N,S,V,F': float(np.nanmean(f1[[0, 1, 2, 3]])), 'all5': float(np.nanmean(np.nan_to_num(f1)))},
            N_beats_misclassified_as_nonN=int(cm[0, 1:].sum()), N_beats_total=int(cm[0].sum()))
    pd.DataFrame(rows_dec).to_csv(os.path.join(OUT, f'posthoc_{dset}_class_decomposition.csv'), index=False)
    sd = pd.DataFrame(rows_subj); sd.to_csv(os.path.join(OUT, f'posthoc_{dset}_per_subject_recall.csv'), index=False)
    pd.DataFrame(rows_flow).to_csv(os.path.join(OUT, f'posthoc_{dset}_error_flows.csv'), index=False)
    if len(sd): sd.groupby(['model', 'cls'])['recall'].describe()[['count', 'min', '25%', '50%', '75%', 'max']].to_csv(os.path.join(OUT, f'posthoc_{dset}_per_subject_recall_summary.csv'))
json.dump(dict(note='POST-HOC DESCRIPTIVE; decomposition of the locked primary metric; not part of the pre-specified protocol', results=res),
          open(os.path.join(OUT, 'posthoc_class_driver_summary.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
