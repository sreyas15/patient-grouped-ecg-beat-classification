"""R2 Stage 5: pre-specified evaluation of the one-time scores (frozen with the protocol).

MIT-BIH INTERNAL (patient-grouped nested CV): pooled out-of-fold predictions, each fold's probabilities scaled by its own validation T.
  Primary model: B (pre-specified). A and C are comparators. Pre-specified comparisons: B vs A, B vs C. A vs C and any ranking = exploratory.
EXTERNAL (INCART): ensemble = mean of the five locked fold models' temperature-scaled softmax. Primary metric: macro-F1 over N,S,V,F
  (a Q prediction is an error; Q itself not evaluable: 6 beats). Sensitivity analysis: peak-recentred windows.
CIs: 95% percentile, cluster bootstrap over subjects (MIT-BIH 47) / patients (INCART 32), B=2000, seed 20261102; paired resamples across models.
"""
import os, sys, json, numpy as np, pandas as pd
from scipy.special import softmax
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2common as rc

SC = os.path.join(rc.R2, 'results', 'scores'); OUT = os.path.join(rc.R2, 'results', 'evaluation'); os.makedirs(OUT, exist_ok=True)
NB, BSEED, NBIN = 2000, 20261102, 15

# ---------------- metric definitions ----------------
def cm_of(y, pred, K=5): return np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
def from_cm(cm, cls):
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp; tn = cm.sum() - tp - fp - fn
    with np.errstate(divide='ignore', invalid='ignore'):
        sens = tp / (tp + fn); spec = tn / (tn + fp); ppv = tp / (tp + fp); f1 = 2 * tp / (2 * tp + fp + fn)
    c = list(cls)
    return dict(accuracy=float(tp.sum() / cm.sum()), macro_f1=float(np.nanmean(np.nan_to_num(f1[c]))), balanced_accuracy=float(np.nanmean(sens[c])),
                sens=sens, spec=spec, ppv=ppv, f1=f1)
def calib(p, y):
    """top-label ECE (15 equal-width bins, |acc-conf| weighted by bin share), signed gap (mean conf - acc), MCE (max bin |gap|),
    class-wise ECE (mean over classes of one-vs-rest ECE of p[:,c] vs 1[y=c]), multiclass Brier (sum over classes), NLL."""
    conf = p.max(1); hit = (p.argmax(1) == y).astype(float); b = np.minimum((conf * NBIN).astype(int), NBIN - 1)
    ece = 0.0; mce = 0.0; rows = []
    for k in range(NBIN):
        s = b == k
        if s.any():
            gap = conf[s].mean() - hit[s].mean(); ece += s.mean() * abs(gap); mce = max(mce, abs(gap))
            rows.append(dict(bin=k, lo=k / NBIN, hi=(k + 1) / NBIN, n=int(s.sum()), mean_conf=float(conf[s].mean()), accuracy=float(hit[s].mean()), signed_gap=float(gap)))
    oh = np.eye(p.shape[1])[y]; cw = []
    for c in range(p.shape[1]):
        bc = np.minimum((p[:, c] * NBIN).astype(int), NBIN - 1); e = 0.0
        for k in range(NBIN):
            s = bc == k
            if s.any(): e += s.mean() * abs(p[s, c].mean() - oh[s, c].mean())
        cw.append(e)
    return dict(ece_top=float(ece), signed_gap=float(conf.mean() - hit.mean()), mce=float(mce), ece_classwise=float(np.mean(cw)),
                ece_classwise_per_class=[float(v) for v in cw], brier=float(((p - oh) ** 2).sum(1).mean()),
                nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())), rows
def auc_pr(p, y, c):
    t = (y == c).astype(int)
    if t.min() == t.max(): return np.nan, np.nan
    return roc_auc_score(t, p[:, c]), average_precision_score(t, p[:, c])

def evaluate(name, P, y, grp, cls, Pextra=None):
    """P: dict model -> probs (TS); grp: cluster id per beat; cls: classes in the primary macro metrics."""
    pred = {m: p.argmax(1) for m, p in P.items()}; groups = np.unique(grp); gi = {g: np.where(grp == g)[0] for g in groups}
    gcm = {m: {g: cm_of(y[ix], pred[m][ix]) for g, ix in gi.items()} for m in P}
    rows, pcrows, cms = [], [], {}
    point = {}
    for m, p in P.items():
        cm = sum(gcm[m].values()); cms[m] = cm; r = from_cm(cm, cls); cal, rel = calib(p, y)
        pd.DataFrame(rel).to_csv(os.path.join(OUT, f'{name}_{m}_reliability_bins.csv'), index=False)
        au = [auc_pr(p, y, c) for c in range(5)]
        point[m] = dict(accuracy=r['accuracy'], macro_f1=r['macro_f1'], balanced_accuracy=r['balanced_accuracy'],
                        macro_auroc=float(np.nanmean([au[c][0] for c in cls])), macro_auprc=float(np.nanmean([au[c][1] for c in cls])), **cal)
        if Pextra is not None and m in Pextra: point[m].update({f'raw_{k_}': v for k_, v in calib(Pextra[m], y)[0].items() if k_ != 'ece_classwise_per_class'})
        for c in range(5):
            pcrows.append(dict(model=m, cls=rc.CLS[c], support_beats=int((y == c).sum()), support_subjects=int(len(np.unique(grp[y == c]))),
                               sens=r['sens'][c], spec=r['spec'][c], ppv=r['ppv'][c], f1=r['f1'][c], auroc=au[c][0], auprc=au[c][1],
                               in_primary_macro=c in cls))
        pd.DataFrame(cm, index=[f'true_{c}' for c in rc.CLS], columns=[f'pred_{c}' for c in rc.CLS]).to_csv(os.path.join(OUT, f'{name}_{m}_confusion_counts.csv'))
        with np.errstate(invalid='ignore'):
            pd.DataFrame(cm / cm.sum(1, keepdims=True), index=[f'true_{c}' for c in rc.CLS], columns=[f'pred_{c}' for c in rc.CLS]).round(4).to_csv(os.path.join(OUT, f'{name}_{m}_confusion_rownorm.csv'))
    # ---- paired cluster bootstrap ----
    rng = np.random.RandomState(BSEED); boot = {m: [] for m in P}; bpc = {m: [] for m in P}
    for b in range(NB):
        sg = rng.choice(groups, len(groups), replace=True); ix = np.concatenate([gi[g] for g in sg])
        for m, p in P.items():
            cm = sum(gcm[m][g] for g in sg); r = from_cm(cm, cls); cal, _ = calib(p[ix], y[ix])
            au = [auc_pr(p[ix], y[ix], c)[0] for c in cls]
            boot[m].append(dict(accuracy=r['accuracy'], macro_f1=r['macro_f1'], balanced_accuracy=r['balanced_accuracy'], macro_auroc=float(np.nanmean(au)),
                                ece_top=cal['ece_top'], signed_gap=cal['signed_gap'], brier=cal['brier']))
            bpc[m].append(np.r_[r['sens'], r['ppv'], r['f1']])
    ci = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
    for m in P:
        bd = pd.DataFrame(boot[m])
        for k_ in bd.columns: point[m][f'{k_}_ci95'] = ci(bd[k_].values)
        arr = np.array(bpc[m])
        for row in pcrows:
            if row['model'] != m: continue
            c = rc.CLS.index(row['cls'])
            for j, met in enumerate(('sens', 'ppv', 'f1')):
                row[f'{met}_ci95'] = ci(arr[:, j * 5 + c]); row[f'{met}_boot_valid'] = int(np.isfinite(arr[:, j * 5 + c]).sum())
    rows = [dict(model=m, **v) for m, v in point.items()]
    # ---- pre-specified paired comparisons (primary B) + exploratory A vs C ----
    comp = []
    for a_, b_, kind in (('B', 'A', 'pre-specified'), ('B', 'C', 'pre-specified'), ('C', 'A', 'exploratory')):
        if a_ not in P or b_ not in P: continue
        dm = np.array([x['macro_f1'] for x in boot[a_]]) - np.array([x['macro_f1'] for x in boot[b_]])
        ds = np.array(bpc[a_])[:, :5] - np.array(bpc[b_])[:, :5]
        ca, cb = pred[a_] == y, pred[b_] == y; n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        mc = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else np.nan
        comp.append(dict(comparison=f'{a_} - {b_}', kind=kind, delta_macro_f1=point[a_]['macro_f1'] - point[b_]['macro_f1'], delta_macro_f1_ci95=ci(dm),
                         **{f'delta_sens_{rc.CLS[c]}_ci95': ci(ds[:, c]) for c in range(5)},
                         mcnemar_exact_p=float(mc), mcnemar_only_first_correct=n01, mcnemar_only_second_correct=n10,
                         mcnemar_note='identical beats; beats are clustered within subjects, so the exact McNemar p-value is approximate (independence violated)'))
    pd.DataFrame(rows).to_json(os.path.join(OUT, f'{name}_summary.json'), orient='records', indent=1)
    pd.DataFrame(pcrows).to_csv(os.path.join(OUT, f'{name}_per_class.csv'), index=False)
    pd.DataFrame(comp).to_json(os.path.join(OUT, f'{name}_comparisons.json'), orient='records', indent=1)
    return rows, pcrows, comp

if __name__ == '__main__':
    assert os.path.exists(os.path.join(SC, 'SCORED.json')), 'run 07_score.py first'
    # ---------- MIT-BIH internal ----------
    P, Praw, perfold = {}, {}, []
    for m in 'ABC':
        parts = [np.load(os.path.join(SC, f'mit_{m}_fold{k}.npz')) for k in range(rc.K)]
        order = np.argsort(np.concatenate([q['beat_index'] for q in parts]))
        lg = np.concatenate([q['logits'] for q in parts])[order]; T = np.concatenate([np.full(len(q['y']), float(q['T'])) for q in parts])[order]
        y = np.concatenate([q['y'] for q in parts])[order]; subj = np.concatenate([q['subject'] for q in parts])[order]
        P[m] = softmax(lg / T[:, None], 1); Praw[m] = softmax(lg, 1)
        for k, q in enumerate(parts):
            pr = softmax(q['logits'] / float(q['T']), 1).argmax(1); r = from_cm(cm_of(q['y'], pr), [c for c in range(5) if (q['y'] == c).any()])
            perfold.append(dict(model=m, fold=k, n=len(q['y']), accuracy=r['accuracy'], macro_f1_classes_present=r['macro_f1'],
                                **{f'sens_{rc.CLS[c]}': r['sens'][c] for c in range(5)}, **{f'n_{rc.CLS[c]}': int((q['y'] == c).sum()) for c in range(5)}))
    assert len(y) == 109305
    pd.DataFrame(perfold).to_csv(os.path.join(OUT, 'mit_internal_per_fold.csv'), index=False)
    evaluate('mit_internal', P, y, subj, cls=list(range(5)), Pextra=Praw)
    # ---------- INCART external (ensemble of 5 locked fold models) ----------
    lab = np.load(os.path.join(SC, 'incart_labels.npz')); yi, pat = lab['y'], lab['patient']
    for tag, key in (('incart_external', 'logits'), ('incart_external_recentred', 'logits_recentred')):
        PE, PEraw, pf = {}, {}, []
        for m in 'ABC':
            qs = [np.load(os.path.join(SC, f'incart_{m}_fold{k}.npz')) for k in range(rc.K)]
            PE[m] = np.mean([softmax(q[key] / float(q['T']), 1) for q in qs], 0); PEraw[m] = np.mean([softmax(q[key], 1) for q in qs], 0)
            for k, q in enumerate(qs):
                r = from_cm(cm_of(yi, softmax(q[key] / float(q['T']), 1).argmax(1)), [0, 1, 2, 3])
                pf.append(dict(model=m, fold_model=k, macro_f1_NSVF=r['macro_f1'], accuracy=r['accuracy'], **{f'sens_{rc.CLS[c]}': r['sens'][c] for c in range(4)}))
        pd.DataFrame(pf).to_csv(os.path.join(OUT, f'{tag}_individual_fold_models.csv'), index=False)
        evaluate(tag, PE, yi, pat, cls=[0, 1, 2, 3], Pextra=PEraw)
    print('evaluation written to', OUT)
