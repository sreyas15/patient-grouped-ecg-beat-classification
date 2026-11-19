"""INDEPENDENT recomputation of every reported metric from the saved score files (does not call the frozen evaluate()).
usage: python -P -B verify_independent.py mit | incart
mit    : reads results/scores_mit/*.npz (+ source labels in data/mitbih_beats_r2.npz, locked folds, data audit counts); compares with results/evaluation_mit/*
incart : (only after INCART authorisation) reads results/scores_incart/*.npz; compares with results/evaluation_incart/*
Writes results/verification_<mode>/verification_<mode>.json (exclusive-create). Own implementations: confusion-matrix metrics, rank-based AUROC, step-wise average precision,
ECE / class-wise ECE / MCE / Brier / NLL, subject-cluster bootstrap (same seed/RNG protocol as the frozen spec), McNemar exact test."""
import os, sys, json, ast, time, datetime
import numpy as np, pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata, binomtest
R2 = r"D:\explainable_ai\revision\R2"; sys.path.insert(0, R2 + r"\experiments"); sys.dont_write_bytecode = True
mode = sys.argv[1]; assert mode in ('mit', 'incart')
import a1common as a1
if mode == 'mit': a1.install_no_external_guard()
CLS = ['N', 'S', 'V', 'F', 'Q']; K = 5; NB = 2000; SEED = 20261102; NBIN = 15
SC = f"{R2}\\results\\scores_{mode}"; EV = f"{R2}\\results\\evaluation_{mode}"; OUTD = f"{R2}\\results\\verification_{mode}"
checks = []
def chk(name, ok, detail=''): checks.append(dict(check=name, ok=bool(ok), detail=str(detail)[:300])); print(('PASS ' if ok else 'FAIL ') + name, str(detail)[:150], flush=True)
def close(a, b, tol):
    a, b = np.asarray(a, float), np.asarray(b, float); both_nan = np.isnan(a) & np.isnan(b)
    return float(np.max(np.where(both_nan, 0, np.abs(np.nan_to_num(a, nan=1e9) - np.nan_to_num(b, nan=-1e9))))) if a.size else 0.0

# ---------- independent metric code ----------
def cm_np(y, pred): return np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
def rates(cm):
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp; tn = cm.sum() - tp - fp - fn
    with np.errstate(divide='ignore', invalid='ignore'): return dict(sens=tp / (tp + fn), spec=tn / (tn + fp), ppv=tp / (tp + fp), f1=2 * tp / (2 * tp + fp + fn))
def macro_f1(cm, cls): f = rates(cm)['f1'][cls]; return float(np.mean(np.where(np.isnan(f), 0.0, f)))      # absent class counts as 0 (frozen definition)
def bal_acc(cm, cls): s = rates(cm)['sens'][cls]; return float(np.nanmean(s))
def auroc_rank(s, pos):
    n1 = int(pos.sum()); n0 = len(pos) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def ap_step(s, pos):
    if pos.sum() == 0: return np.nan
    o = np.argsort(-s, kind='mergesort'); ss = s[o]; t = pos[o]; tp = np.cumsum(t); fp = np.cumsum(~t)
    idx = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]; prec = tp[idx] / (tp[idx] + fp[idx]); rec = tp[idx] / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
def calib_np(p, y):
    conf = p.max(1); hit = (p.argmax(1) == y).astype(float); b = np.minimum(np.floor(conf * NBIN).astype(int), NBIN - 1); ece = mce = 0.0; rows = []
    for k in range(NBIN):
        s = b == k
        if s.any(): g = conf[s].mean() - hit[s].mean(); ece += s.mean() * abs(g); mce = max(mce, abs(g)); rows.append((k, int(s.sum()), conf[s].mean(), hit[s].mean(), g))
    oh = np.eye(p.shape[1])[y]; cw = []
    for c in range(p.shape[1]):
        bc = np.minimum(np.floor(p[:, c] * NBIN).astype(int), NBIN - 1); e = 0.0
        for k in range(NBIN):
            s = bc == k
            if s.any(): e += s.mean() * abs(p[s, c].mean() - oh[s, c].mean())
        cw.append(e)
    return dict(ece_top=ece, signed_gap=float(conf.mean() - hit.mean()), mce=mce, ece_classwise=float(np.mean(cw)), brier=float(((p - oh) ** 2).sum(1).mean()), nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())), rows

def run_eval(tag, P, Praw, y, grp, cls, results):
    """Independent recomputation for one evaluation set; compares with files in EV."""
    S = {r['model']: r for r in json.load(open(f"{EV}\\{tag}_summary.json"))}; PCs = pd.read_csv(f"{EV}\\{tag}_per_class.csv"); CMP = {r['comparison']: r for r in json.load(open(f"{EV}\\{tag}_comparisons.json"))}
    groups = np.unique(grp); gi = {g: np.where(grp == g)[0] for g in groups}; pred = {m: P[m].argmax(1) for m in P}
    gcm = {m: {g: cm_np(y[gi[g]], pred[m][gi[g]]) for g in groups} for m in P}; res = {}
    for m in 'ABC':
        cm = cm_np(y, pred[m]); r = rates(cm); p = P[m]; cal, rows = calib_np(p, y); au = [auroc_rank(p[:, c], y == c) for c in range(K)]; ap = [ap_step(p[:, c], y == c) for c in range(K)]
        pt = dict(accuracy=float(np.trace(cm) / cm.sum()), macro_f1=macro_f1(cm, cls), balanced_accuracy=bal_acc(cm, cls), macro_auroc=float(np.nanmean([au[c] for c in cls])), macro_auprc=float(np.nanmean([ap[c] for c in cls])), **cal)
        rawcal, _ = calib_np(Praw[m], y); pt.update({f'raw_{k}': v for k, v in rawcal.items()})
        d = {k: abs(S[m][k] - v) for k, v in pt.items() if k in S[m]}
        chk(f'[{tag}] {m}: point metrics (accuracy, macro-F1, balanced acc., macro AUROC/AUPRC, ECE, signed gap, MCE, class-wise ECE, Brier, NLL, raw-score calibration) == independent', max(d.values()) < 1e-6, f'max abs diff {max(d.values()):.2e} over {len(d)} metrics')
        cmf = pd.read_csv(f'{EV}\\{tag}_{m}_confusion_counts.csv', index_col=0).values; chk(f'[{tag}] {m}: 5x5 confusion matrix == independent', np.array_equal(cmf, cm), cm.sum())
        rn = pd.read_csv(f'{EV}\\{tag}_{m}_confusion_rownorm.csv', index_col=0).values
        with np.errstate(invalid='ignore'): chk(f'[{tag}] {m}: row-normalised confusion matrix', close(rn, np.round(cm / cm.sum(1, keepdims=True), 4), 1e-9) < 1e-9)
        rel = pd.read_csv(f'{EV}\\{tag}_{m}_reliability_bins.csv'); ok = len(rel) == len(rows) and all(int(a.n) == b[1] and abs(a.mean_conf - b[2]) < 1e-9 and abs(a.accuracy - b[3]) < 1e-9 and abs(a.signed_gap - b[4]) < 1e-9 for a, b in zip(rel.itertuples(), rows)); chk(f'[{tag}] {m}: reliability bins (n, confidence, accuracy, signed gap)', ok, len(rows))
        pc = PCs[PCs.model == m].set_index('cls').loc[CLS]; sup = [int((y == c).sum()) for c in range(K)]
        ok = (np.allclose(pc.sens.values.astype(float), r['sens'], equal_nan=True, atol=1e-9) and np.allclose(pc.spec.values.astype(float), r['spec'], equal_nan=True, atol=1e-9) and np.allclose(pc.ppv.values.astype(float), r['ppv'], equal_nan=True, atol=1e-9)
              and np.allclose(pc.f1.values.astype(float), r['f1'], equal_nan=True, atol=1e-9) and np.allclose(pc.auroc.values.astype(float), au, equal_nan=True, atol=1e-9) and np.allclose(pc.auprc.values.astype(float), ap, equal_nan=True, atol=1e-9)
              and list(pc.support_beats) == sup and list(pc.support_subjects) == [len(np.unique(grp[y == c])) for c in range(K)] and list(pc.in_primary_macro) == [c in cls for c in range(K)])
        chk(f'[{tag}] {m}: per-class sensitivity, specificity, PPV, F1, AUROC, AUPRC, beat and cluster support, primary-macro flags == independent', ok)
        res[m] = dict(point=pt, cm=cm.tolist(), per_class=dict(sens=r['sens'].tolist(), spec=r['spec'].tolist(), ppv=r['ppv'].tolist(), f1=r['f1'].tolist(), auroc=au, auprc=ap), support=sup)
    # ---- independent cluster bootstrap (same RNG protocol), B=NB ----
    rng = np.random.RandomState(SEED); keys = ['accuracy', 'macro_f1', 'balanced_accuracy', 'macro_auroc', 'ece_top', 'signed_gap', 'brier']
    boot = {m: {k: [] for k in keys} for m in 'ABC'}; bpc = {m: [] for m in 'ABC'}; t0 = time.time()
    for b in range(NB):
        sg = rng.choice(groups, len(groups), replace=True); ix = np.concatenate([gi[g] for g in sg])
        for m in 'ABC':
            cm = sum(gcm[m][g] for g in sg); r = rates(cm); yy = y[ix]; pp = P[m][ix]; cal, _ = calib_np(pp, yy)
            boot[m]['accuracy'].append(np.trace(cm) / cm.sum()); boot[m]['macro_f1'].append(macro_f1(cm, cls)); boot[m]['balanced_accuracy'].append(bal_acc(cm, cls))
            boot[m]['macro_auroc'].append(float(np.nanmean([auroc_rank(pp[:, c], yy == c) for c in cls]))); boot[m]['ece_top'].append(cal['ece_top']); boot[m]['signed_gap'].append(cal['signed_gap']); boot[m]['brier'].append(cal['brier'])
            bpc[m].append(np.r_[r['sens'], r['ppv'], r['f1']])
        if b % 250 == 0: print(f'   bootstrap {b}/{NB} ({time.time() - t0:.0f}s)', flush=True)
    ci = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
    for m in 'ABC':
        bad = [k for k in keys if max(abs(a - b) for a, b in zip(S[m][f'{k}_ci95'], ci(boot[m][k]))) > 1e-9]; chk(f'[{tag}] {m}: 95% subject-cluster bootstrap CIs (accuracy, macro-F1, balanced acc., macro AUROC, ECE, signed gap, Brier) == independent bootstrap (B={NB})', not bad, bad)
        arr = np.array(bpc[m]); pc = PCs[PCs.model == m].set_index('cls').loc[CLS]; bad = []
        for c in range(K):
            for j, met in enumerate(('sens', 'ppv', 'f1')):
                ref = ci(arr[:, j * K + c]); got = ast.literal_eval(pc[f'{met}_ci95'].iloc[c]) if isinstance(pc[f'{met}_ci95'].iloc[c], str) else pc[f'{met}_ci95'].iloc[c]
                if np.any(np.isnan(ref)) != np.any(np.isnan(got)) or (not np.any(np.isnan(ref)) and max(abs(np.array(ref) - np.array(got))) > 1e-9) or int(pc[f'{met}_boot_valid'].iloc[c]) != int(np.isfinite(arr[:, j * K + c]).sum()): bad.append((CLS[c], met))
        chk(f'[{tag}] {m}: per-class sensitivity / PPV / F1 CIs and number of valid resamples == independent', not bad, bad[:4])
        res[m]['ci'] = {k: ci(boot[m][k]) for k in keys}
    for a_, b_ in (('B', 'A'), ('B', 'C'), ('C', 'A')):
        c = CMP[f'{a_} - {b_}']; dm = np.array(boot[a_]['macro_f1']) - np.array(boot[b_]['macro_f1']); ds = np.array(bpc[a_])[:, :K] - np.array(bpc[b_])[:, :K]
        n01 = int(((pred[a_] == y) & (pred[b_] != y)).sum()); n10 = int(((pred[a_] != y) & (pred[b_] == y)).sum()); pval = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else np.nan
        ok = (abs(c['delta_macro_f1'] - (res[a_]['point']['macro_f1'] - res[b_]['point']['macro_f1'])) < 1e-9 and max(abs(np.array(c['delta_macro_f1_ci95']) - np.array(ci(dm)))) < 1e-9 and c['mcnemar_only_first_correct'] == n01 and c['mcnemar_only_second_correct'] == n10
              and abs(c['mcnemar_exact_p'] - pval) < 1e-12 and all(max(abs(np.array(c[f'delta_sens_{CLS[k]}_ci95']) - np.array(ci(ds[:, k])))) < 1e-9 for k in range(K) if np.isfinite(ds[:, k]).any()))
        chk(f'[{tag}] paired {a_} - {b_}: delta macro-F1 + paired CI, per-class delta-sensitivity CIs, McNemar discordant counts and exact p == independent', ok, f'n01={n01} n10={n10} p={pval:.3g}')
        res[f'{a_}-{b_}'] = dict(delta_macro_f1=res[a_]['point']['macro_f1'] - res[b_]['point']['macro_f1'], ci=ci(dm), n01=n01, n10=n10, mcnemar_p=float(pval), kind=c['kind'])
    results[tag] = res

# ---------- load ----------
t00 = time.time(); results = {}
if mode == 'mit':
    folds = json.load(open(f"{R2}\\results\\splits\\folds_LOCKED.json"))['folds']; src = np.load(f"{R2}\\data\\mitbih_beats_r2.npz"); audit = json.load(open(f"{R2}\\results\\data_audit\\mitbih_summary.json"))
    P, Praw, perfold = {}, {}, []; ref = None
    for m in 'ABC':
        parts = [np.load(f"{SC}\\mit_{m}_fold{k}.npz") for k in range(5)]; bi = np.concatenate([q['beat_index'] for q in parts]); o = np.argsort(bi)
        lg = np.concatenate([q['logits'] for q in parts])[o]; T = np.concatenate([np.full(len(q['y']), float(q['T'])) for q in parts])[o]; y = np.concatenate([q['y'] for q in parts])[o]; subj = np.concatenate([q['subject'] for q in parts])[o]; rec = np.concatenate([q['record'] for q in parts])[o]
        P[m] = softmax(lg / T[:, None], 1); Praw[m] = softmax(lg, 1)
        chk(f'{m}: pooled predictions = 109,305, every beat index exactly once (0..109304)', len(bi) == 109305 and np.array_equal(np.sort(bi), np.arange(109305)), len(bi))
        chk(f'{m}: stored labels / subjects / records equal the source data at those beat indices', np.array_equal(y, src['y'][np.sort(bi)]) and np.array_equal(subj, src['subject'][np.sort(bi)]) and np.array_equal(rec, src['record'][np.sort(bi)]))
        fs = {}
        for k, q in enumerate(parts): fs[k] = sorted(set(q['subject'].tolist()))
        chk(f'{m}: each fold\'s test subjects == locked fold k; subjects are disjoint across folds; 201/202 grouped as 201_202', all(fs[k] == sorted(('201_202' if r == '201+202' else r) for r in folds[str(k)]) for k in range(5)) and sum(len(v) for v in fs.values()) == len(set(sum(fs.values(), []))) == 47 and '201_202' in fs[3], [len(v) for v in fs.values()])
        rr = {}; [rr.setdefault(r_, set()).add(s_) for r_, s_ in zip(rec.tolist(), subj.tolist())]; chk(f'{m}: every record belongs to exactly one subject (48 records, 47 subjects)', all(len(v) == 1 for v in rr.values()) and len(rr) == 48 and len(set(subj.tolist())) == 47, (len(rr), len(set(subj.tolist()))))
        cnt = np.bincount(y, minlength=5).tolist(); chk(f'{m}: true-label counts == data-audit counts and == frozen protocol (N 90,472 / S 2,775 / V 7,229 / F 802 / Q 8,027)', cnt == [audit['class_totals'][c] for c in CLS] == [90472, 2775, 7229, 802, 8027], cnt)
        pr = P[m].argmax(1); chk(f'{m}: predicted-label counts', True, np.bincount(pr, minlength=5).tolist())
        for k, q in enumerate(parts):
            pk = softmax(q['logits'] / float(q['T']), 1).argmax(1); cmk = cm_np(q['y'], pk); present = [c for c in range(5) if (q['y'] == c).any()]; rk = rates(cmk)
            perfold.append(dict(model=m, fold=k, n=len(q['y']), accuracy=float(np.trace(cmk) / cmk.sum()), macro_f1_classes_present=float(np.mean(np.where(np.isnan(rk['f1'][present]), 0, rk['f1'][present]))), **{f'sens_{CLS[c]}': float(rk['sens'][c]) for c in range(5)}, **{f'n_{CLS[c]}': int((q['y'] == c).sum()) for c in range(5)}))
        ref = (y, subj)
    y, subj = ref; pf = pd.read_csv(f'{EV}\\mit_internal_per_fold.csv'); mine = pd.DataFrame(perfold)
    num = [c for c in mine.columns if c not in ('model', 'fold')]; chk('per-fold table (n, accuracy, macro-F1 over classes present, per-class sensitivity, per-class counts) == independent', len(pf) == 15 and max(close(pf[c].values.astype(float), mine[c].values.astype(float), 1e-9) for c in num) < 1e-9)
    run_eval('mit_internal', P, Praw, y, subj, list(range(5)), results)
else:
    parts_ = {m: [np.load(f"{SC}\\incart_{m}_fold{k}.npz") for k in range(5)] for m in 'ABC'}; lab = np.load(f"{SC}\\incart_labels.npz"); y, pat, rec = lab['y'], lab['patient'], lab['record']
    chk('INCART: labels, patients, records present; counts', len(y) > 0, (len(y), len(np.unique(pat)), len(np.unique(rec)), np.bincount(y, minlength=5).tolist()))
    for tag, key in (('incart_external', 'logits'), ('incart_external_recentred', 'logits_recentred')):
        P = {m: np.mean([softmax(q[key] / float(q['T']), 1) for q in parts_[m]], 0) for m in 'ABC'}; Praw = {m: np.mean([softmax(q[key], 1) for q in parts_[m]], 0) for m in 'ABC'}
        run_eval(tag, P, Praw, y, pat, [0, 1, 2, 3], results)
        ind = pd.read_csv(f'{EV}\\{tag}_individual_fold_models.csv'); bad = []
        for _, r_ in ind.iterrows():
            q = parts_[r_.model][int(r_.fold_model)]; pk = softmax(q[key] / float(q['T']), 1).argmax(1); cmk = cm_np(y, pk)
            if abs(r_.macro_f1_NSVF - macro_f1(cmk, [0, 1, 2, 3])) > 1e-9: bad.append((r_.model, int(r_.fold_model)))
        chk(f'[{tag}] individual fold-model macro-F1 over N,S,V,F == independent', not bad, bad)
os.makedirs(OUTD, exist_ok=True)
rep = dict(mode=mode, created=datetime.datetime.now().isoformat(timespec='seconds'), seconds=round(time.time() - t00), n_checks=len(checks), n_failed=sum(not c['ok'] for c in checks), checks=checks, independent_results=results)
with open(f"{OUTD}\\verification_{mode}.json", 'x') as f: json.dump(rep, f, indent=1, default=str)
print(f"\nTOTAL checks {len(checks)} | FAILED {rep['n_failed']} | {rep['seconds']}s")
