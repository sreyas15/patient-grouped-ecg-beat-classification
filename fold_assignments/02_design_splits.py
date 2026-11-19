"""R2 Stage 1b: PROPOSED patient-grouped fold assignment for MIT-BIH (not locked until author approval).

Unit of assignment = subject (47 subjects; records 201 and 202 = one subject).
Uses ONLY per-subject beat-label counts from the audit (no model output, no performance metric).

Design: 5 outer folds. For outer fold k: test = fold k, validation = fold (k+1) mod 5, training = remaining 3 folds.
Hard constraints (pre-specified):
  H1  the four paced subjects (102, 104, 107, 217; 99.8% of Q beats) are in four different folds;
  H2  records 208 and 213 (91.5% of F beats) are in different, cyclically NON-adjacent folds, so that for every
      outer fold at least one of them is in the training folds;
  H3  every fold contains >= 1 subject with S beats and >= 1 subject with V beats.
Soft objective: sum over classes and folds of the squared relative deviation of fold class counts from total/5,
plus the same term for total beats. Best of N random assignments under a fixed seed.
"""
import os, json, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); R2 = os.path.dirname(HERE)
AUD = os.path.join(R2, 'results', 'data_audit'); OUT = os.path.join(R2, 'results', 'splits'); os.makedirs(OUT, exist_ok=True)
SEED, N_TRIALS, K = 20261102, 200_000, 5
C = list('NSVFQ')

rec = pd.read_csv(os.path.join(AUD, 'mitbih_records.csv'), dtype={'record': str, 'subject': str})
sub = rec.groupby('subject')[[f'n_{c}' for c in C] + ['n_total']].sum()
sub['records'] = rec.groupby('subject')['record'].apply(lambda s: '+'.join(s))
S = sub.index.to_numpy(); cnt = sub[[f'n_{c}' for c in C]].to_numpy(float); tot = sub['n_total'].to_numpy(float)
paced = [np.where(S == s)[0][0] for s in ('102', '104', '107', '217')]
i208, i213 = np.where(S == '208')[0][0], np.where(S == '213')[0][0]
target = cnt.sum(0) / K; ttarget = tot.sum() / K

def cost(a):
    fc = np.stack([cnt[a == k].sum(0) for k in range(K)]); ft = np.array([tot[a == k].sum() for k in range(K)])
    return float((((fc - target) / target) ** 2).sum() + (((ft - ttarget) / ttarget) ** 2).sum())
def feasible(a):
    if len(set(a[paced])) != 4: return False
    d = (a[i208] - a[i213]) % K
    if d in (0, 1, K - 1): return False
    for k in range(K):
        m = a == k
        if cnt[m, 1].sum() == 0 or cnt[m, 2].sum() == 0: return False
    return True

rng = np.random.RandomState(SEED); best = (np.inf, None); n_feas = 0
base = np.arange(len(S)) % K   # near-equal subject numbers per fold (10,10,9,9,9)
for _ in range(N_TRIALS):
    a = rng.permutation(base)
    if not feasible(a): continue
    n_feas += 1; c = cost(a)
    if c < best[0]: best = (c, a.copy())
c, a = best
folds = {k: sorted(S[a == k].tolist()) for k in range(K)}
rows = []
for k in range(K):
    v = (k + 1) % K; tr = [j for j in range(K) if j not in (k, v)]
    part = {'train': np.isin(a, tr), 'val': a == v, 'test': a == k}
    for p, m in part.items():
        rows.append(dict(outer_fold=k, partition=p, n_subjects=int(m.sum()), **{c_: int(cnt[m, i].sum()) for i, c_ in enumerate(C)}, beats=int(tot[m].sum()),
                         subjects=' '.join(sorted(S[m].tolist()))))
tab = pd.DataFrame(rows)
# leakage self-check: subject sets disjoint within every outer fold; every subject tested exactly once
for k in range(K):
    t = tab[tab.outer_fold == k].set_index('partition')['subjects'].map(lambda s: set(s.split()))
    assert not (t['train'] & t['val']) and not (t['train'] & t['test']) and not (t['val'] & t['test'])
assert sorted(sum((folds[k] for k in range(K)), [])) == sorted(S.tolist())
out = dict(status='PROPOSED - not locked; awaiting author approval', seed=SEED, n_trials=N_TRIALS, n_feasible=n_feas, cost=c,
           unit='subject', n_subjects=len(S), grouping={'201_202': ['201', '202']},
           scheme='outer fold k: test=fold k, val=fold (k+1)%5, train=other 3 folds',
           folds={str(k): [sub.loc[s, 'records'] for s in folds[k]] for k in range(K)})
json.dump(out, open(os.path.join(OUT, 'proposed_folds.json'), 'w'), indent=1)
tab.to_csv(os.path.join(OUT, 'proposed_fold_class_support.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
print(json.dumps({k: v for k, v in out.items() if k != 'folds'}, indent=1))
for k in range(K): print('fold', k, ':', ' '.join(out['folds'][str(k)]))
print(tab.drop(columns='subjects').to_string(index=False))
