"""Compare every number in manuscript Tables 2, 4, 5, 7-10 with the tables regenerated from the locked results (generate_tables.py), and re-derive the in-text
failure-mode / concentration / comparison numbers from the locked confusion matrices, comparison files and per-beat scores. Read-only.
Usage: python check_manuscript_tables.py --docx <manuscript.docx> --tables <generated dir> --root <archive or R2 root> [--layout original]"""
import argparse, csv, json, os, sys, docx, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--docx', required=True); ap.add_argument('--tables', required=True)
ap.add_argument('--root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); ap.add_argument('--layout', default='archive', choices=['archive', 'original'])
ap.add_argument('--out', default='check_manuscript_tables_result.json'); a = ap.parse_args()
L = dict(archive=dict(mit='bootstrap/evaluation_mit', inc='bootstrap/evaluation_incart', sc='predictions/mitbih'),
         original=dict(mit='results/evaluation_mit', inc='results/evaluation_incart', sc='results/scores_mit'))[a.layout]
P = lambda k, *f: os.path.join(a.root, L[k], *f)
d = docx.Document(a.docx)
T = {i + 1: [[' '.join(c.text.split()) for c in r.cells] for r in t.rows] for i, t in enumerate(d.tables)}
fails = []; n = [0]
def chk(name, got, exp):
    n[0] += 1
    if got != exp: fails.append([name, str(got), str(exp)])
def gen(fn): return [[' '.join(c.split()) for c in r] for r in list(csv.reader(open(os.path.join(a.tables, fn), encoding='utf-8')))[1:]]
for num, fn in ((2, 'Table_2_class_distribution.csv'), (4, 'Table_4_parameters.csv'), (5, 'Table_5_calibration.csv'), (7, 'Table_7_ModelB_MITBIH.csv'),
                (8, 'Table_8_ModelsAC_MITBIH.csv'), (9, 'Table_9_ModelB_INCART.csv'), (10, 'Table_10_INCART_ensembles.csv')):
    g = gen(fn); m = T[num][1:]
    chk(f'Table {num} n_rows', len(m), len(g))
    for i, (rm, rg) in enumerate(zip(m, g)):
        if num == 4: rg = [{'A': 'A: 1-D CNN', 'B': 'B: CNN-LSTM-Att. (port)', 'C': 'C: ECGTransForm'}[rg[0]], rg[1]]
        chk(f'Table {num} row {i + 1}', rm, rg)
# in-text numbers re-derived from locked files
R = lambda k, f: P(k, f)
C = np.array([[int(v) for v in r[1:]] for r in list(csv.reader(open(R('mit', 'mit_internal_B_confusion_counts.csv'))))[1:]]); tot = C.sum(1)
chk('S->N %', round(100 * C[1, 0] / tot[1], 1), 61.1); chk('S->V %', round(100 * C[1, 2] / tot[1], 1), 9.4)
chk('F->N %', round(100 * C[3, 0] / tot[3], 1), 61.3); chk('F->V %', round(100 * C[3, 2] / tot[3], 1), 17.5); chk('F->Q %', round(100 * C[3, 4] / tot[3], 1), 1.6)
chk('true N predicted F', int(C[0, 3]), 5627); chk('true N predicted V', int(C[0, 2]), 7240); chk('false F predictions', int(C[:, 3].sum() - C[3, 3]), 6142); chk('all F predictions', int(C[:, 3].sum()), 6287)
chk('F precision %', round(100 * C[3, 3] / C[:, 3].sum(), 1), 2.3); chk('V precision %', round(100 * C[2, 2] / C[:, 2].sum(), 1), 42.8)
Ci = np.array([[int(v) for v in r[1:]] for r in list(csv.reader(open(R('inc', 'incart_external_B_confusion_counts.csv'))))[1:]])
chk('INCART Q predictions', int(Ci[:, 4].sum()), 6801); chk('INCART true Q recovered', int(Ci[4, 4]), 0)
cm = {x['comparison']: x for x in json.load(open(R('mit', 'mit_internal_comparisons.json')))}
chk('B-A', round(cm['B - A']['delta_macro_f1'], 3), 0.051); chk('B-C', round(cm['B - C']['delta_macro_f1'], 3), 0.028); chk('C-A', round(cm['C - A']['delta_macro_f1'], 3), 0.023)
chk('B-A CI', [round(v, 3) for v in cm['B - A']['delta_macro_f1_ci95']], [-0.005, 0.163]); chk('B-C CI', [round(v, 3) for v in cm['B - C']['delta_macro_f1_ci95']], [-0.01, 0.075])
chk('C-A CI', [round(v, 3) for v in cm['C - A']['delta_macro_f1_ci95']], [-0.017, 0.093])
chk('McNemar B-A', f"{cm['B - A']['mcnemar_exact_p']:.1e}", '1.2e-18'); chk('McNemar B-C', f"{cm['B - C']['mcnemar_exact_p']:.1e}", '4.7e-03')
ic = {x['comparison']: x for x in json.load(open(R('inc', 'incart_external_comparisons.json')))}
chk('INCART B-A', round(ic['B - A']['delta_macro_f1'], 3), -0.066); chk('INCART B-C', round(ic['B - C']['delta_macro_f1'], 3), -0.061)
chk('INCART B-A CI', [round(v, 3) for v in ic['B - A']['delta_macro_f1_ci95']], [-0.096, -0.035]); chk('INCART B-C CI', [round(v, 3) for v in ic['B - C']['delta_macro_f1_ci95']], [-0.104, -0.01])
sc = [np.load(os.path.join(a.root, L['sc'], f'mit_B_fold{k}.npz')) for k in range(5)]
rec = np.concatenate([s['record'] for s in sc]); y = np.concatenate([s['y'] for s in sc]); pr = np.concatenate([s['logits'] for s in sc]).argmax(1)
chk('F beats from 208+213 %', round(100 * np.isin(rec[y == 3], ['208', '213']).mean(), 1), 91.5); chk('S beats from 232 %', round(100 * (rec[y == 1] == '232').mean(), 1), 49.7)
fp = (pr == 3) & (y != 3); top = {r: int(((rec == r) & fp).sum()) for r in ['234', '109', '105']}
chk('false-F by record', top, {'234': 1401, '109': 1033, '105': 957}); chk('top-3 share of false F %', round(100 * sum(top.values()) / fp.sum()), 55)
p3 = sc[3]['logits'].argmax(1); chk('fold3 predicted F', int((p3 == 3).sum()), 3028); chk('fold3 n', len(p3), 21564); chk('fold3 true F', int((sc[3]['y'] == 3).sum()), 11); chk('fold3 true N predicted F', int(((p3 == 3) & (sc[3]['y'] == 0)).sum()), 2940)
res = dict(checks=n[0], failures=fails); json.dump(res, open(a.out, 'w'), indent=1); print('checks', n[0], 'failures', len(fails)); [print('FAIL', f) for f in fails]
sys.exit(1 if fails else 0)
