"""Generate the numeric tables of the manuscript (Tables 2, 4, 5, 7, 8, 9, 10) directly from the LOCKED result files.
Read-only on the locked outputs; writes CSV files only.
Usage (archive layout):   python generate_tables.py --root <archive root> --out <dir>
Usage (original layout):  python generate_tables.py --root <R2 root> --layout original --arch <architecture_verification.json> --out <dir>"""
import argparse, csv, json, ast, os

ap = argparse.ArgumentParser()
ap.add_argument('--root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
ap.add_argument('--layout', default='archive', choices=['archive', 'original'])
ap.add_argument('--arch', default=None); ap.add_argument('--out', default='tables_generated')
a = ap.parse_args()
L = dict(archive=dict(mit='bootstrap/evaluation_mit', inc='bootstrap/evaluation_incart', ext='preprocessing/data_audit/extract_summary.json',
                      arch='figure_table_generation/architecture/architecture_verification.json'),
         original=dict(mit='results/evaluation_mit', inc='results/evaluation_incart', ext='results/data_audit/extract_summary.json', arch=None))[a.layout]
P = lambda k: os.path.join(a.root, L[k])
ARCH = a.arch or P('arch')
os.makedirs(a.out, exist_ok=True)
CLS = ['N', 'S', 'V', 'F', 'Q']

def rows(p): return list(csv.DictReader(open(p, encoding='utf-8')))
ci = ast.literal_eval
p1 = lambda x: f'{100 * float(x):.1f}'
fci = lambda c, nd=1: f'[{100 * c[0]:.{nd}f}, {100 * c[1]:.{nd}f}]'
def write(name, header, body):
    with open(os.path.join(a.out, name), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(body)
def cls_cells(r):
    return [f"{p1(r['sens'])} {fci(ci(r['sens_ci95']))}", p1(r['spec']), f"{p1(r['ppv'])} {fci(ci(r['ppv_ci95']))}", f"{p1(r['f1'])} {fci(ci(r['f1_ci95']))}", f"{float(r['auroc']):.3f}", f"{float(r['auprc']):.3f}"]
def overall(s):
    return [f"acc. {p1(s['accuracy'])} {fci(s['accuracy_ci95'])}", '', f"bal. acc. {p1(s['balanced_accuracy'])}", f"macro-F1 {p1(s['macro_f1'])} {fci(s['macro_f1_ci95'])}", f"{s['macro_auroc']:.3f}", f"{s['macro_auprc']:.3f}"]
HDR = ['Class (beats / patients)', 'Sensitivity % [95% CI]', 'Spec. %', 'Precision % [95% CI]', 'F1 % [95% CI]', 'AUROC', 'AUPRC']
mit = {x['model']: x for x in json.load(open(os.path.join(P('mit'), 'mit_internal_summary.json')))}
inc = {x['model']: x for x in json.load(open(os.path.join(P('inc'), 'incart_external_summary.json')))}
rec = {x['model']: x for x in json.load(open(os.path.join(P('inc'), 'incart_external_recentred_summary.json')))}
pcm = [r for r in rows(os.path.join(P('mit'), 'mit_internal_per_class.csv'))]; pci = rows(os.path.join(P('inc'), 'incart_external_per_class.csv'))
def sup(r, c): return f"{c} ({int(r['support_beats']):,} / {r['support_subjects']})"
# Table 7
bm = {r['cls']: r for r in pcm if r['model'] == 'B'}
write('Table_7_ModelB_MITBIH.csv', HDR, [[sup(bm[c], c)] + cls_cells(bm[c]) for c in CLS] + [['Overall'] + overall(mit['B'])])
# Table 8
body = []
for m in 'AC':
    pm = {r['cls']: r for r in pcm if r['model'] == m}; body.append([f'Model {m}'] + [''] * 6)
    body += [[sup(pm[c], c)] + cls_cells(pm[c]) for c in CLS] + [['Overall'] + overall(mit[m])]
write('Table_8_ModelsAC_MITBIH.csv', HDR, body)
# Table 9
bi = {r['cls']: r for r in pci if r['model'] == 'B'}
def sup9(r, c): return sup(r, c) if c != 'Q' else f"Q (not in macro) ({int(r['support_beats']):,} / {r['support_subjects']})"
write('Table_9_ModelB_INCART.csv', HDR, [[sup9(bi[c], c)] + cls_cells(bi[c]) for c in CLS] + [['Overall'] + overall(inc['B'])])
# Table 10
write('Table_10_INCART_ensembles.csv', ['Model', 'Accuracy % [95% CI]', 'Macro-F1 % [CI], as given', 'Balanced acc. %', 'Macro AUROC', 'Macro-F1 % [CI], recentred'],
      [[f'Model {m}', f"{p1(inc[m]['accuracy'])} {fci(inc[m]['accuracy_ci95'])}", f"{p1(inc[m]['macro_f1'])} {fci(inc[m]['macro_f1_ci95'])}", p1(inc[m]['balanced_accuracy']),
        f"{inc[m]['macro_auroc']:.3f}", f"{p1(rec[m]['macro_f1'])} {fci(rec[m]['macro_f1_ci95'])}"] for m in 'ABC'])
# Table 5
write('Table_5_calibration.csv', ['Model', 'ECE (scaled) [95% CI]', 'ECE (raw)', 'Signed gap', 'MCE', 'Class-wise ECE', 'Brier', 'NLL'],
      [[f'Model {m}', f"{mit[m]['ece_top']:.3f} [{mit[m]['ece_top_ci95'][0]:.3f}, {mit[m]['ece_top_ci95'][1]:.3f}]", f"{mit[m]['raw_ece_top']:.3f}", f"{mit[m]['signed_gap']:+.3f}",
        f"{mit[m]['mce']:.3f}", f"{mit[m]['ece_classwise']:.3f}", f"{mit[m]['brier']:.3f}", f"{mit[m]['nll']:.3f}"] for m in 'ABC'])
# Table 2 (class distribution; order as in the manuscript)
cc = json.load(open(P('ext')))['mitbih']['class_counts']; n = sum(cc); idx = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
DESC = {'N': 'Normal sinus, LBBB, RBBB, atrial/nodal escape', 'V': 'PVC, ventricular escape', 'S': 'APB, aberrated APB, nodal, SV premature',
        'F': 'Fusion (ventricular + normal)', 'Q': 'Paced, fusion of paced and normal, unclassifiable'}
write('Table_2_class_distribution.csv', ['AAMI', 'Description', 'Count', 'Proportion'],
      [[c, DESC[c], f'{cc[idx[c]]:,}', f'{100 * cc[idx[c]] / n:.1f}%'] for c in ['N', 'V', 'S', 'F', 'Q']] + [['Total', '', f'{n:,}', '100%']])
# Table 4 (trainable parameters from the model objects)
pc = json.load(open(ARCH))['parameters_trainable']
write('Table_4_parameters.csv', ['Model', 'Trainable parameters'], [[f'{m}', f'{pc[m]:,}'] for m in 'ABC'])
print('tables written to', a.out)
