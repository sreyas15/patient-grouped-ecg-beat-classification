"""R2 Stage 1c: INCART (St Petersburg INCART 12-lead Arrhythmia Database 1.0.0) audit as a candidate EXTERNAL test set.

No model is loaded. INCART is never used for training, tuning, calibration or model selection.
Checks: file integrity (done at download), headers (fs, leads, duration), record->patient mapping, annotation symbols,
AAMI mapping coverage, paced/Q content, lead II availability, and R-peak annotation alignment after the planned
resampling (257 Hz -> 360 Hz), measured identically on MIT-BIH (reference) so the two are comparable.

Outputs: revision/R2/results/data_audit/incart_*.csv|json
"""
import os, re, json, collections
import numpy as np, pandas as pd, wfdb
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__)); R2 = os.path.dirname(HERE)
INC = os.path.join(R2, 'data', 'incartdb'); MIT = r"D:\explainable_ai\Journel paper project\WORKFLOW\Dataset\mit-bih-arrhythmia-database-1.0.0"
OUT = os.path.join(R2, 'results', 'data_audit'); os.makedirs(OUT, exist_ok=True)
ECG_CLASS_SET = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
def aami(s): i = ECG_CLASS_SET.index(s); return 'N' if i <= 4 else 'S' if i <= 8 else 'V' if i <= 10 else 'F' if i == 11 else 'Q'

def r_offset(sig, fs, ann, half_ms=75):
    """Offset (ms) between each annotation and the max |x - median| within +-half_ms (R-peak proxy)."""
    h = int(half_ms * fs / 1000); off = []
    for t in ann:
        if t - h < 0 or t + h >= len(sig): continue
        w = sig[t - h:t + h + 1]; off.append((np.argmax(np.abs(w - np.median(w))) - h) * 1000 / fs)
    return np.array(off)

# ---------- MIT-BIH reference alignment (same measure, ch0, 360 Hz, raw signal) ----------
mit_off = []
for r in [l.strip() for l in open(os.path.join(MIT, 'RECORDS')) if l.strip()]:
    rec = wfdb.rdrecord(os.path.join(MIT, r), channels=[0]); a = wfdb.rdann(os.path.join(MIT, r), 'atr')
    t = np.array([s for s, c in zip(a.sample, a.symbol) if c in ECG_CLASS_SET])
    mit_off.append(r_offset(rec.p_signal[:, 0], 360, t))
mit_off = np.concatenate(mit_off)

# ---------- INCART ----------
recs = [l.strip() for l in open(os.path.join(INC, 'RECORDS')) if l.strip()]
rows, symrows, inc_off, inc_off_class = [], [], [], collections.defaultdict(list)
for r in recs:
    h = wfdb.rdheader(os.path.join(INC, r)); a = wfdb.rdann(os.path.join(INC, r), 'atr')
    sym = np.array(a.symbol); cnt = collections.Counter(sym)
    for s, c in cnt.items(): symrows.append(dict(record=r, symbol=s, count=c))
    names = [n.strip() for n in h.sig_name]
    lead = 'II' if 'II' in names else None
    kept = collections.Counter(aami(s) for s in sym if s in ECG_CLASS_SET)
    if lead:
        x = wfdb.rdrecord(os.path.join(INC, r), channel_names=[lead]).p_signal[:, 0]
        x360 = resample_poly(x, 360, int(h.fs))            # planned conversion, gcd(360,257)=1
        tb = [(int(round(t * 360 / h.fs)), aami(s)) for t, s in zip(a.sample, sym) if s in ECG_CLASS_SET]
        o = r_offset(x360, 360, [t for t, _ in tb]); inc_off.append(o)
        o_all = r_offset(x360, 360, [t for t, _ in tb])
        for (t, c), v in zip([p for p in tb if int(75 * 360 / 1000) <= p[0] < len(x360) - int(75 * 360 / 1000)], o_all): inc_off_class[c].append(v)
    rows.append(dict(record=r, fs=h.fs, sig_len=h.sig_len, duration_min=round(h.sig_len / h.fs / 60, 2), n_leads=len(names),
                     leads=','.join(names), lead_II_present=lead is not None, header_comment=' | '.join(h.comments)[:300],
                     **{f'n_{c}': kept.get(c, 0) for c in 'NSVFQ'}, n_beats_mapped=sum(kept.values()),
                     unmapped_symbols={s: int(c) for s, c in cnt.items() if s not in ECG_CLASS_SET}))
df = pd.DataFrame(rows)

# record -> patient mapping from files-patients-diagnoses.txt (format audited, not assumed)
fpd = open(os.path.join(INC, 'files-patients-diagnoses.txt'), encoding='latin-1').read()
pmap, pdx, lines = {}, {}, [l.strip() for l in fpd.splitlines()]
for i, l in enumerate(lines):
    m = re.fullmatch(r'patient\s+(\d+)', l)
    if m:
        p = int(m.group(1)); recs_p = lines[i + 1].split()
        dx = lines[i + 2] if i + 2 < len(lines) and not lines[i + 2].startswith('patient') else ''
        for r_ in recs_p: pmap[r_] = p
        pdx[p] = dx
assert set(pmap) == set(recs), 'record->patient mapping incomplete'
df.insert(1, 'patient', df.record.map(pmap)); df.insert(2, 'diagnosis', df.patient.map(pdx))
df.to_csv(os.path.join(OUT, 'incart_records.csv'), index=False)
pd.DataFrame(symrows).pivot_table(index='record', columns='symbol', values='count', fill_value=0).to_csv(os.path.join(OUT, 'incart_symbol_counts.csv'))
q = lambda v: {k: round(float(np.percentile(v, p)), 2) for k, p in (('p05', 5), ('median', 50), ('p95', 95))} | {'mean': round(float(v.mean()), 2), 'frac_abs_le_14ms': round(float((np.abs(v) <= 14).mean()), 3), 'n': int(len(v))}
inc_off = np.concatenate(inc_off)
summ = dict(records=len(df), patients=int(df.patient.nunique()),
            patients_with_class={c: int(df.groupby('patient')[f'n_{c}'].sum().gt(0).sum()) for c in 'NSVFQ'}, fs=sorted(df.fs.unique().tolist()), duration_min=sorted(df.duration_min.unique().tolist()),
            lead_sets=df.leads.value_counts().to_dict(), lead_II_in_all=bool(df.lead_II_present.all()),
            class_totals={c: int(df[f'n_{c}'].sum()) for c in 'NSVFQ'}, total_mapped=int(df.n_beats_mapped.sum()),
            records_with_class={c: int((df[f'n_{c}'] > 0).sum()) for c in 'NSVFQ'},
            unmapped_symbol_totals=dict(sum((collections.Counter(d) for d in df.unmapped_symbols), collections.Counter())),
            paced_symbols_present=int(sum(df.unmapped_symbols.map(lambda d: 0)) + pd.DataFrame(symrows).query("symbol in ['/','f']")['count'].sum()),
            r_alignment_ms={'MITBIH_ch0_360Hz_reference': q(mit_off), 'INCART_leadII_resampled_360Hz': q(inc_off),
                            'INCART_by_class': {c: q(np.array(v)) for c, v in inc_off_class.items() if len(v)}},
            alignment_note='offset = argmax |x - median| within +-75 ms of the annotation, minus annotation position; positive = peak after annotation; 14 ms = 5 samples at 360 Hz')
json.dump(summ, open(os.path.join(OUT, 'incart_summary.json'), 'w'), indent=1, default=int)
print(json.dumps(summ, indent=1, default=int))
pd.set_option('display.width', 250)
print(df[['record', 'fs', 'duration_min', 'n_N', 'n_S', 'n_V', 'n_F', 'n_Q', 'unmapped_symbols', 'header_comment']].to_string(index=False))
print('---- files-patients-diagnoses.txt (verbatim, first 4000 chars) ----'); print(fpd[:4000])
