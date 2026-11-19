"""R2 Stage 1a: MIT-BIH dataset audit (no model, no training).

Records per-record headers, leads, annotation symbol counts, the explicit
WFDB-symbol -> intermediate index -> AAMI class mapping used by the original
code (run_local.py / revision/prep_data.py), beats dropped by that mapping,
beats lost at record edges, and the subject grouping (201+202 = one subject,
MIT-BIH intro.htm).

Outputs: revision/R2/results/data_audit/mitbih_*.csv|json
"""
import os, json, collections
import numpy as np, pandas as pd, wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
R2 = os.path.dirname(HERE)
MIT = r"D:\explainable_ai\Journel paper project\WORKFLOW\Dataset\mit-bih-arrhythmia-database-1.0.0"
OUT = os.path.join(R2, "results", "data_audit"); os.makedirs(OUT, exist_ok=True)

# Exactly as in run_local.py (ecgClassSet) and the 15->5 loop
ECG_CLASS_SET = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
def aami(idx): return 'N' if idx <= 4 else 'S' if idx <= 8 else 'V' if idx <= 10 else 'F' if idx == 11 else 'Q'
SYMBOL_DESC = {  # PhysioNet WFDB annotation codes
    'N': 'Normal beat', 'L': 'Left bundle branch block beat', 'R': 'Right bundle branch block beat',
    'e': 'Atrial escape beat', 'j': 'Nodal (junctional) escape beat', 'A': 'Atrial premature beat',
    'a': 'Aberrated atrial premature beat', 'J': 'Nodal (junctional) premature beat',
    'S': 'Supraventricular premature or ectopic beat', 'V': 'Premature ventricular contraction',
    'E': 'Ventricular escape beat', 'F': 'Fusion of ventricular and normal beat', '/': 'Paced beat',
    'f': 'Fusion of paced and normal beat', 'Q': 'Unclassifiable beat',
    '!': 'Ventricular flutter wave', 'x': 'Non-conducted P-wave (blocked APC)', '|': 'Isolated QRS-like artifact',
    '~': 'Change in signal quality', '+': 'Rhythm change', '"': 'Comment annotation',
    '[': 'Start of ventricular flutter/fibrillation', ']': 'End of ventricular flutter/fibrillation'}
# Subject grouping: MIT-BIH Directory intro: "Records 201 and 202 came from the same male subject."
def subject_of(rec): return '201_202' if rec in ('201', '202') else rec
PACED = {'102', '104', '107', '217'}  # MIT-BIH Directory: records with paced rhythm

recs = [l.strip() for l in open(os.path.join(MIT, 'RECORDS')) if l.strip()]
assert len(recs) == 48
rows, symrows, mapcount = [], [], collections.Counter()
for r in recs:
    h = wfdb.rdheader(os.path.join(MIT, r))
    a = wfdb.rdann(os.path.join(MIT, r), 'atr')
    sym = np.array(a.symbol); smp = a.sample
    cnt = collections.Counter(sym)
    for s, c in cnt.items(): symrows.append(dict(record=r, symbol=s, count=c))
    # replicate original extraction: annotation indices 2 .. len-4 (i < len-3), symbol in ECG_CLASS_SET, window R-99:R+201
    kept = collections.Counter(); edge_short = 0; skipped_first_last = collections.Counter(); unmapped_beatlike = collections.Counter()
    for i, (s, t) in enumerate(zip(sym, smp)):
        if s not in ECG_CLASS_SET:
            if s in ('!', 'x', '|'): unmapped_beatlike[s] += 1
            continue
        if i < 2 or i >= len(sym) - 3: skipped_first_last[aami(ECG_CLASS_SET.index(s))] += 1; continue
        if t - 99 < 0 or t + 201 > h.sig_len: edge_short += 1; continue
        kept[aami(ECG_CLASS_SET.index(s))] += 1; mapcount[s] += 1
    rows.append(dict(record=r, subject=subject_of(r), fs=h.fs, sig_len=h.sig_len, duration_min=round(h.sig_len / h.fs / 60, 2),
                     ch0=h.sig_name[0], ch1=h.sig_name[1], paced_record=r in PACED,
                     header_comment=' | '.join(h.comments)[:200],
                     **{f'n_{c}': kept.get(c, 0) for c in 'NSVFQ'}, n_total=sum(kept.values()),
                     dropped_edge_window=edge_short, dropped_first2_last3=sum(skipped_first_last.values()),
                     unmapped_beatlike=dict(unmapped_beatlike)))

df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'mitbih_records.csv'), index=False)
pd.DataFrame(symrows).pivot_table(index='record', columns='symbol', values='count', fill_value=0).to_csv(os.path.join(OUT, 'mitbih_symbol_counts.csv'))
mp = pd.DataFrame([dict(wfdb_symbol=s, description=SYMBOL_DESC[s], intermediate_index=i, aami_class=aami(i), beats_extracted=mapcount.get(s, 0))
                   for i, s in enumerate(ECG_CLASS_SET)])
mp.to_csv(os.path.join(OUT, 'mitbih_label_mapping.csv'), index=False)
tot = {c: int(df[f'n_{c}'].sum()) for c in 'NSVFQ'}
summary = dict(records=len(df), subjects=int(df.subject.nunique()), fs=sorted(df.fs.unique().tolist()),
               ch0_names=df.ch0.value_counts().to_dict(), records_ch0_not_MLII=df.loc[df.ch0 != 'MLII', 'record'].tolist(),
               class_totals=tot, total_beats=int(sum(tot.values())),
               Q_beats_in_paced_records=int(df.loc[df.paced_record, 'n_Q'].sum()),
               Q_beats_in_nonpaced_records=int(df.loc[~df.paced_record, 'n_Q'].sum()),
               dropped_edge_window=int(df.dropped_edge_window.sum()), dropped_first2_last3=int(df.dropped_first2_last3.sum()),
               unmapped_beatlike_total=dict(sum((collections.Counter(d) for d in df.unmapped_beatlike), collections.Counter())),
               records_with_class={c: int((df[f'n_{c}'] > 0).sum()) for c in 'NSVFQ'},
               note='RR<250 ms exclusion stated in the manuscript is NOT implemented in run_local.py or prep_data.py')
json.dump(summary, open(os.path.join(OUT, 'mitbih_summary.json'), 'w'), indent=1, default=int)
print(json.dumps(summary, indent=1, default=int))
pd.set_option('display.width', 250)
print(df[['record', 'subject', 'ch0', 'ch1', 'paced_record', 'n_N', 'n_S', 'n_V', 'n_F', 'n_Q', 'n_total', 'unmapped_beatlike']].to_string(index=False))
print(mp.to_string(index=False))
