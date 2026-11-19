"""R2 Stage 2a: build the model-input beat tensors (identical pipeline for MIT-BIH and INCART).

Per record (no parameter is fitted across records or subjects; no labels used in preprocessing):
  lead: MIT-BIH MLII where available, else V5 (records 102, 104); INCART lead II
  resample to 360 Hz if needed (INCART 257 -> 360, resample_poly(360, 257))
  denoise(): original db5 level-9 routine (run_local.py), verbatim
  per-record z-score of the denoised lead (manuscript Eq. 5)
  beats: annotation indices 2 .. len-4 (original edge rule), symbol in the 15-symbol table, window R-99 .. R+200 (300 samples, 360 Hz)
  model input: 300 -> 186 samples (scipy.signal.resample, FFT), identical for Models A, B, C (decision D1)
INCART additionally: peak-recentred windows (sensitivity analysis, decision D7): R re-set to argmax |x - median(x_window)|
  within +-75 ms (27 samples) of the annotation on the processed signal (signal only, no labels).

Outputs: revision/R2/data/mitbih_beats_r2.npz, revision/R2/data/incart_beats_r2.npz, results/data_audit/extract_summary.json
"""
import os, re, json, numpy as np, wfdb, pywt
from scipy.signal import resample, resample_poly

HERE = os.path.dirname(os.path.abspath(__file__)); R2 = os.path.dirname(HERE)
MIT = r"D:\explainable_ai\Journel paper project\WORKFLOW\Dataset\mit-bih-arrhythmia-database-1.0.0"
INC = os.path.join(R2, 'data', 'incartdb')
CS = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']
AAMI = {s: (0 if i <= 4 else 1 if i <= 8 else 2 if i <= 10 else 3 if i == 11 else 4) for i, s in enumerate(CS)}  # N,S,V,F,Q = 0..4
PRE, POST, L_IN, HALF = 99, 201, 186, 27

def denoise(data):  # verbatim logic from run_local.py
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    threshold = (np.median(np.abs(coeffs[-1])) / 0.6745) * (np.sqrt(2 * np.log(len(coeffs[-1]))))
    coeffs[-1].fill(0); coeffs[-2].fill(0)
    for i in range(1, len(coeffs) - 2): coeffs[i] = pywt.threshold(coeffs[i], threshold)
    return pywt.waverec(coeffs=coeffs, wavelet='db5')[:len(data)]
def zscore(x): return (x - x.mean()) / (x.std() + 1e-8)
def beats(s, samples, symbols, recentre=False):
    X, y, keep_t = [], [], []
    for i in range(2, len(symbols) - 3):
        c = symbols[i]
        if c not in AAMI: continue
        t = int(samples[i])
        if recentre and HALF <= t < len(s) - HALF:
            w = s[t - HALF:t + HALF + 1]; t = t - HALF + int(np.argmax(np.abs(w - np.median(w))))
        if t - PRE < 0 or t + POST > len(s): continue
        X.append(s[t - PRE:t + POST]); y.append(AAMI[c]); keep_t.append((int(samples[i]), t, c))
    X = resample(np.asarray(X, np.float64), L_IN, axis=1).astype(np.float32) if X else np.zeros((0, L_IN), np.float32)
    return X, np.array(y, np.int64), keep_t

summary = {}
# ---------------- MIT-BIH ----------------
recs = [l.strip() for l in open(os.path.join(MIT, 'RECORDS')) if l.strip()]
X, y, rec, sub, sym, smp, lead = [], [], [], [], [], [], []
for r in recs:
    h = wfdb.rdheader(os.path.join(MIT, r)); ch = 'MLII' if 'MLII' in h.sig_name else 'V5'
    s = zscore(denoise(wfdb.rdrecord(os.path.join(MIT, r), channel_names=[ch]).p_signal[:, 0]))
    a = wfdb.rdann(os.path.join(MIT, r), 'atr')
    xb, yb, tk = beats(s, a.sample, a.symbol)
    X.append(xb); y.append(yb); n = len(yb)
    rec += [r] * n; sub += ['201_202' if r in ('201', '202') else r] * n; lead += [ch] * n
    sym += [c for _, _, c in tk]; smp += [t for t, _, _ in tk]
X = np.concatenate(X); y = np.concatenate(y)
np.savez_compressed(os.path.join(R2, 'data', 'mitbih_beats_r2.npz'), X=X, y=y, record=np.array(rec), subject=np.array(sub),
                    symbol=np.array(sym), sample=np.array(smp), lead=np.array(lead))
summary['mitbih'] = dict(n=int(len(y)), class_counts=np.bincount(y, minlength=5).tolist(), subjects=len(set(sub)),
                         leads={l: int(c) for l, c in zip(*np.unique(lead, return_counts=True))},
                         records_by_lead={l: sorted(set(r_ for r_, l_ in zip(rec, lead) if l_ == l)) for l in set(lead) if l != 'MLII'},
                         shape=list(X.shape))
# ---------------- INCART ----------------
fpd = [l.strip() for l in open(os.path.join(INC, 'files-patients-diagnoses.txt'), encoding='latin-1')]
pmap = {r_: int(m.group(1)) for i, l in enumerate(fpd) if (m := re.fullmatch(r'patient\s+(\d+)', l)) for r_ in fpd[i + 1].split()}
irecs = [l.strip() for l in open(os.path.join(INC, 'RECORDS')) if l.strip()]
XA, XR, yi, ri, pi, si = [], [], [], [], [], []; shift = []
for r in irecs:
    h = wfdb.rdheader(os.path.join(INC, r)); assert h.fs == 257
    x = wfdb.rdrecord(os.path.join(INC, r), channel_names=['II']).p_signal[:, 0]
    s = zscore(denoise(resample_poly(x, 360, 257)))
    a = wfdb.rdann(os.path.join(INC, r), 'atr'); t360 = np.round(a.sample * 360 / 257).astype(int)
    xa, ya, ta = beats(s, t360, a.symbol); xr, yr, tr = beats(s, t360, a.symbol, recentre=True)
    # keep only beats present in both versions so that the sensitivity analysis is paired beat-for-beat
    ka = {t0: k for k, (t0, _, _) in enumerate(ta)}; kr = {t0: k for k, (t0, _, _) in enumerate(tr)}; common = [t0 for t0 in ka if t0 in kr]
    XA.append(xa[[ka[t] for t in common]]); XR.append(xr[[kr[t] for t in common]]); yi.append(ya[[ka[t] for t in common]])
    shift += [tr[kr[t]][1] - t for t in common]
    ri += [r] * len(common); pi += [pmap[r]] * len(common); si += [ta[ka[t]][2] for t in common]
XA, XR, yi = np.concatenate(XA), np.concatenate(XR), np.concatenate(yi); shift = np.array(shift)
np.savez_compressed(os.path.join(R2, 'data', 'incart_beats_r2.npz'), X=XA, X_recentred=XR, y=yi, record=np.array(ri), patient=np.array(pi),
                    symbol=np.array(si), recentre_shift_samples=shift)
summary['incart'] = dict(n=int(len(yi)), class_counts=np.bincount(yi, minlength=5).tolist(), patients=len(set(pi)), records=len(set(ri)), shape=list(XA.shape),
                         recentre_shift_samples=dict(median=float(np.median(shift)), p05=float(np.percentile(shift, 5)), p95=float(np.percentile(shift, 95)),
                                                     frac_zero=float((shift == 0).mean())))
json.dump(summary, open(os.path.join(R2, 'results', 'data_audit', 'extract_summary.json'), 'w'), indent=1)
print(json.dumps(summary, indent=1))
