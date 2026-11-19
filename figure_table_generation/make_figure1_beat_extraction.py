"""Figure 1: R-peak-anchored beat extraction on MIT-BIH, using the LOCKED preprocessing of experiments/04_extract_beats.py
(denoise() = db5 level-9 routine, per-record z-score, window R-99 .. R+200 = 300 samples at 360 Hz).
Illustration only; no result is computed. Beats shown are chosen by a fixed rule: first retained N and V beat of record 100,
first retained F beat of record 108 (if absent, the first retained F beat of the lowest-numbered record that has one)."""
import os, numpy as np, wfdb, pywt
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
MIT = os.environ.get('MITBIH_DIR', 'mit-bih-arrhythmia-database-1.0.0')   # PhysioNet MIT-BIH download
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
CS = ['N','L','R','e','j','A','a','J','S','V','E','F','/','f','Q']
PRE, POST = 99, 201
def denoise(data):                      # verbatim from 04_extract_beats.py
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    threshold = (np.median(np.abs(coeffs[-1])) / 0.6745) * (np.sqrt(2 * np.log(len(coeffs[-1]))))
    coeffs[-1].fill(0); coeffs[-2].fill(0)
    for i in range(1, len(coeffs) - 2): coeffs[i] = pywt.threshold(coeffs[i], threshold)
    return pywt.waverec(coeffs=coeffs, wavelet='db5')[:len(data)]
def zscore(x): return (x - x.mean()) / (x.std() + 1e-8)
def load(r):
    h = wfdb.rdheader(os.path.join(MIT, r)); ch = 'MLII' if 'MLII' in h.sig_name else 'V5'
    s = zscore(denoise(wfdb.rdrecord(os.path.join(MIT, r), channel_names=[ch]).p_signal[:, 0])); a = wfdb.rdann(os.path.join(MIT, r), 'atr')
    return s, a, ch
def first(s, a, sym_set):
    for i in range(2, len(a.symbol) - 3):
        t = int(a.sample[i])
        if a.symbol[i] in sym_set and t - PRE >= 0 and t + POST <= len(s): return i, t
    return None
s100, a100, ch100 = load('100')
n_i, n_t = first(s100, a100, {'N'}); v_i, v_t = first(s100, a100, {'V'})
sel_f = None
for r in ['108'] + [x.strip() for x in open(os.path.join(MIT, 'RECORDS')) if x.strip()]:
    s, a, ch = load(r); f = first(s, a, {'F'})
    if f: sel_f = (r, s, a, f); break
fr, sF, aF, (f_i, f_t) = sel_f
print('F beat from record', fr, 'sample', f_t)
fig = plt.figure(figsize=(10, 6.2)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.5, wspace=0.28)
ax = fig.add_subplot(gs[0, :]); N = 3600
ax.plot(np.arange(N), s100[:N], 'k', lw=0.8, label='denoised, z-scored MLII (record 100)')
pk = [(int(t), c) for t, c in zip(a100.sample, a100.symbol) if t < N and c in CS]
ax.plot([t for t, _ in pk], [s100[t] for t, _ in pk], 'o', color='#b2182b', ms=4, label='annotated beats (R-peak annotation)')
ax.set_xlabel('sample index (360 Hz)'); ax.set_ylabel('amplitude (z-score)'); ax.set_ylim(-2.5, 9.5); ax.legend(loc='upper center', fontsize=8, ncol=2); ax.set_title('(A) Record 100: processed signal with beat annotations (first 10 s)', fontsize=10)
for k, (name, sig, t, col, rec) in enumerate([('N (normal)', s100, n_t, '#2c6fbb', '100'), ('V (premature ventricular)', s100, v_t, '#d98314', '100'), ('F (fusion)', sF, f_t, '#7a3fc0', fr)]):
    ax = fig.add_subplot(gs[1, k]); x = np.arange(-PRE, POST); ax.plot(x, sig[t - PRE:t + POST], color=col, lw=1.3); ax.axvline(0, color='0.4', ls='--', lw=0.9)
    ax.set_title(f'({"BCD"[k]}) {name}, record {rec}', fontsize=9.5); ax.set_xlabel('samples from R peak'); ax.set_xlim(-PRE, POST - 1)
    if k == 0: ax.set_ylabel('amplitude (z-score)')
os.makedirs(OUT, exist_ok=True); fig.savefig(os.path.join(OUT, 'Figure_1_beat_extraction.png'), dpi=220, bbox_inches='tight')


