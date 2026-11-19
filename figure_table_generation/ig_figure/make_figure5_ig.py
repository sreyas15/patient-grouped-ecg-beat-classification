"""Figure 5 (formerly Figure 6): two illustrative MIT-BIH beats, predicted by the LOCKED Model B fold-0 checkpoint (test fold 0: the model never saw these patients).
Explanation method: Integrated Gradients, zero-signal baseline, 50 steps (as in the released demonstration code), applied to the 186-sample model input.
Read-only use of locked files; no training, evaluation or result is modified. Selection rule (fixed before any attribution was computed):
  Example 1: first correctly classified V beat (lowest beat_index) in test fold 0; Example 2: first misclassified F beat in test fold 0."""
import sys, os, json, hashlib
import numpy as np, torch
R2 = os.environ.get('R2_ROOT', '.')
sys.path.insert(0, os.path.join(R2, 'experiments'))
import r2common as rc
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = os.path.dirname(os.path.abspath(__file__)); torch.set_grad_enabled(True)
CLS = ['N', 'S', 'V', 'F', 'Q']; K = 0
run = os.path.join(R2, 'results', 'runs', f'B_fold{K}'); sel = json.load(open(os.path.join(run, 'selected.json')))
ck = os.path.join(run, sel['checkpoint']); sha = hashlib.sha256(open(ck, 'rb').read()).hexdigest()
sc = np.load(os.path.join(R2, 'results', 'scores_mit', f'mit_B_fold{K}.npz')); d = np.load(os.path.join(R2, 'data', 'mitbih_beats_r2.npz'))
X, rec = d['X'], d['record']; lg, y, bi = sc['logits'], sc['y'], sc['beat_index']; pred = lg.argmax(1)
model = rc.build('B'); model.load_state_dict(torch.load(ck, map_location='cpu')); model.eval()
def pick():
    a = np.where((y == 2) & (pred == 2))[0]; b = np.where((y == 3) & (pred != 3))[0]
    return [a[np.argmin(bi[a])], b[np.argmin(bi[b])]]
ex = pick(); prov = dict(model='B', fold=K, checkpoint=sel['checkpoint'], checkpoint_sha256=sha, selected_epoch=sel['selected_epoch'], examples=[])
def ig(x, c, steps=50):
    base = torch.zeros_like(x); al = torch.linspace(0, 1, steps + 1).view(-1, 1, 1)
    xs = (base + al * (x - base)).requires_grad_(True)
    p = torch.softmax(model(xs), 1)[:, c].sum(); g, = torch.autograd.grad(p, xs)
    avg = g.mean(0)
    return ((x - base) * avg).squeeze().detach().numpy()
fig, axs = plt.subplots(3, 2, figsize=(10, 7.6), gridspec_kw=dict(height_ratios=[1.3, 0.8, 1.1]))
t = (np.arange(186) - 61.4) * 833 / 186          # ms relative to the R peak (R at raw sample 99 of 300 -> 61.4 of 186)
for col, i in enumerate(ex):
    x = torch.from_numpy(X[bi[i]]).view(1, 1, 186); out = model(x).detach()
    assert np.allclose(out.numpy()[0], lg[i], atol=5e-3), (out.numpy()[0], lg[i])      # recomputed (CPU) logits agree with the saved locked (GPU) scores within float tolerance
    pr = torch.softmax(out, 1).numpy()[0]; c = int(pr.argmax()); a = ig(x, c)
    comp = float(a.sum()); full = float(torch.softmax(model(x), 1)[0, c] - torch.softmax(model(torch.zeros_like(x)), 1)[0, c])
    prov['examples'].append(dict(max_abs_logit_diff_vs_saved=float(np.abs(out.numpy()[0]-lg[i]).max()), beat_index=int(bi[i]), record=str(rec[bi[i]]), true_class=CLS[y[i]], predicted_class=CLS[c], softmax=[float(v) for v in pr], ig_sum=comp, prob_difference_vs_zero_baseline=full))
    ax = axs[0, col]; ax.plot(t, X[bi[i]], 'k', lw=1.2); ax.axvline(0, color='0.7', ls=':', lw=1); ax.set_ylabel('z-scored amplitude (a.u.)')
    ax.set_title(f"({'AB'[col]}) Record {rec[bi[i]]}, annotated {CLS[y[i]]}, predicted {CLS[c]}", fontsize=10)
    ax = axs[1, col]; ax.bar(CLS, pr, color=['#33506b' if k == c else '#9db4c8' for k in range(5)]); ax.set_ylim(0, 1); ax.set_ylabel('softmax probability\n(uncalibrated)')
    for k, v in enumerate(pr): ax.text(k, min(v + 0.02, 0.86), f'{v:.2f}', ha='center', fontsize=8)
    ax = axs[2, col]; m = np.abs(a).max() + 1e-12; ax.fill_between(t, a, 0, where=a >= 0, color='#b2182b', alpha=.8, label='raises'); ax.fill_between(t, a, 0, where=a < 0, color='#2166ac', alpha=.8, label='lowers')
    ax.axvline(0, color='0.7', ls=':', lw=1); ax.set_ylabel(f'IG attribution for class {CLS[c]}\n(probability units)'); ax.set_xlabel('time from annotated R peak (ms)'); ax.legend(fontsize=7, loc='upper right')
fig.suptitle('Illustrative Model B (fold 0) outputs and Integrated Gradients (zero baseline); experimental, not diagnostic', fontsize=10.5, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(OUT, 'Figure_5_IG_examples.png'), dpi=220); json.dump(prov, open(os.path.join(OUT, 'figure6_provenance.json'), 'w'), indent=1)
print(json.dumps(prov, indent=1))

