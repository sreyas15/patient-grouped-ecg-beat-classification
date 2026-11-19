"""R2 shared code: data access, locked fold partitions, SMOTE (training partition only), models A/B/C, training, temperature.
Frozen by hash in locked_protocol.json; any change makes 07_score.py refuse to run."""
import os, sys, json, hashlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__)); R2 = os.path.dirname(HERE)
ECGTF = r"D:\explainable_ai\Journel paper project\ECGTransForm"   # upstream ECGTransForm code, imported read-only
CLS = ['N', 'S', 'V', 'F', 'Q']; K = 5

CONFIG = dict(
    input_len=186, smote_target=20000, smote_k=5, loss='unweighted cross-entropy', optimizer='Adam', batch_size=128, weight_decay=1e-4,
    lr_grid=[1e-4, 5e-4, 1e-3], epochs={'A': 15, 'B': 20, 'C': 15}, fold_seed_base=1000,
    checkpoint_rule='max validation macro-F1 over classes with >=10 validation beats (per epoch); lr chosen by the same criterion',
    temperature='single scalar T, NLL on the selected checkpoint validation logits (LBFGS on log T)',
    primary_model='B', comparators=['A', 'C'], primary_model_reason='Model B is the backbone of the manuscript clinical pipeline (pre-specified before any R2 result)',
    scheme='outer fold k: test=fold k, val=fold (k+1)%5, train=other 3 folds')

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

# ---------------- data and partitions ----------------
def load_mit():
    d = np.load(os.path.join(R2, 'data', 'mitbih_beats_r2.npz')); return {k: d[k] for k in d.files}
def load_incart():
    d = np.load(os.path.join(R2, 'data', 'incart_beats_r2.npz')); return {k: d[k] for k in d.files}
def fold_of_subject():
    f = json.load(open(os.path.join(R2, 'results', 'splits', 'folds_LOCKED.json')))
    m = {}
    for k, recs in f['folds'].items():
        for r in recs: m['201_202' if r == '201+202' else r] = int(k)
    return m
def partition(subject, k):
    fs = fold_of_subject(); fold = np.array([fs[s] for s in subject]); v = (k + 1) % K
    tr, va, te = ~np.isin(fold, [k, v]), fold == v, fold == k
    assert not (set(subject[tr]) & set(subject[va]) or set(subject[tr]) & set(subject[te]) or set(subject[va]) & set(subject[te]))
    return tr, va, te

def smote_balance(X, y, seed, target, k=5):
    """Training partition only. Classes above target are randomly subsampled; classes below are SMOTE-oversampled to target."""
    rng = np.random.RandomState(seed); keep = []
    for c in range(5):
        idx = np.where(y == c)[0]
        keep.append(rng.choice(idx, target, replace=False) if len(idx) > target else idx)
    keep = np.concatenate(keep); Xk, yk = X[keep], y[keep]
    up = {c: target for c in range(5) if (yk == c).sum() < target}
    Xs, ys = SMOTE(sampling_strategy=up, k_neighbors=k, random_state=seed).fit_resample(Xk, yk) if up else (Xk, yk)
    p = rng.permutation(len(ys)); return Xs[p].astype(np.float32), ys[p].astype(np.int64)

# ---------------- models ----------------
class ModelA(nn.Module):   # arrhythmia_cnn.py architecture (fc 128*20 matches 186/187-sample input)
    def __init__(s):
        super().__init__(); s.c1 = nn.Conv1d(1, 32, 5); s.c2 = nn.Conv1d(32, 64, 5); s.c3 = nn.Conv1d(64, 128, 3)
        s.f1 = nn.Linear(128 * 20, 128); s.f2 = nn.Linear(128, 5)
    def forward(s, x):
        for c in (s.c1, s.c2, s.c3): x = F.max_pool1d(F.relu(c(x)), 2)
        return s.f2(F.relu(s.f1(x.flatten(1))))
class ChAtt(nn.Module):
    def __init__(s, f, r=8): super().__init__(); s.a = nn.Linear(f, f // r); s.b = nn.Linear(f // r, f)
    def forward(s, x):
        m = lambda v: s.b(F.relu(s.a(v)))
        return x * torch.sigmoid(m(x.mean(2)) + m(x.amax(2))).unsqueeze(2)
class SpAtt(nn.Module):
    def __init__(s, k=7): super().__init__(); s.c = nn.Conv1d(2, 1, k, padding=k // 2, bias=False)
    def forward(s, x): return x * torch.sigmoid(s.c(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
class ModelB(nn.Module):   # PyTorch port of the Keras CNN-LSTM-Attention in run_local.py buildModel() (same as revision/exp.py; flatten size from input length)
    def __init__(s, L=186):
        super().__init__(); ch = [1, 16, 32, 64, 128]; ks = [21, 23, 25, 27]
        s.convs = nn.ModuleList([nn.Conv1d(ch[i], ch[i + 1], ks[i], padding=ks[i] // 2) for i in range(4)]); s.ca = nn.ModuleList([ChAtt(c) for c in ch[1:]])
        s.sa = SpAtt(7); s.l1 = nn.LSTM(128, 64, batch_first=True); s.l2 = nn.LSTM(64, 32, batch_first=True); s.d = nn.Dropout(.2)
        T = L
        for _ in range(3): T = (T + 2 - 3) // 2 + 1
        s.f1 = nn.Linear(T * 32, 128); s.f2 = nn.Linear(128, 5)
        for l in (s.l1, s.l2): l.bias_hh_l0.data.zero_(); l.bias_hh_l0.requires_grad = False   # Keras LSTM has a single bias vector
    def forward(s, x):
        for i in range(4):
            x = s.ca[i](F.relu(s.convs[i](x)))
            if i < 2: x = F.max_pool1d(x, 3, 2, 1)
            elif i == 2: x = F.avg_pool1d(x, 3, 2, 1, count_include_pad=False)
        x = s.sa(x).transpose(1, 2); x, _ = s.l1(x); x = s.d(x); x, _ = s.l2(x)
        return s.f2(s.d(F.relu(s.f1(x.flatten(1)))))
def build(m):
    if m == 'A': return ModelA()
    if m == 'B': return ModelB(CONFIG['input_len'])
    if ECGTF not in sys.path: sys.path.insert(0, ECGTF)
    from models import ecgTransForm; from configs.data_configs import mit; from configs.hparams import supervised
    cfg = mit(); assert cfg.sequence_len == 186 and cfg.trans_dim == 25
    return ecgTransForm(cfg, supervised().train_params)
def nparams(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------- training ----------------
def set_det(seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
@torch.no_grad()
def logits(model, X, dev='cuda', bs=4096):
    model.eval(); return torch.cat([model(torch.from_numpy(X[i:i + bs]).unsqueeze(1).to(dev)).float().cpu() for i in range(0, len(X), bs)])
def val_classes(yva, min_n=10): return [c for c in range(5) if (yva == c).sum() >= min_n]
def train_one(m, Xtr, ytr, Xva, yva, lr, epochs, seed, dev='cuda'):
    set_det(seed); model = build(m).to(dev); vc = val_classes(yva)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=CONFIG['weight_decay'])
    Xg = torch.from_numpy(Xtr).unsqueeze(1).to(dev); yg = torch.from_numpy(ytr).to(dev); n = len(yg)
    g = torch.Generator(device=dev); g.manual_seed(seed); bs = CONFIG['batch_size']; hist = []; best = (-1.0, 0, None)
    for e in range(epochs):
        model.train(); perm = torch.randperm(n, device=dev, generator=g); tl = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; loss = F.cross_entropy(model(Xg[idx]), yg[idx]); opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(idx)
        lv = logits(model, Xva, dev); pred = lv.argmax(1).numpy()
        f = f1_score(yva, pred, labels=vc, average='macro', zero_division=0)
        pcf = f1_score(yva, pred, labels=list(range(5)), average=None, zero_division=0)
        hist.append(dict(epoch=e + 1, train_loss=tl / n, val_macro_f1=float(f), val_f1_per_class=[float(v) for v in pcf],
                         val_nll=float(F.cross_entropy(lv, torch.from_numpy(yva)).item())))
        if f > best[0]: best = (float(f), e + 1, {k_: v.detach().clone().cpu() for k_, v in model.state_dict().items()})
    return best, hist, vc
def fit_T(lv, yv):
    lt = torch.zeros(1, requires_grad=True); opt = torch.optim.LBFGS([lt], lr=.1, max_iter=200); y = torch.from_numpy(yv).long()
    def c():
        opt.zero_grad(); l = F.cross_entropy(lv / lt.exp(), y); l.backward(); return l
    opt.step(c); return float(lt.exp())
