"""Architecture figures for the EVALUATED PyTorch models A, B, C (186 x 1 input).
Everything numeric in the figures is taken from the live model objects: shapes come from forward hooks on the real modules
(experiments/r2common.py; ECGTransForm models.py imported read-only), parameter counts from nparams().
Block text (kernel sizes, channels, pooling) is checked against the module attributes by assertions below, so a mismatch aborts the build.
No training or scoring is performed; weights are randomly initialised for shape tracing only."""
import sys, os, json, hashlib, warnings
warnings.filterwarnings('ignore')
import torch
sys.path.insert(0, os.path.join(os.environ.get('R2_ROOT', '.'), 'experiments'))
import r2common as rc
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', 'figures')


def trace(model, L=186):
    io = {}; hs = []
    def mk(n):
        def h(m, i, o):
            t = lambda v: tuple(v.shape) if torch.is_tensor(v) else (tuple(v[0].shape) if isinstance(v, tuple) else None)
            io.setdefault(n, []).append((t(i[0]), t(o)))
        return h
    for n, m in model.named_modules(): hs.append(m.register_forward_hook(mk(n)))
    model.eval()
    with torch.no_grad(): y = model(torch.zeros(1, 1, L))
    for h in hs: h.remove()
    return io, tuple(y.shape)


sh = lambda t: f'{t[1]} x {t[2]}'
mods = {m: rc.build(m) for m in 'ABC'}; tr = {m: trace(mods[m]) for m in 'ABC'}; npar = {m: rc.nparams(mods[m]) for m in 'ABC'}
assert npar == {'A': 363653, 'B': 451361, 'C': 637689}, npar
A, B, C = mods['A'], mods['B'], mods['C']; iA, iB, iC = tr['A'][0], tr['B'][0], tr['C'][0]
for m in 'ABC': assert tr[m][1] == (1, 5)

COL = dict(inp='#dbe8fb', conv='#d9f2e3', att='#fdeeb9', rec='#e6d9fb', dense='#fbd5da', trans='#d3e3f0', pool='#eeeeee')


def draw(blocks, title, fname, per_row, par, note=''):
    rows = [blocks[i:i + per_row] for i in range(0, len(blocks), per_row)]
    W, H, GX, GY = 1.0, 0.92, 0.36, 0.48
    sc = 6.6 / (per_row * (W + GX))                                   # inches per data unit (figure width 6.6 in)
    fig, ax = plt.subplots(figsize=(6.6, len(rows) * (H + GY) * sc + 0.55)); ax.axis('off')
    ax.set_xlim(-0.05, per_row * (W + GX) - GX + 0.05); ax.set_ylim(-(len(rows) - 1) * (H + GY) - H - 0.3, 0.08)
    pos = []
    for r, row in enumerate(rows):
        y0 = -r * (H + GY)
        for c, (kind, name, detail, shape) in enumerate(row):
            x0 = c * (W + GX) if r % 2 == 0 else (per_row - 1 - c) * (W + GX)     # snake layout
            ax.add_patch(FancyBboxPatch((x0, y0 - H), W, H, boxstyle='round,pad=0.015,rounding_size=0.06', fc=COL[kind], ec='#333', lw=1.0))
            ax.text(x0 + W / 2, y0 - 0.2, name, ha='center', va='center', fontsize=(6.0 if per_row >= 4 else 7.4), fontweight='bold', linespacing=1.15)
            ax.text(x0 + W / 2, y0 - 0.54, detail, ha='center', va='center', fontsize=(5.8 if per_row >= 4 else 6.6), linespacing=1.15)
            ax.text(x0 + W / 2, y0 - H + 0.12, shape, ha='center', va='center', fontsize=7.4, color='#1a3a6b', fontweight='bold')
            pos.append((x0, y0))
    for k in range(len(pos) - 1):
        (xa, ya), (xb, yb) = pos[k], pos[k + 1]
        if ya == yb:
            s_, e_ = ((xa + W, ya - H / 2), (xb, yb - H / 2)) if xb > xa else ((xa, ya - H / 2), (xb + W, yb - H / 2))
        else:
            s_, e_ = (xa + W / 2, ya - H), (xb + W / 2, yb)
        ax.add_patch(FancyArrowPatch(s_, e_, arrowstyle='-|>', mutation_scale=10, lw=1.1, color='#444'))
    ax.set_title(title, fontsize=8.6, fontweight='bold')
    ax.text((per_row * (W + GX) - GX) / 2, -(len(rows) - 1) * (H + GY) - H - 0.1, f'Trainable parameters: {par:,} (counted from the model object).  ' + note, ha='center', va='top', fontsize=6.8)
    fig.savefig(os.path.join(OUT, fname), dpi=300, bbox_inches='tight'); plt.close(fig)


# ---------------- Model A ----------------
assert (A.c1.in_channels, A.c1.out_channels, A.c1.kernel_size[0]) == (1, 32, 5) and (A.c2.out_channels, A.c2.kernel_size[0]) == (64, 5) and (A.c3.out_channels, A.c3.kernel_size[0]) == (128, 3)
assert (A.f1.in_features, A.f1.out_features, A.f2.out_features) == (2560, 128, 5)
pooled = lambda out: (1, out[1], out[2] // 2)          # F.max_pool1d(x, 2) in ModelA.forward
assert pooled(iA['c3'][0][1])[2] == 20 and 128 * 20 == iA['f1'][0][0][1]
bA = [('inp', 'Input', 'z-scored beat', f'1 x {iA["c1"][0][0][2]}'),
      ('conv', 'Conv1d(k=5) + ReLU\nMaxPool(2)', '32 channels', sh(pooled(iA['c1'][0][1]))),
      ('conv', 'Conv1d(k=5) + ReLU\nMaxPool(2)', '64 channels', sh(pooled(iA['c2'][0][1]))),
      ('conv', 'Conv1d(k=3) + ReLU\nMaxPool(2)', '128 channels', sh(pooled(iA['c3'][0][1]))),
      ('pool', 'Flatten', '128 x 20', str(iA['f1'][0][0][1])),
      ('dense', 'Dense + ReLU', '128 units', str(iA['f1'][0][1][1])),
      ('dense', 'Dense(5)', 'logits; softmax', str(iA['f2'][0][1][1]))]
draw(bA, 'Model A: 1-D CNN (evaluated PyTorch model)', 'Figure_2_ModelA_architecture.png', 3, npar['A'], 'Shapes: channels x time.')

# ---------------- Model B ----------------
assert [c.kernel_size[0] for c in B.convs] == [21, 23, 25, 27] and [c.out_channels for c in B.convs] == [16, 32, 64, 128]
assert B.ca[0].a.out_features == 2 and B.ca[3].a.out_features == 16 and B.sa.c.kernel_size[0] == 7 and B.l1.hidden_size == 64 and B.l2.hidden_size == 32
assert (B.f1.in_features, B.f1.out_features, B.f2.out_features, B.d.p) == (768, 128, 5, 0.2)
assert iB['l2'][0][1][1] == 24 and iB['sa.c'][0][1][2] == 24
bB = [('inp', 'Input', 'z-scored beat', f'1 x {iB["convs.0"][0][0][2]}'),
      ('conv', 'Conv k=21 + ReLU\nChannel att.\nMaxPool(3, s2)', '16 channels', f'16 x {iB["convs.1"][0][0][2]}'),
      ('conv', 'Conv k=23 + ReLU\nChannel att.\nMaxPool(3, s2)', '32 channels', f'32 x {iB["convs.2"][0][0][2]}'),
      ('conv', 'Conv k=25 + ReLU\nChannel att.\nAvgPool(3, s2)', '64 channels', f'64 x {iB["convs.3"][0][0][2]}'),
      ('conv', 'Conv k=27 + ReLU\nChannel att.', '128 ch., no pooling', f'128 x {iB["convs.3"][0][1][2]}'),
      ('att', 'Spatial attention', 'Conv k=7 on channel\nmean/max, sigmoid', f'128 x {iB["sa.c"][0][1][2]}'),
      ('rec', 'LSTM(64)\nDropout 0.2', 'time x units', f'{iB["l1"][0][1][1]} x 64'),
      ('rec', 'LSTM(32)', 'time x units', f'{iB["l2"][0][1][1]} x 32'),
      ('pool', 'Flatten', '24 x 32', str(iB['f1'][0][0][1])),
      ('dense', 'Dense(128) + ReLU\nDropout 0.2', '', str(iB['f1'][0][1][1])),
      ('dense', 'Dense(5)', 'logits; softmax', str(iB['f2'][0][1][1]))]
draw(bB, 'Model B: CNN-LSTM with channel and spatial attention (evaluated PyTorch port)', 'Figure_3_ModelB_architecture.png', 4, npar['B'],
     'Shapes: channels x time, except LSTM outputs (time x units).')

# ---------------- Model C ----------------
assert [c.kernel_size[0] for c in (C.conv1, C.conv2, C.conv3)] == [5, 9, 11] and C.conv1.out_channels == 32
assert (C.conv_block2[0].kernel_size[0], C.conv_block2[0].out_channels, C.conv_block3[0].out_channels) == (8, 64, 128)
assert len(C.crm) == 3 and len(C.transformer_encoder.layers) == 3 and C.encoder_layer.self_attn.num_heads == 5 and C.encoder_layer.self_attn.embed_dim == 25
assert iC['transformer_encoder'][0][0] == (1, 128, 25) and tr['C'][1] == (1, 5)
ffn = C.encoder_layer.linear1.out_features
bC = [('inp', 'Input', 'z-scored beat', f'1 x {iC["conv1"][0][0][2]}'),
      ('conv', 'Three parallel Conv1d\nk = 5, 9, 11 (32 ch each)', 'branch outputs averaged', f'32 x {iC["conv1"][0][1][2]}'),
      ('conv', 'BatchNorm + ReLU\nMaxPool(2) + Dropout 0.2', '', f'32 x {iC["do"][0][1][2]}'),
      ('conv', 'Conv1d(k=8) + BN + ReLU\nMaxPool(2)', '64 channels', f'64 x {iC["conv_block2.3"][0][1][2]}'),
      ('conv', 'Conv1d(k=8) + BN + ReLU\nMaxPool(2)', '128 channels', f'128 x {iC["conv_block3.3"][0][1][2]}'),
      ('att', '3 x SE-residual block', '1x1 convs, BN,\nSE (reduction 4)', f'128 x {iC["crm.2"][0][1][2]}'),
      ('trans', 'Transformer encoder\n3 layers, 5 heads', f'd_model 25, FFN {ffn};\nrun on the map and on its\nreversal along the 25-axis,\noutputs summed', f'128 tokens x {iC["transformer_encoder"][0][1][2]}'),
      ('pool', 'Adaptive average pool', 'over the 25-dim axis', str(iC['aap'][0][1][1])),
      ('dense', 'Linear', '5 classes\n(logits; softmax)', str(iC['clf'][0][1][1]))]
draw(bC, 'Model C: ECGTransForm-style Conv-Transformer (evaluated PyTorch model)', 'Figure_4_ModelC_architecture.png', 3, npar['C'],
     'Shapes: channels x time; transformer tokens are the 128 channels.')
mp = os.environ.get('ECGTF_MODELS_PY', os.path.join('training', 'ecgtransform', 'models.py'))
json.dump(dict(parameters_trainable=npar, parameters_total={m: sum(p.numel() for p in mods[m].parameters()) for m in 'ABC'}, input_shape=[1, 186], output_shape=[1, 5],
               r2common_sha256=hashlib.sha256(open(os.path.join(os.environ.get('R2_ROOT', '.'), 'experiments', 'r2common.py'), 'rb').read()).hexdigest(),
               ecgtransform_models_py_sha256=hashlib.sha256(open(mp, 'rb').read()).hexdigest(),
               note='ECGTransForm models.py is imported read-only from the upstream code folder; its hash is pinned in the locked-protocol manifest (protocol_lock.frozen_files) and the hash recorded here is its current hash.'),
          open(os.path.join(HERE, 'architecture_verification.json'), 'w'), indent=1)
print('ok', npar)

