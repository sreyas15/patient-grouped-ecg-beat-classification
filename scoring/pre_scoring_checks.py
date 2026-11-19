"""R2 operational pre-checks (READ-ONLY; writes nothing). Approved safeguards, to be run immediately before 07_score.py and before 08_evaluate.py.
  python -P -B experiments/pre_scoring_checks.py pre07     # before scoring
  python -P -B experiments/pre_scoring_checks.py pre08     # SUPERSEDED by addendum A1 (08a/08b run their own pre-checks); do not use for the separated flow
Exit code 0 only if every check passes. This script never starts scoring/evaluation and never modifies any file."""
import os, sys, json, stat, hashlib, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); R2 = os.path.dirname(HERE); sys.path.insert(0, HERE)
stage = sys.argv[1] if len(sys.argv) > 1 else ''
if stage not in ('pre07', 'pre08'): sys.exit('usage: pre_scoring_checks.py pre07|pre08')
res = []
def chk(name, ok, detail=''): res.append(bool(ok)); print(('PASS ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail else ''))
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
ps = lambda c: subprocess.run(['powershell', '-NoProfile', '-Command', c], capture_output=True, text=True).stdout.strip()

import a1common as a1                       # addendum A1: scoped lock check (12 of 13 frozen files; the external-dataset file is deliberately not read) + access guard
a1.install_no_external_guard()
try: lock = a1.verify_lock(include_external=False); chk('protocol lock verifies (12 frozen files hashed + lock self-hash; external-dataset file not read)', True, lock['lock_file_sha256'][:16] + '...')
except RuntimeError as e: chk('protocol lock verifies', False, e); sys.exit(1)

man = json.load(open(os.path.join(R2, 'results', 'manifests', 'manifest_post_training.json')))
chk('post-training manifest lock hash == current lock', man['protocol_lock']['lock_file_sha256'] == lock['lock_file_sha256'])
want = {f"results/runs/{r['run']}/{n}": v['sha256'] for r in man['completed_runs']['runs'] for n, v in r['files'].items()}
bad = [k for k, h in want.items() if not os.path.exists(os.path.join(R2, k)) or sha(os.path.join(R2, k)) != h]
chk(f'all {len(want)} run files (incl. 45 checkpoints, 15 selected.json, 15 val_logits) match manifest_post_training.json', not bad, bad[:3])
selbad = []
for r in man['completed_runs']['runs']:
    sj = json.load(open(os.path.join(R2, 'results', 'runs', r['run'], 'selected.json')))
    if sj['checkpoint'] != r['selection']['checkpoint'] or sha(os.path.join(R2, 'results', 'runs', r['run'], sj['checkpoint'])) != r['files'][sj['checkpoint']]['sha256']: selbad.append(r['run'])
chk('15 selected checkpoints: file named in selected.json == manifest selection and hash matches', not selbad, selbad)
pm = os.path.join(R2, 'results', 'manifests', 'manifest_pre_resume.json'); chk('pre-resume manifest unchanged (sha256)', sha(pm) == '6d564a8cb46fb89c99ad76e2781acd457bb7f63353346ab75311a62d0ca3f563')

# backup
B = os.path.join(R2, 'backup_pre_scoring'); idx = os.path.join(B, 'BACKUP_SHA256.txt'); bb = []; ro = []; n = 0
if os.path.exists(idx):
    for ln in open(idx).read().splitlines():
        if ln.startswith('#') or not ln.strip(): continue
        h, rel = ln.split('  ', 1); n += 1; p = os.path.join(B, rel.replace('/', '\\')); src = os.path.join(R2, rel.replace('/', '\\'))
        if not os.path.exists(p) or sha(p) != h or not os.path.exists(src) or sha(src) != h: bb.append(rel)
        if os.path.exists(p) and os.stat(p).st_mode & stat.S_IWRITE: ro.append(rel)
chk('backup exists; every file hash == backup index == live source file', os.path.exists(idx) and n == 137 and not bb, f'{n} files; mismatches {bb[:3]}')
chk('backup files are read-only', os.path.exists(idx) and not ro, ro[:3])

SC = os.path.join(R2, 'results', 'scores'); EV = os.path.join(R2, 'results', 'evaluation')
if stage == 'pre07':
    chk('results/scores/ absent (no unexpected or partial outputs)', not os.path.exists(SC))
    chk('results/evaluation/ empty', os.listdir(EV) == [] if os.path.isdir(EV) else True)
    chk('no scoring/evaluation logs exist', not any(os.path.exists(os.path.join(R2, 'logs', f)) for f in ('07_score.log', '08_evaluate.log', '09_posthoc.log', 'scoring_run.log')))
    chk('no other python process running', ps(f"@(Get-Process python -ErrorAction SilentlyContinue | Where-Object {{ $_.Id -ne {os.getpid()} }}).Count") == '0')
    free =float(ps("[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)")); chk('free RAM >= 10 GB', free >= 10, f'{free} GB')
    chk('on AC power', ps("(Get-CimInstance Win32_Battery).BatteryStatus") == '2')
    print('NOTE: sleep/hibernate prevention and foreground logging cannot be verified from here; start the keep-awake wrapper before 07.')
else:
    exp_files = [f'mit_{m}_fold{k}.npz' for m in 'ABC' for k in range(5)] + [f'incart_{m}_fold{k}.npz' for m in 'ABC' for k in range(5)] + ['incart_labels.npz', 'SCORED.json']
    have = sorted(os.listdir(SC)) if os.path.isdir(SC) else []
    chk('results/scores/ holds exactly the 32 expected files (no partial/unexpected)', have == sorted(exp_files), f'missing {sorted(set(exp_files) - set(have))[:3]} extra {sorted(set(have) - set(exp_files))[:3]}')
    if 'SCORED.json' in have:
        fl = json.load(open(os.path.join(SC, 'SCORED.json'))); chk('SCORED.json lock hash == current lock', fl['lock_sha256'] == lock['lock_file_sha256'])
        mm = {r['run']: r['selection']['checkpoint'] for r in man['completed_runs']['runs']}; sh = {r['run']: r['files'][r['selection']['checkpoint']]['sha256'] for r in man['completed_runs']['runs']}
        chk('SCORED.json: 15 entries, checkpoint names and SHA-256 == manifest', len(fl['runs']) == 15 and all(e['checkpoint'] == mm[f"{e['model']}_fold{e['fold']}"] and e['checkpoint_sha256'] == sh[f"{e['model']}_fold{e['fold']}"] for e in fl['runs']))
        sizes = [22821, 21988, 22367, 21564, 20565]; chk('SCORED.json n_test per fold == locked fold sizes', all(e['n_test'] == sizes[e['fold']] for e in fl['runs']))
    chk('results/evaluation/ empty (nothing overwritten)', os.listdir(EV) == [] if os.path.isdir(EV) else True)
    chk('no other python process running', ps(f"@(Get-Process python -ErrorAction SilentlyContinue | Where-Object {{ $_.Id -ne {os.getpid()} }}).Count") == '0')
print('\nALL CHECKS PASSED' if all(res) else '\nCHECKS FAILED: do not proceed'); sys.exit(0 if all(res) else 1)
