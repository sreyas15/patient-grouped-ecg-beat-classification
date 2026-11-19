"""R2 protocol addendum A1: shared helpers for the separated scoring / evaluation drivers (07a, 07b, 08a, 08b).
Definitions only: importing this module has no side effects (it opens no data file and installs no hook). Every driver verifies the hash of this
file (through the addendum) at the start of run(), before touching any data."""
import os, sys, json, hashlib, datetime, stat, ast, importlib.util, subprocess

R2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT_REL = 'data/incart_beats_r2.npz'          # the external-dataset file inside the frozen-file list (string comparison only)
ADDENDUM_JSON, ADDENDUM_SHA = 'protocol_addendum_A1.json', 'protocol_addendum_A1.sha256'
_LOGF = [None]

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def now(): return datetime.datetime.now().isoformat(timespec='seconds')
def log(msg):
    line = f'{now()} {msg}'; print(line, flush=True)
    if _LOGF[0] is not None: _LOGF[0].write(line + '\n'); _LOGF[0].flush()
def start_log(log_dir, name):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{name}_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}.log")
    _LOGF[0] = open(path, 'x', encoding='utf-8'); log(f'log file: {path}'); return path
def keep_awake():
    """Hold ES_SYSTEM_REQUIRED for the life of the calling thread/process (released automatically on exit). No power setting is changed."""
    try:
        import ctypes; return bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    except Exception: return False
def write_json_x(path, obj):
    with open(path, 'x') as f: json.dump(obj, f, indent=1, default=str)
def savez_x(path, **arrays):
    import numpy as np
    with open(path, 'xb') as f: np.savez_compressed(f, **arrays)

# ---------------- integrity verification ----------------
def verify_lock(include_external, root=R2):
    """Same logic as the frozen verify_lock.verify(), except that the external-dataset file can be left out of the hashing
    (include_external=False) so that the MIT-BIH stage never reads a single byte of it."""
    lp = os.path.join(root, 'locked_protocol.json')
    if not os.path.exists(lp): raise RuntimeError('protocol not locked (locked_protocol.json missing)')
    if sha256_file(lp) != open(os.path.join(root, 'locked_protocol.sha256')).read().split()[0]: raise RuntimeError('locked_protocol.json itself was modified')
    lock = json.load(open(lp)); bad = []; skipped = []
    for rel, h in lock['frozen_files'].items():
        if (not include_external) and rel.replace('\\', '/') == EXT_REL: skipped.append(rel); continue
        p = rel if os.path.isabs(rel) else os.path.join(root, rel)
        if not os.path.exists(p) or sha256_file(p) != h: bad.append(rel)
    if bad: raise RuntimeError(f'locked files changed or missing: {bad}')
    lock['lock_file_sha256'] = sha256_file(lp); lock['_not_hashed_in_this_stage'] = skipped; return lock
def verify_addendum(root=R2):
    ap, hp = os.path.join(root, ADDENDUM_JSON), os.path.join(root, ADDENDUM_SHA)
    if not (os.path.exists(ap) and os.path.exists(hp)): raise RuntimeError('protocol addendum A1 (json/sha256) missing')
    if sha256_file(ap) != open(hp).read().split()[0]: raise RuntimeError('addendum json does not match its .sha256')
    add = json.load(open(ap))
    if add['original_lock_sha256'] != sha256_file(os.path.join(root, 'locked_protocol.json')): raise RuntimeError('addendum refers to a different protocol lock')
    bad = [rel for rel, h in {**add['files'], **add['frozen_scripts_unchanged']}.items() if (not os.path.exists(os.path.join(root, rel))) or sha256_file(os.path.join(root, rel)) != h]
    if bad: raise RuntimeError(f'addendum-listed files changed: {bad}')
    add['addendum_json_sha256'] = sha256_file(ap); return add
def verify_manifest_runs(root=R2, manifest_path=None):
    """Hash every run file against manifest_post_training.json BEFORE any checkpoint is loaded. Returns per-run selection info."""
    mp = manifest_path or os.path.join(root, 'results', 'manifests', 'manifest_post_training.json'); man = json.load(open(mp)); info = {}; bad = []
    if man['protocol_lock']['lock_file_sha256'] != sha256_file(os.path.join(root, 'locked_protocol.json')): bad.append('manifest lock hash != current lock')
    for r in man['completed_runs']['runs']:
        d = os.path.join(root, 'results', 'runs', r['run'])
        for n, v in r['files'].items():
            p = os.path.join(d, n)
            if not os.path.exists(p) or sha256_file(p) != v['sha256']: bad.append(f"{r['run']}/{n}")
        sel = json.load(open(os.path.join(d, 'selected.json')))
        if sel['checkpoint'] != r['selection']['checkpoint']: bad.append(f"{r['run']}: selected checkpoint name")
        info[r['run']] = dict(checkpoint=sel['checkpoint'], checkpoint_sha256=r['files'][sel['checkpoint']]['sha256'], T=sel['temperature'])
    if len(info) != 15: bad.append('manifest does not list 15 runs')
    if bad: raise RuntimeError(f'run files differ from manifest_post_training.json: {bad[:5]}')
    return info, sha256_file(mp)
def verify_backup(root=R2):
    B = os.path.join(root, 'backup_pre_scoring'); idx = os.path.join(B, 'BACKUP_SHA256.txt')
    if not os.path.exists(idx): raise RuntimeError('backup index missing');
    n = 0; bad = []
    for ln in open(idx).read().splitlines():
        if ln.startswith('#') or not ln.strip(): continue
        h, rel = ln.split('  ', 1); n += 1; p = os.path.join(B, rel.replace('/', '\\')); s = os.path.join(root, rel.replace('/', '\\'))
        if not (os.path.exists(p) and sha256_file(p) == h and os.path.exists(s) and sha256_file(s) == h) or os.stat(p).st_mode & stat.S_IWRITE: bad.append(rel)
    if n != 137 or bad: raise RuntimeError(f'backup verification failed ({n} files): {bad[:3]}')
    return n
def snapshot_dir(d):
    return {n: sha256_file(os.path.join(d, n)) for n in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, n))}

# ---------------- process-level access guards (audit hooks cannot be removed once installed) ----------------
def install_no_external_guard():
    """Any attempt to open / list / scan a path containing 'incart' raises PermissionError BEFORE the operation happens."""
    # The only exception: READ-ONLY access to the two driver SOURCE files, so that the addendum can verify their hashes. No data, directory, score or evaluation path is exempt.
    ok_tail = tuple(os.sep + os.path.normcase(os.path.join('experiments', n)) for n in ('07b_score_incart.py', '08b_evaluate_incart.py'))
    def hook(event, args):
        if event in ('open', 'os.listdir', 'os.scandir') and args:
            p = args[0]
            if isinstance(p, (str, bytes, os.PathLike)):
                try: s = os.fsdecode(p).lower()
                except Exception: return
                if 'incart' in s:
                    mode = args[1] if len(args) > 1 else None
                    if event == 'open' and isinstance(mode, str) and not any(c in mode for c in 'wax+') and os.path.normcase(os.path.normpath(s)).endswith(ok_tail): return
                    raise PermissionError(f'A1 guard: access to an external-dataset path is forbidden in this stage: {s}')
    sys.addaudithook(hook)
def install_no_write_guard(protected_dirs):
    """Any write / create / rename / delete under a protected directory raises PermissionError (reads stay allowed)."""
    prot = [os.path.normcase(os.path.abspath(d)) for d in protected_dirs]
    def under(p):
        try: s = os.path.normcase(os.path.abspath(os.fsdecode(p)))
        except Exception: return False
        return any(s == q or s.startswith(q + os.sep) for q in prot)
    wbits = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
    def hook(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            mode = args[1] if len(args) > 1 else None; flags = args[2] if len(args) > 2 else 0
            if ((isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & wbits)) and under(args[0]):
                raise PermissionError(f'A1 guard: write under a protected directory is forbidden: {args[0]}')
        elif event in ('os.rename', 'os.replace', 'os.remove', 'os.rmdir', 'os.mkdir', 'shutil.rmtree') and args:
            for a_ in args[:2]:
                if isinstance(a_, (str, bytes, os.PathLike)) and under(a_): raise PermissionError(f'A1 guard: {event} under a protected directory is forbidden: {a_}')
    sys.addaudithook(hook)

# ---------------- reuse of the frozen 08_evaluate.py ----------------
def load_frozen_eval(root=R2):
    p = os.path.join(root, 'experiments', '08_evaluate.py'); sys.path.insert(0, os.path.join(root, 'experiments'))
    spec = importlib.util.spec_from_file_location('frozen_08_evaluate', p); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def frozen_main_blocks(root=R2):
    """The MIT-BIH block (lines 114-127) and the INCART block (lines 129-139) of the frozen script's __main__ section, as AST statements.
    They are executed verbatim (not re-typed); the file itself is hash-locked."""
    src = open(os.path.join(root, 'experiments', '08_evaluate.py'), encoding='utf-8').read(); tree = ast.parse(src)
    main = [n for n in tree.body if isinstance(n, ast.If) and '__main__' in ast.unparse(n.test)]
    if len(main) != 1: raise RuntimeError('frozen 08_evaluate.py structure changed')
    body = main[0].body; mit = [n for n in body if 114 <= n.lineno <= 127]; inc = [n for n in body if 129 <= n.lineno <= 139]
    if [n.lineno for n in mit] != [114, 115, 125, 126, 127] or [n.lineno for n in inc] != [129, 129, 130]: raise RuntimeError('frozen 08_evaluate.py main block layout differs from the reviewed one')
    if "evaluate('mit_internal'" not in ast.unparse(mit[-1]) or 'incart_labels' not in ast.unparse(inc[0]): raise RuntimeError('frozen blocks are not the expected statements')
    comp = lambda nodes, name: compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec')
    return comp(mit, '<frozen 08_evaluate.py lines 114-127 (MIT-BIH block)>'), comp(inc, '<frozen 08_evaluate.py lines 129-139 (INCART block)>')

def verify_scored_dir(scores_dir, scope, lock_sha, minfo, allow_partition=('test',)):
    """Check a stage's SCORED.json against the files actually present, the lock and the manifest-verified checkpoint hashes."""
    fl = json.load(open(os.path.join(scores_dir, 'SCORED.json')))
    names = [f'{"mit" if scope == "MIT-BIH" else "incart"}_{m}_fold{k}.npz' for m in 'ABC' for k in range(5)] + (['incart_labels.npz'] if scope == 'INCART' else [])
    ok = (fl['scope'] == scope and fl['partition'] in allow_partition and fl['lock_sha256'] == lock_sha and
          sorted(os.listdir(scores_dir)) == sorted(names + ['SCORED.json', '_STARTED.json']) and
          all(sha256_file(os.path.join(scores_dir, n)) == fl['output_files'][n] for n in names) and
          len(fl['runs']) == 15 and all(e['checkpoint_sha256'] == minfo[f"{e['model']}_fold{e['fold']}"]['checkpoint_sha256'] for e in fl['runs']))
    if not ok: raise RuntimeError(f'{scope} scores directory failed verification against its SCORED.json / lock / manifest: {scores_dir}')
    return fl
def env_info():
    import platform, numpy, torch, sklearn, scipy
    smi = lambda q: subprocess.run(['nvidia-smi', f'--query-gpu={q}', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    return dict(python=platform.python_version(), numpy=numpy.__version__, torch=torch.__version__, cuda=torch.version.cuda, sklearn=sklearn.__version__, scipy=scipy.__version__, gpu=smi('name'), driver=smi('driver_version'))
