"""R2 protocol addendum A1 -- 07a: MIT-BIH-ONLY one-time test scoring (replaces the MIT-BIH half of the frozen 07_score.py; the frozen file is not modified or run).
Same selected checkpoints, same locked test partitions, same inference function (rc.logits, batch 4096) and the same output schema as 07_score.py
(logits, y, subject, record, beat_index, T per fold) -> results/scores_mit/mit_{A,B,C}_fold{0-4}.npz. No metric is computed here.
ISOLATION: a process-level guard raises PermissionError on ANY attempt to open/list a path containing 'incart'; the scoped lock check never hashes the
INCART data file; nothing INCART-related is imported or called. WRITE-ONCE: outputs go to results/scores_mit_INPROGRESS/ (marker _STARTED.json); only after
all 15 files are re-read and verified (coverage of every beat exactly once, sizes, labels, T, finite) is SCORED.json written (last) and the directory renamed to
results/scores_mit/. If staging or final directory already exists the script refuses. On any failure everything is preserved and the script stops.
Usage (only after explicit approval):  python -P -B experiments/07a_score_mit.py --authorised-mit-test-scoring"""
import os, sys, json, time, gc, argparse, traceback
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import a1common as a1
import numpy as np, torch
import r2common as rc

def score_fold(m, k, d, sel, run_dir, part):
    """Identical expressions to the frozen 07_score.py lines 19-22; `idx` is that script's `te` (part='test'). part='val' exists only for the validation-only test harness."""
    model = rc.build(m).to('cuda'); model.load_state_dict(torch.load(os.path.join(run_dir, sel['checkpoint']))); model.eval()
    _, va, te = rc.partition(d['subject'], k); idx = te if part == 'test' else va
    rec = dict(logits=rc.logits(model, d['X'][idx]).numpy(), y=d['y'][idx], subject=d['subject'][idx], record=d['record'][idx], beat_index=np.where(idx)[0], T=sel['temperature'])
    del model; gc.collect(); torch.cuda.empty_cache()
    return rec

def run(part='test', out_final=None, out_staging=None, log_dir=None, manifest_path=None, addendum_check=True, root=a1.R2):
    assert part in ('test', 'val')
    out_final = out_final or os.path.join(root, 'results', 'scores_mit'); out_staging = out_staging or os.path.join(root, 'results', 'scores_mit_INPROGRESS')
    a1.install_no_external_guard()                                   # first action: from here on no external-dataset path can be opened or listed
    a1.start_log(log_dir or os.path.join(root, 'logs'), '07a_score_mit'); awake = a1.keep_awake()
    a1.log(f'07a MIT-BIH-only scoring | partition={part} | keep-awake request held: {awake} | final={out_final}')
    lock = a1.verify_lock(include_external=False, root=root); a1.log(f"lock OK (external file not hashed in this stage): {lock['lock_file_sha256']}")
    add = a1.verify_addendum(root) if addendum_check else None; a1.log('addendum A1 verified' if add else 'ADDENDUM CHECK SKIPPED (test harness only)')
    minfo, man_sha = a1.verify_manifest_runs(root, manifest_path); a1.log('all 135 run files == manifest_post_training.json')
    a1.log(f'backup verified: {a1.verify_backup(root)} files')
    if os.path.exists(out_final) or os.path.exists(out_staging): raise RuntimeError(f'refusing: {out_final} or {out_staging} already exists (a previous or interrupted run must be inspected, never overwritten)')
    d = rc.load_mit(); order = [(m, k) for m in 'ABC' for k in range(5)]
    exp_n = {}
    for k in range(5): _, va, te = rc.partition(d['subject'], k); exp_n[k] = int((te if part == 'test' else va).sum())
    os.makedirs(out_staging)                                          # exclusive: raises if it exists
    a1.write_json_x(os.path.join(out_staging, '_STARTED.json'), dict(started=a1.now(), partition=part, script_sha256=a1.sha256_file(os.path.abspath(__file__)), lock_sha256=lock['lock_file_sha256']))
    runs_log = []; t_all = time.time()
    try:
        for (m, k) in order:
            run_dir = os.path.join(root, 'results', 'runs', f'{m}_fold{k}'); sel = json.load(open(os.path.join(run_dir, 'selected.json'))); info = minfo[f'{m}_fold{k}']
            assert sel['checkpoint'] == info['checkpoint']
            t0 = time.time(); rec = score_fold(m, k, d, sel, run_dir, part)
            a1.savez_x(os.path.join(out_staging, f'mit_{m}_fold{k}.npz'), **rec)
            runs_log.append(dict(model=m, fold=k, checkpoint=sel['checkpoint'], checkpoint_sha256=info['checkpoint_sha256'], T=sel['temperature'], n_test=int(len(rec['y']))))
            a1.log(f'scored {m} fold{k}: n={len(rec["y"])} ({time.time() - t0:.0f}s)')
        # ---- verification of the complete staging directory before the flag is written ----
        a1.log('verifying all 15 staged files'); bi = []
        for (m, k) in order:
            with np.load(os.path.join(out_staging, f'mit_{m}_fold{k}.npz')) as z:          # closed immediately: Windows cannot rename a directory holding open files
                n = len(z['y'])
                ok = (set(z.files) == {'logits', 'y', 'subject', 'record', 'beat_index', 'T'} and z['logits'].shape == (n, 5) and np.isfinite(z['logits']).all() and n == exp_n[k]
                      and np.array_equal(z['y'], d['y'][z['beat_index']]) and np.array_equal(z['subject'], d['subject'][z['beat_index']]) and abs(float(z['T']) - minfo[f'{m}_fold{k}']['T']) < 1e-12)
                if not ok: raise RuntimeError(f'verification failed for {m} fold{k}')
                if m == 'A': bi.append(z['beat_index'])
        allb = np.sort(np.concatenate(bi))
        if not np.array_equal(allb, np.arange(len(d['y']))): raise RuntimeError('beats are not covered exactly once by the five partitions')
        a1.log(f'coverage OK: {len(allb)} beats, each exactly once')
        outs = {n: a1.sha256_file(os.path.join(out_staging, n)) for n in sorted(os.listdir(out_staging)) if n.startswith('mit_')}
        a1.write_json_x(os.path.join(out_staging, 'SCORED.json'), dict(scope='MIT-BIH', partition=part, scored_at=a1.now(), seconds=round(time.time() - t_all), lock_sha256=lock['lock_file_sha256'],
                         addendum_sha256=(add or {}).get('addendum_json_sha256'), manifest_post_training_sha256=man_sha, script_sha256=a1.sha256_file(os.path.abspath(__file__)),
                         environment=a1.env_info(), runs=runs_log, output_files=outs, external_dataset_accessed=False, note='flag written last, after verification'))
        os.rename(out_staging, out_final); a1.log(f'DONE: {out_final}')
        return dict(final=out_final, runs=runs_log)
    except Exception:
        a1.log('FAILED; staging directory and all partial outputs are preserved for inspection: ' + out_staging); a1.log(traceback.format_exc()); raise

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--authorised-mit-test-scoring', action='store_true'); a = p.parse_args()
    if not a.authorised_mit_test_scoring: sys.exit('Refusing: explicit user authorisation flag --authorised-mit-test-scoring not given.')
    try: run(part='test')
    except Exception: sys.exit(2)
