"""R2 protocol addendum A1 -- 07b: INCART-ONLY one-time scoring (the external-dataset half of the frozen 07_score.py, lines 23-24 and 28; the frozen file is not modified or run).
SEPARATELY AUTHORISED STAGE. It refuses to run unless the MIT-BIH stage is complete: results/scores_mit/SCORED.json and results/evaluation_mit/EVALUATED.json must exist and
every file in those directories must still match the hashes recorded in them. A process-level write guard makes any write/rename/delete under the MIT-BIH score or
evaluation directories (final or staging) raise PermissionError, so this stage cannot influence MIT-BIH results; MIT-BIH directories are re-hashed before the flag is written.
Uses exactly the selected checkpoints (hash-verified against manifest_post_training.json before loading) and the frozen scoring expressions:
logits and logits_recentred for X and X_recentred, T = selected temperature (no re-fit) -> results/scores_incart/incart_{A,B,C}_fold{0-4}.npz + incart_labels.npz. No metric is computed.
WRITE-ONCE: results/scores_incart_INPROGRESS/ -> verified -> SCORED.json last -> rename. Refuses if staging or final exists.
Usage (only after explicit INCART authorisation):  python -P -B experiments/07b_score_incart.py --authorised-incart-scoring"""
import os, sys, json, time, gc, argparse, traceback
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import a1common as a1
import numpy as np, torch
import r2common as rc

def score_external_fold(m, k, ext, sel, run_dir):
    """Identical expressions to the frozen 07_score.py lines 19, 23-24."""
    model = rc.build(m).to('cuda'); model.load_state_dict(torch.load(os.path.join(run_dir, sel['checkpoint']))); model.eval()
    rec = dict(logits=rc.logits(model, ext['X']).numpy(), logits_recentred=rc.logits(model, ext['X_recentred']).numpy(), T=sel['temperature'])
    del model; gc.collect(); torch.cuda.empty_cache()
    return rec

def run(loader=None, out_final=None, out_staging=None, log_dir=None, manifest_path=None, addendum_check=True, verify_external_file=True, root=a1.R2, mit_scores=None, mit_eval=None):
    out_final = out_final or os.path.join(root, 'results', 'scores_incart'); out_staging = out_staging or os.path.join(root, 'results', 'scores_incart_INPROGRESS')
    mit_scores = mit_scores or os.path.join(root, 'results', 'scores_mit'); mit_eval = mit_eval or os.path.join(root, 'results', 'evaluation_mit')
    a1.install_no_write_guard([mit_scores, mit_scores + '_INPROGRESS', mit_eval, mit_eval + '_INPROGRESS'])
    a1.start_log(log_dir or os.path.join(root, 'logs'), '07b_score_incart'); awake = a1.keep_awake(); a1.log(f'07b INCART-only scoring | keep-awake: {awake} | final={out_final}')
    lock = a1.verify_lock(include_external=verify_external_file, root=root); a1.log(f"lock OK: {lock['lock_file_sha256']}")
    add = a1.verify_addendum(root) if addendum_check else None; a1.log('addendum A1 verified' if add else 'ADDENDUM CHECK SKIPPED (test harness only)')
    minfo, man_sha = a1.verify_manifest_runs(root, manifest_path); a1.verify_backup(root); a1.log('manifest + backup verified')
    for f in (os.path.join(mit_scores, 'SCORED.json'), os.path.join(mit_eval, 'EVALUATED.json')):
        if not os.path.exists(f): raise RuntimeError(f'refusing: MIT-BIH stage not complete ({f} missing)')
    fs, fe = json.load(open(os.path.join(mit_scores, 'SCORED.json'))), json.load(open(os.path.join(mit_eval, 'EVALUATED.json')))
    snap_before = dict(scores=a1.snapshot_dir(mit_scores), evaluation=a1.snapshot_dir(mit_eval))
    ok = all(snap_before['scores'].get(n) == h for n, h in fs['output_files'].items()) and all(snap_before['evaluation'].get(n) == h for n, h in fe['output_files'].items())
    if not ok: raise RuntimeError('refusing: MIT-BIH outputs no longer match the hashes recorded in their SCORED/EVALUATED flags')
    a1.log('MIT-BIH stage verified complete and unchanged')
    if os.path.exists(out_final) or os.path.exists(out_staging): raise RuntimeError(f'refusing: {out_final} or {out_staging} already exists')
    ext = (loader or rc.load_incart)(); n_ext = len(ext['y']); order = [(m, k) for m in 'ABC' for k in range(5)]
    os.makedirs(out_staging); a1.write_json_x(os.path.join(out_staging, '_STARTED.json'), dict(started=a1.now(), script_sha256=a1.sha256_file(os.path.abspath(__file__)), lock_sha256=lock['lock_file_sha256']))
    runs_log = []; t_all = time.time()
    try:
        for (m, k) in order:
            run_dir = os.path.join(root, 'results', 'runs', f'{m}_fold{k}'); sel = json.load(open(os.path.join(run_dir, 'selected.json'))); info = minfo[f'{m}_fold{k}']
            assert sel['checkpoint'] == info['checkpoint']
            t0 = time.time(); rec = score_external_fold(m, k, ext, sel, run_dir); a1.savez_x(os.path.join(out_staging, f'incart_{m}_fold{k}.npz'), **rec)
            runs_log.append(dict(model=m, fold=k, checkpoint=sel['checkpoint'], checkpoint_sha256=info['checkpoint_sha256'], T=sel['temperature'], n_beats=int(n_ext)))
            a1.log(f'scored {m} fold{k}: n={n_ext} ({time.time() - t0:.0f}s)')
        a1.savez_x(os.path.join(out_staging, 'incart_labels.npz'), y=ext['y'], patient=ext['patient'], record=ext['record'])
        a1.log('verifying staged files')
        for (m, k) in order:
            with np.load(os.path.join(out_staging, f'incart_{m}_fold{k}.npz')) as z:       # closed immediately: Windows cannot rename a directory holding open files
                if not (set(z.files) == {'logits', 'logits_recentred', 'T'} and z['logits'].shape == (n_ext, 5) and z['logits_recentred'].shape == (n_ext, 5) and np.isfinite(z['logits']).all()
                        and np.isfinite(z['logits_recentred']).all() and abs(float(z['T']) - minfo[f'{m}_fold{k}']['T']) < 1e-12): raise RuntimeError(f'verification failed for {m} fold{k}')
        with np.load(os.path.join(out_staging, 'incart_labels.npz')) as zl:
            if not (set(zl.files) == {'y', 'patient', 'record'} and len(zl['y']) == n_ext): raise RuntimeError('labels verification failed')
        snap_after = dict(scores=a1.snapshot_dir(mit_scores), evaluation=a1.snapshot_dir(mit_eval))
        if snap_after != snap_before: raise RuntimeError('MIT-BIH outputs changed during the INCART stage')
        outs = {n: a1.sha256_file(os.path.join(out_staging, n)) for n in sorted(os.listdir(out_staging)) if n.startswith('incart_')}
        a1.write_json_x(os.path.join(out_staging, 'SCORED.json'), dict(scope='INCART', partition='external', scored_at=a1.now(), seconds=round(time.time() - t_all), lock_sha256=lock['lock_file_sha256'],
                         addendum_sha256=(add or {}).get('addendum_json_sha256'), manifest_post_training_sha256=man_sha, script_sha256=a1.sha256_file(os.path.abspath(__file__)), environment=a1.env_info(),
                         runs=runs_log, output_files=outs, mit_stage_snapshot_unchanged=snap_before, note='flag written last, after verification; MIT-BIH directories verified unchanged'))
        os.rename(out_staging, out_final); a1.log(f'DONE: {out_final}'); return dict(final=out_final, runs=runs_log)
    except Exception:
        a1.log('FAILED; staging directory and partial outputs preserved: ' + out_staging); a1.log(traceback.format_exc()); raise

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--authorised-incart-scoring', action='store_true'); a = p.parse_args()
    if not a.authorised_incart_scoring: sys.exit('Refusing: explicit user authorisation flag --authorised-incart-scoring not given.')
    try: run()
    except Exception: sys.exit(2)
