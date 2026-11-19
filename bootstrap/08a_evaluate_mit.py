"""R2 protocol addendum A1 -- 08a: MIT-BIH-ONLY primary evaluation (replaces the MIT-BIH half of the frozen 08_evaluate.py main section).
It imports the frozen 08_evaluate.py (hash-locked; importing runs only definitions) and EXECUTES ITS OWN STATEMENTS, verbatim, from the __main__ section:
lines 114-127 (pooled out-of-fold assembly with per-fold temperatures, per-fold table, evaluate('mit_internal', ..., cls=all five classes)). Nothing is re-typed;
all metric definitions, the bootstrap (2000 resamples, seed 20261102) and the comparisons are the frozen ones. The frozen INCART statements (129-139) are NOT executed.
ISOLATION: process-level guard raises on any open/list of a path containing 'incart'. WRITE-ONCE: staging dir results/evaluation_mit_INPROGRESS/ -> verified -> EVALUATED.json
written last -> renamed to results/evaluation_mit/. Refuses if either exists. Inputs are verified against SCORED.json, the lock and the manifest first.
Usage (only after MIT scoring and explicit approval):  python -P -B experiments/08a_evaluate_mit.py --authorised-mit-evaluation"""
import os, sys, json, time, argparse, traceback
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import a1common as a1

def expected_outputs(models='ABC'):
    n = ['mit_internal_per_fold.csv', 'mit_internal_summary.json', 'mit_internal_per_class.csv', 'mit_internal_comparisons.json']
    return n + [f'mit_internal_{m}_{s}.csv' for m in models for s in ('reliability_bins', 'confusion_counts', 'confusion_rownorm')]

def run(scores_dir=None, out_final=None, out_staging=None, log_dir=None, nb=None, manifest_path=None, addendum_check=True, allow_partition=('test',), root=a1.R2):
    scores_dir = scores_dir or os.path.join(root, 'results', 'scores_mit'); out_final = out_final or os.path.join(root, 'results', 'evaluation_mit')
    out_staging = out_staging or os.path.join(root, 'results', 'evaluation_mit_INPROGRESS')
    a1.install_no_external_guard()
    a1.start_log(log_dir or os.path.join(root, 'logs'), '08a_evaluate_mit'); awake = a1.keep_awake(); a1.log(f'08a MIT-BIH-only evaluation | keep-awake: {awake} | scores={scores_dir}')
    lock = a1.verify_lock(include_external=False, root=root); a1.log(f"lock OK: {lock['lock_file_sha256']}")
    add = a1.verify_addendum(root) if addendum_check else None; a1.log('addendum A1 verified' if add else 'ADDENDUM CHECK SKIPPED (test harness only)')
    minfo, man_sha = a1.verify_manifest_runs(root, manifest_path); a1.verify_backup(root); a1.log('manifest + backup verified')
    flag = a1.verify_scored_dir(scores_dir, 'MIT-BIH', lock['lock_file_sha256'], minfo, allow_partition); a1.log('MIT-BIH scores verified against SCORED.json, lock and manifest')
    if os.path.exists(out_final) or os.path.exists(out_staging): raise RuntimeError(f'refusing: {out_final} or {out_staging} already exists')
    ev = a1.load_frozen_eval(root); mit_code, _ = a1.frozen_main_blocks(root)
    ev.SC = scores_dir; ev.OUT = out_staging
    if nb is not None: ev.NB = int(nb)                                   # test harness only; production leaves the frozen value (2000)
    os.makedirs(out_staging); a1.write_json_x(os.path.join(out_staging, '_STARTED.json'), dict(started=a1.now(), script_sha256=a1.sha256_file(os.path.abspath(__file__)), nb=ev.NB))
    t0 = time.time()
    try:
        a1.log(f'executing frozen 08_evaluate.py lines 114-127 (B={ev.NB}, seed={ev.BSEED})'); exec(mit_code, ev.__dict__)
        have = sorted(n for n in os.listdir(out_staging) if n != '_STARTED.json')
        if have != sorted(expected_outputs()): raise RuntimeError(f'unexpected output set: {sorted(set(have) ^ set(expected_outputs()))}')
        outs = {n: a1.sha256_file(os.path.join(out_staging, n)) for n in have}
        a1.write_json_x(os.path.join(out_staging, 'EVALUATED.json'), dict(scope='MIT-BIH', evaluated_at=a1.now(), seconds=round(time.time() - t0), lock_sha256=lock['lock_file_sha256'],
                         addendum_sha256=(add or {}).get('addendum_json_sha256'), manifest_post_training_sha256=man_sha, scored_json_sha256=a1.sha256_file(os.path.join(scores_dir, 'SCORED.json')),
                         frozen_08_evaluate_sha256=lock['frozen_files']['experiments/08_evaluate.py'], script_sha256=a1.sha256_file(os.path.abspath(__file__)), bootstrap=dict(B=ev.NB, seed=ev.BSEED),
                         output_files=outs, external_dataset_accessed=False))
        os.rename(out_staging, out_final); a1.log(f'DONE: {out_final}'); return dict(final=out_final, outputs=outs)
    except Exception:
        a1.log('FAILED; staging directory and partial outputs preserved: ' + out_staging); a1.log(traceback.format_exc()); raise

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--authorised-mit-evaluation', action='store_true'); a = p.parse_args()
    if not a.authorised_mit_evaluation: sys.exit('Refusing: explicit user authorisation flag --authorised-mit-evaluation not given.')
    try: run()
    except Exception: sys.exit(2)
