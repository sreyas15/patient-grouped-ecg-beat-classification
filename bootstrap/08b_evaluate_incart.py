"""R2 protocol addendum A1 -- 08b: INCART-ONLY external evaluation (the INCART half of the frozen 08_evaluate.py main section).
SEPARATELY AUTHORISED STAGE. It imports the frozen 08_evaluate.py (hash-locked; definitions only) and EXECUTES ITS OWN STATEMENTS, verbatim: lines 129-139 (INCART labels,
ensemble = mean of the five fold models' temperature-scaled softmax, as-given and peak-recentred variants, macro-F1 over N,S,V,F via cls=[0,1,2,3], patient-cluster bootstrap,
individual fold-model table, evaluate('incart_external*', ...)). The frozen MIT-BIH statements (114-127) are not executed.
Preconditions: INCART scores verified against results/scores_incart/SCORED.json, lock, manifest; MIT-BIH scores and evaluation exist and are byte-identical to what was recorded
when INCART was scored. A write guard forbids any write/rename/delete under the MIT-BIH directories.
WRITE-ONCE: results/evaluation_incart_INPROGRESS/ -> verified -> EVALUATED.json last -> rename to results/evaluation_incart/. Refuses if either exists.
Usage (only after explicit INCART evaluation authorisation):  python -P -B experiments/08b_evaluate_incart.py --authorised-incart-evaluation"""
import os, sys, json, time, argparse, traceback
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import a1common as a1

def expected_outputs(models='ABC'):
    n = []
    for tag in ('incart_external', 'incart_external_recentred'):
        n += [f'{tag}_individual_fold_models.csv', f'{tag}_summary.json', f'{tag}_per_class.csv', f'{tag}_comparisons.json']
        n += [f'{tag}_{m}_{s}.csv' for m in models for s in ('reliability_bins', 'confusion_counts', 'confusion_rownorm')]
    return n

def run(scores_dir=None, out_final=None, out_staging=None, log_dir=None, nb=None, manifest_path=None, addendum_check=True, verify_external_file=True, root=a1.R2, mit_scores=None, mit_eval=None):
    scores_dir = scores_dir or os.path.join(root, 'results', 'scores_incart'); out_final = out_final or os.path.join(root, 'results', 'evaluation_incart')
    out_staging = out_staging or os.path.join(root, 'results', 'evaluation_incart_INPROGRESS')
    mit_scores = mit_scores or os.path.join(root, 'results', 'scores_mit'); mit_eval = mit_eval or os.path.join(root, 'results', 'evaluation_mit')
    a1.install_no_write_guard([mit_scores, mit_scores + '_INPROGRESS', mit_eval, mit_eval + '_INPROGRESS'])
    a1.start_log(log_dir or os.path.join(root, 'logs'), '08b_evaluate_incart'); awake = a1.keep_awake(); a1.log(f'08b INCART-only evaluation | keep-awake: {awake} | scores={scores_dir}')
    lock = a1.verify_lock(include_external=verify_external_file, root=root); a1.log(f"lock OK: {lock['lock_file_sha256']}")
    add = a1.verify_addendum(root) if addendum_check else None; a1.log('addendum A1 verified' if add else 'ADDENDUM CHECK SKIPPED (test harness only)')
    minfo, man_sha = a1.verify_manifest_runs(root, manifest_path); a1.verify_backup(root); a1.log('manifest + backup verified')
    flag = a1.verify_scored_dir(scores_dir, 'INCART', lock['lock_file_sha256'], minfo, ('external',)); a1.log('INCART scores verified against SCORED.json, lock and manifest')
    now_snap = dict(scores=a1.snapshot_dir(mit_scores), evaluation=a1.snapshot_dir(mit_eval))
    if now_snap != flag['mit_stage_snapshot_unchanged']: raise RuntimeError('refusing: MIT-BIH outputs differ from the snapshot recorded when INCART was scored')
    a1.log('MIT-BIH outputs unchanged since INCART scoring')
    if os.path.exists(out_final) or os.path.exists(out_staging): raise RuntimeError(f'refusing: {out_final} or {out_staging} already exists')
    ev = a1.load_frozen_eval(root); _, inc_code = a1.frozen_main_blocks(root)
    ev.SC = scores_dir; ev.OUT = out_staging
    if nb is not None: ev.NB = int(nb)                                   # test harness only; production leaves the frozen value (2000)
    os.makedirs(out_staging); a1.write_json_x(os.path.join(out_staging, '_STARTED.json'), dict(started=a1.now(), script_sha256=a1.sha256_file(os.path.abspath(__file__)), nb=ev.NB))
    t0 = time.time()
    try:
        a1.log(f'executing frozen 08_evaluate.py lines 129-139 (B={ev.NB}, seed={ev.BSEED})'); exec(inc_code, ev.__dict__)
        have = sorted(n for n in os.listdir(out_staging) if n != '_STARTED.json')
        if have != sorted(expected_outputs()): raise RuntimeError(f'unexpected output set: {sorted(set(have) ^ set(expected_outputs()))}')
        if {k: v for k, v in dict(scores=a1.snapshot_dir(mit_scores), evaluation=a1.snapshot_dir(mit_eval)).items()} != now_snap: raise RuntimeError('MIT-BIH outputs changed during the INCART evaluation')
        outs = {n: a1.sha256_file(os.path.join(out_staging, n)) for n in have}
        a1.write_json_x(os.path.join(out_staging, 'EVALUATED.json'), dict(scope='INCART', evaluated_at=a1.now(), seconds=round(time.time() - t0), lock_sha256=lock['lock_file_sha256'],
                         addendum_sha256=(add or {}).get('addendum_json_sha256'), manifest_post_training_sha256=man_sha, scored_json_sha256=a1.sha256_file(os.path.join(scores_dir, 'SCORED.json')),
                         frozen_08_evaluate_sha256=lock['frozen_files']['experiments/08_evaluate.py'], script_sha256=a1.sha256_file(os.path.abspath(__file__)), bootstrap=dict(B=ev.NB, seed=ev.BSEED),
                         primary_macro_f1_classes='N,S,V,F (Q excluded: 6 beats, no paced patients; frozen protocol 6/D6)', output_files=outs, mit_outputs_verified_unchanged=True))
        os.rename(out_staging, out_final); a1.log(f'DONE: {out_final}'); return dict(final=out_final, outputs=outs)
    except Exception:
        a1.log('FAILED; staging directory and partial outputs preserved: ' + out_staging); a1.log(traceback.format_exc()); raise

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--authorised-incart-evaluation', action='store_true'); a = p.parse_args()
    if not a.authorised_incart_evaluation: sys.exit('Refusing: explicit user authorisation flag --authorised-incart-evaluation not given.')
    try: run()
    except Exception: sys.exit(2)
