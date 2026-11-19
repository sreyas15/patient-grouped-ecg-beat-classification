#!/bin/sh
# Waits for the training chain, checks all 15 runs succeeded, then: one-time scoring -> locked evaluation -> post-hoc analysis.
cd /d/explainable_ai/revision/R2
until grep -q "ALL TRAINING COMPLETE" logs/run_training.log; do sleep 60; done
n_ok=$(grep -c "rc=0" logs/run_training.log); n_sel=$(ls results/runs/*/selected.json | wc -l)
echo "training finished: rc=0 runs=$n_ok selected=$n_sel"
if [ "$n_ok" -ne 15 ] || [ "$n_sel" -ne 15 ]; then echo "HALT: not all 15 runs completed"; exit 1; fi
PYTHONIOENCODING=utf-8 python -W ignore experiments/07_score.py > logs/07_score.log 2>&1 || { echo "HALT: scoring failed"; exit 1; }
echo "SCORING DONE"
PYTHONIOENCODING=utf-8 python -W ignore experiments/08_evaluate.py > logs/08_evaluate.log 2>&1 || { echo "HALT: evaluation failed"; exit 1; }
echo "EVALUATION DONE"
PYTHONIOENCODING=utf-8 python -W ignore experiments/09_class_driver_analysis.py > logs/09_posthoc.log 2>&1 || { echo "HALT: post-hoc failed"; exit 1; }
echo "POSTHOC DONE"
