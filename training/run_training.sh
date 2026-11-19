#!/bin/sh
# R2 training chain under the locked protocol: fold by fold, models A, B, C. Each run verifies the lock itself.
cd /d/explainable_ai/revision/R2
for k in 0 1 2 3 4; do for m in A B C; do
  echo "=== START $m fold$k $(date -Iseconds)"
  PYTHONIOENCODING=utf-8 python -W ignore experiments/06_train.py $m $k > logs/train_${m}_fold${k}.log 2>&1
  rc=$?; echo "=== END $m fold$k rc=$rc $(date -Iseconds)"; tail -1 logs/train_${m}_fold${k}.log
done; echo "=== FOLD $k COMPLETE $(date -Iseconds)"; done
echo "=== ALL TRAINING COMPLETE"
