#!/bin/sh
# R2 resume chain (authors' recovery option (a); run in your own terminal, NOT by Claude Code).
# - Same locked 06_train.py; completed runs (selected.json present) are skipped by 06_train.py itself.
# - Strictly sequential: one training process at a time; aborts if another 06_train.py is already running.
# - Before every run: waits until >= MIN_FREE_GB physical RAM is free (Model C needs ~8-9 GB, see results/diagnostics/memory_diag.json).
# - STOPS the whole chain on the first failed run (rc != 0) instead of continuing.
# - Does NOT run test scoring or INCART evaluation (authors' safeguards 8-9).
cd "$(dirname "$0")/.." || exit 1
LOG=logs/run_training_resume.log; MIN_FREE_GB=10
free_gb() { powershell -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)" | tr -d '\r'; }
busy() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '06_train\.py' }).Count" | tr -d '\r'; }
echo "=== RESUME START $(date -Iseconds)" | tee -a $LOG
lock_out=$(python experiments/verify_lock.py 2>&1); lock_rc=$?; echo "$lock_out" | tee -a $LOG
[ $lock_rc -eq 0 ] || { echo "HALT: lock verification failed (rc=$lock_rc)" | tee -a $LOG; exit 1; }
for job in "C 2" "A 3" "B 3" "C 3" "A 4" "B 4" "C 4"; do
  set -- $job; m=$1; k=$2
  if [ -f results/runs/${m}_fold${k}/selected.json ]; then echo "=== SKIP $m fold$k (already complete)" | tee -a $LOG; continue; fi
  if [ "$(busy)" != "0" ]; then echo "HALT: another 06_train.py is running; refusing concurrent training" | tee -a $LOG; exit 1; fi
  while [ "$(free_gb)" -lt $MIN_FREE_GB ]; do echo "=== WAIT $m fold$k: free RAM $(free_gb) GB < $MIN_FREE_GB GB $(date -Iseconds)" | tee -a $LOG; sleep 120; done
  echo "=== START $m fold$k free_RAM=$(free_gb)GB $(date -Iseconds)" | tee -a $LOG
  PYTHONIOENCODING=utf-8 python -W ignore experiments/06_train.py $m $k > logs/train_${m}_fold${k}.log 2>&1
  rc=$?; echo "=== END $m fold$k rc=$rc $(date -Iseconds)" | tee -a $LOG; tail -1 logs/train_${m}_fold${k}.log | tee -a $LOG
  if [ $rc -ne 0 ] || [ ! -f results/runs/${m}_fold${k}/selected.json ]; then echo "HALT: $m fold$k failed (rc=$rc). Chain stopped; nothing else run." | tee -a $LOG; exit 1; fi
  if [ -f results/runs/A_fold${k}/selected.json ] && [ -f results/runs/B_fold${k}/selected.json ] && [ -f results/runs/C_fold${k}/selected.json ]; then
    echo "=== FOLD $k COMPLETE (A, B, C all have selected.json) $(date -Iseconds)" | tee -a $LOG; fi
done
n=$(ls results/runs/*/selected.json 2>/dev/null | wc -l)
echo "=== RESUME FINISHED: $n/15 runs complete $(date -Iseconds). Test scoring NOT run." | tee -a $LOG
