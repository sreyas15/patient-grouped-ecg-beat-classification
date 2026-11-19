# Interrupted run: Model C, outer fold 2, attempt 1 (preserved, not used)

- Started 2026-11-02 19:02:10 under the locked protocol (lock sha256 ed0717f2...61478c), seed 1002.
- Learning rate 1e-4 completed at 19:40:20 (best validation macro-F1 0.3581, epoch 12); checkpoint and history kept here.
- Learning rate 5e-4 was in progress when the laptop entered sleep at 20:03:25 (System log, Kernel-Power 42). It resumed on 2026-11-03 at 13:43;
  the NVIDIA driver logged three GPU resets (nvlddmkm event 153) at 13:45:14-15 and the run crashed at 13:45:20 with
  `RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED` (rc=1). No `selected.json` was produced; no temperature was fitted.
- The chain log line "FOLD 2 COMPLETE" written afterwards is misleading: the chain script printed it unconditionally after C fold 2 exited with rc=1.
- This attempt is NOT used for any result. Model C fold 2 is re-run from scratch with identical locked code, data, configuration and seed.
