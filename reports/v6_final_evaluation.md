# V6 final evaluation — BLOCKED / NOT RUN ON REAL DATA

There are no trained V6 clinical checkpoints or authorized held-out real subjects.
ROC-AUC/PR-AUC, per-device sensitivity/specificity, calibration/ECE, bootstrap CIs,
HC device-probe gate and paired V5 significance are **NOT MEASURED**.

The evaluation implementation is `src/evaluation/v6_eval.py`. It produces tables
and aggregate ROC/calibration plots when real held-out predictions are supplied.
Synthetic correctness reports and plots were generated only in `/tmp`, never
presented as clinical evaluation. `docs/v6_results.md` is the results register;
`BLOCKERS.md` lists data, governance and GPU prerequisites. Threshold remains0.50.
