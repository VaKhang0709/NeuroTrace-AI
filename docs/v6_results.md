# V6 results register — NOT CLINICALLY TRAINED

## Prespecified target (not a measured result)

Subject-level ROC-AUC >0.97 on locked real holdout and frozen-feature HC device
probe AUC <0.70. No claim of superiority over published work is made: cohort,
modality, sample size, subject splits and test selection must match for comparison.

| Model | Cohort | ROC-AUC [95% CI] | PR-AUC | Sensitivity@.50 | Specificity@.50 | Sensitivity@95%spec | ECE |
|---|---|---|---|---|---|---|---|
| V6 TremorFormer | Real locked holdout | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| V6 per-device Wacom | Real subjects unavailable | N/A | N/A | N/A | N/A | N/A | N/A |
| V6 per-device touch | Real subjects unavailable | N/A | N/A | N/A | N/A | N/A | N/A |
| V6 per-device mouse | Real subjects unavailable | N/A | N/A | N/A | N/A | N/A | N/A |
| V5 paired baseline | No paired V6 holdout predictions | INCONCLUSIVE | N/A | N/A | N/A | N/A | N/A |

Device leakage probe and McNemar p-value: **NOT MEASURED on real data**.
Do not substitute V5 historical metrics, synthetic smoke AUC or randomized network
scores for any missing V6 result. No download link exists for clinically trained
V6 weights. See `reports/v6_correctness_report.md` for engineering-only evidence.
