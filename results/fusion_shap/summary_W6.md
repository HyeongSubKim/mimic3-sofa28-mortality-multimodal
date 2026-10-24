# Late-fusion SHAP, W = 6 h (test n = 853)

## 1. Fusion stacker (linear in logits): exact SHAP

coefficients: TabPFN 0.554, UTDE 0.540; mean |SHAP| TabPFN 0.601, UTDE 0.549 (TabPFN share 0.52); logit correlation 0.67; cases where the two disagree about the direction (vs their medians): 0.26 — fusion AUROC 0.598 on those (TabPFN 0.659, UTDE 0.389) vs 0.883 where they agree.

## 2. TabPFN (tier 4 + age/sex) permutation SHAP, refit test AUROC 0.832, explained cases 853

| rank | feature | mean abs SHAP (prob.) | mean SHAP |
|---|---|---|---|
| 1 | age | 0.0709 | +0.0158 |
| 2 | gcs_last | 0.0685 | -0.0314 |
| 3 | fio2_last | 0.0305 | +0.0110 |
| 4 | lactate_last | 0.0220 | +0.0083 |
| 5 | platelets_count | 0.0211 | +0.0120 |
| 6 | urine_n | 0.0199 | +0.0004 |
| 7 | map_min | 0.0147 | +0.0057 |
| 8 | sofa_coag | 0.0147 | -0.0040 |
| 9 | urine_total | 0.0144 | +0.0007 |
| 10 | lactate_min | 0.0129 | +0.0044 |
| 11 | sofa_total | 0.0127 | -0.0084 |
| 12 | map_last | 0.0115 | +0.0041 |
| 13 | urine_ml_per_h | 0.0115 | -0.0002 |
| 14 | gcs_count | 0.0111 | -0.0014 |
| 15 | bilirubin_last | 0.0110 | -0.0076 |
| 16 | vaso_norepinephrine | 0.0108 | +0.0002 |
| 17 | base_excess_count | 0.0104 | -0.0024 |
| 18 | base_excess_min | 0.0092 | +0.0018 |
| 19 | base_excess_max | 0.0089 | +0.0009 |
| 20 | base_excess_last | 0.0085 | +0.0024 |

### grouped by source variable (sum of mean abs SHAP over its aggregates)

| variable | total mean abs SHAP |
|---|---|
| gcs | 0.0818 |
| age | 0.0709 |
| fio2 | 0.0469 |
| lactate | 0.0465 |
| urine | 0.0458 |
| base_excess | 0.0423 |
| platelets | 0.0393 |
| bilirubin | 0.0362 |
| map | 0.0340 |
| pao2 | 0.0211 |
| sofa_coag | 0.0147 |
| vaso | 0.0137 |
| sofa | 0.0127 |
| creatinine | 0.0110 |
| vaso_norepinephrine | 0.0108 |
