# Late-fusion SHAP, W = 24 h (test n = 1374)

## 1. Fusion stacker (linear in logits): exact SHAP

coefficients: TabPFN 0.411, UTDE 0.429; mean |SHAP| TabPFN 0.530, UTDE 0.739 (TabPFN share 0.42); logit correlation 0.79; cases where the two disagree about the direction (vs their medians): 0.21 — fusion AUROC 0.600 on those (TabPFN 0.565, UTDE 0.524) vs 0.886 where they agree.

## 2. TabPFN (tier 4 + age/sex) permutation SHAP, refit test AUROC 0.837, explained cases 1374

| rank | feature | mean abs SHAP (prob.) | mean SHAP |
|---|---|---|---|
| 1 | gcs_last | 0.1110 | -0.0269 |
| 2 | age | 0.0482 | -0.0051 |
| 3 | urine_n | 0.0206 | -0.0018 |
| 4 | gcs_count | 0.0202 | +0.0033 |
| 5 | lactate_min | 0.0193 | +0.0135 |
| 6 | lactate_last | 0.0182 | +0.0020 |
| 7 | base_excess_last | 0.0166 | -0.0034 |
| 8 | gcs_min | 0.0157 | +0.0093 |
| 9 | urine_total | 0.0147 | -0.0041 |
| 10 | sofa_coag | 0.0136 | -0.0022 |
| 11 | map_mean | 0.0123 | -0.0035 |
| 12 | platelets_count | 0.0122 | +0.0081 |
| 13 | urine_ml_per_h | 0.0111 | -0.0030 |
| 14 | map_last | 0.0102 | +0.0048 |
| 15 | map_min | 0.0102 | +0.0019 |
| 16 | sofa_cns | 0.0099 | +0.0059 |
| 17 | pao2_mean | 0.0080 | +0.0020 |
| 18 | fio2_last | 0.0080 | -0.0013 |
| 19 | fio2_mean | 0.0076 | -0.0032 |
| 20 | base_excess_count | 0.0067 | +0.0048 |

### grouped by source variable (sum of mean abs SHAP over its aggregates)

| variable | total mean abs SHAP |
|---|---|
| gcs | 0.1469 |
| age | 0.0482 |
| urine | 0.0463 |
| lactate | 0.0408 |
| map | 0.0354 |
| fio2 | 0.0286 |
| base_excess | 0.0275 |
| platelets | 0.0212 |
| pao2 | 0.0210 |
| sofa_coag | 0.0136 |
| bilirubin | 0.0105 |
| sofa_cns | 0.0099 |
| creatinine | 0.0087 |
| vaso_epinephrine | 0.0053 |
| sofa_resp | 0.0044 |
