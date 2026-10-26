# Attribution, unified variable set, W = 6 h

## 1. TabPFN (68 features) — permutation SHAP, mean |SHAP| (probability units), top 10

| rank | feature | mean abs SHAP |
|---|---|---|
| 1 | `age` | 0.0709 |
| 2 | `gcs_last` | 0.0685 |
| 3 | `fio2_last` | 0.0305 |
| 4 | `lactate_last` | 0.0220 |
| 5 | `platelets_count` | 0.0211 |
| 6 | `urine_n` | 0.0199 |
| 7 | `map_min` | 0.0147 |
| 8 | `sofa_coag` | 0.0147 |
| 9 | `urine_total` | 0.0144 |
| 10 | `lactate_min` | 0.0129 |

(figure: `results/fusion_shap/tabpfn_shap_W6.png`)

## 2. UTDE-SOFA (15 channels + notes) — occlusion

| full | without notes | without time series | notes only vs TS only |
|---|---|---|---|
| 0.823 | 0.805 (−0.018) | 0.614 (−0.209) | 0.614 vs 0.805 |

| channel | mean abs Δlogit when occluded | AUROC drop |
|---|---|---|
| Age | 0.496 | +0.023 |
| Lactate | 0.281 | +0.007 |
| MeanBP | 0.279 | +0.004 |
| GCS | 0.269 | +0.024 |
| Bilirubin | 0.249 | +0.015 |
| Urine | 0.242 | +0.006 |
| FiO2 | 0.228 | +0.007 |
| Platelets | 0.193 | -0.002 |
| Creatinine | 0.169 | -0.001 |
| Vasopressor | 0.165 | +0.006 |
| BaseExcess | 0.146 | +0.010 |
| PaO2 | 0.127 | +0.001 |
| Sex | 0.110 | -0.002 |
| MeanABP | 0.107 | +0.002 |
| Vent | 0.074 | -0.002 |

## 3. Late fusion TabPFN + UTDE-SOFA — exact stacker SHAP

stacker coefficients TabPFN 0.32, UTDE-SOFA 0.59; mean |SHAP| TabPFN 0.34, UTDE-SOFA 0.75 (UTDE share 69%); logit correlation 0.83; fusion test AUROC 0.839

| rank | input | mean abs attribution (fused logit) |
|---|---|---|
| 1 | notes (U), offset removed | 0.384 |
| 2 | Age (U) | 0.294 |
| 3 | age (T) | 0.208 |
| 4 | gcs_last (T) | 0.208 |
| 5 | Lactate (U) | 0.166 |
| 6 | MeanBP (U) | 0.165 |
| 7 | GCS (U) | 0.159 |
| 8 | Bilirubin (U) | 0.148 |
| 9 | Urine (U) | 0.143 |
| 10 | FiO2 (U) | 0.135 |
| 11 | Platelets (U) | 0.115 |
| 12 | Creatinine (U) | 0.100 |
| 13 | Vasopressor (U) | 0.098 |
| 14 | BaseExcess (U) | 0.087 |
| 15 | fio2_last (T) | 0.077 |

## 4. TabPFN-in-UTDE, replace (TabPFN tokens + notes): not available yet

## 4. TabPFN-in-UTDE, 3 streams (TabPFN tokens + 15-ch time series + notes): not available yet

