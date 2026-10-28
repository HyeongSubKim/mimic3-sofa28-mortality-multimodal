# Occlusion attribution, both models, W = 6 h (AUROC drop when the variable is removed; mean |Δlogit| in brackets)

| variable | TabPFN (group of columns → NaN) | UTDE-SOFA (channel → 0) |
|---|---|---|
| GCS | −0.042 [0.40] | −0.024 [0.27] |
| Age | −0.022 [0.51] | −0.023 [0.50] |
| Urine | −0.009 [0.28] | −0.006 [0.24] |
| Bilirubin | −0.008 [0.08] | −0.015 [0.25] |
| Platelets | −0.007 [0.17] | +0.002 [0.19] |
| MeanBP | −0.005 [0.17] | −0.004 [0.28] |
| BaseExcess | −0.004 [0.14] | −0.010 [0.15] |
| PaO2 | −0.003 [0.09] | −0.001 [0.13] |
| Lactate | −0.003 [0.27] | −0.007 [0.28] |
| FiO2 | −0.001 [0.21] | −0.007 [0.23] |
| Vasopressor | +0.000 [0.14] | −0.006 [0.16] |
| Vent | +0.000 [0.04] | +0.002 [0.07] |
| Creatinine | +0.000 [0.06] | +0.001 [0.17] |
| Sex | +0.001 [0.06] | +0.002 [0.11] |
| SOFA component scores (7 cols) | −0.003 [0.14] | – |
| P/F ratio (3 cols) | −0.000 [0.04] | – |
| notes | – | −0.018 [0.67] |

stacker coefficients b₁ (TabPFN) 0.32, b₂ (UTDE-SOFA) 0.59. MeanBP in UTDE = MeanBP + MeanABP channels (shown: MeanBP).

## fused-logit attribution, both sides occlusion (top 15)

| rank | input | mean abs attribution |
|---|---|---|
| 1 | notes (U) | 0.384 |
| 2 | Age (U) | 0.294 |
| 3 | Lactate (U) | 0.166 |
| 4 | MeanBP (U) | 0.165 |
| 5 | age (T) | 0.161 |
| 6 | GCS (U) | 0.159 |
| 7 | Bilirubin (U) | 0.148 |
| 8 | Urine (U) | 0.143 |
| 9 | FiO2 (U) | 0.135 |
| 10 | gcs (T) | 0.126 |
| 11 | Platelets (U) | 0.115 |
| 12 | Creatinine (U) | 0.100 |
| 13 | Vasopressor (U) | 0.098 |
| 14 | urine (T) | 0.088 |
| 15 | BaseExcess (U) | 0.087 |
