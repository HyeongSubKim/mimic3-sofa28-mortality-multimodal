# Occlusion attribution, both models, W = 24 h (AUROC drop when the variable is removed; mean |Δlogit| in brackets)

| variable | TabPFN (group of columns → NaN) | UTDE-SOFA (channel → 0) |
|---|---|---|
| GCS | −0.059 [0.77] | −0.057 [0.62] |
| Age | −0.017 [0.43] | −0.020 [0.48] |
| Urine | −0.016 [0.34] | −0.005 [0.41] |
| MeanBP | −0.005 [0.25] | −0.005 [0.27] |
| FiO2 | −0.003 [0.13] | −0.004 [0.14] |
| PaO2 | −0.003 [0.10] | −0.008 [0.21] |
| BaseExcess | −0.003 [0.13] | −0.003 [0.14] |
| Platelets | −0.003 [0.09] | −0.002 [0.11] |
| Bilirubin | −0.003 [0.10] | −0.007 [0.11] |
| Lactate | −0.002 [0.27] | −0.008 [0.29] |
| Creatinine | −0.001 [0.06] | +0.001 [0.09] |
| Vent | +0.000 [0.10] | −0.003 [0.33] |
| Sex | +0.000 [0.02] | +0.000 [0.07] |
| Vasopressor | +0.001 [0.17] | −0.002 [0.06] |
| SOFA component scores (7 cols) | −0.009 [0.15] | – |
| P/F ratio (3 cols) | +0.000 [0.04] | – |
| notes | – | −0.023 [0.94] |

stacker coefficients b₁ (TabPFN) 0.55, b₂ (UTDE-SOFA) 0.37. MeanBP in UTDE = MeanBP + MeanABP channels (shown: MeanBP).

## fused-logit attribution, both sides occlusion (top 15)

| rank | input | mean abs attribution |
|---|---|---|
| 1 | gcs (T) | 0.420 |
| 2 | notes (U) | 0.242 |
| 3 | age (T) | 0.236 |
| 4 | GCS (U) | 0.230 |
| 5 | urine (T) | 0.185 |
| 6 | Age (U) | 0.175 |
| 7 | lactate (T) | 0.151 |
| 8 | Urine (U) | 0.150 |
| 9 | map (T) | 0.137 |
| 10 | Vent (U) | 0.123 |
| 11 | Lactate (U) | 0.106 |
| 12 | MeanBP (U) | 0.101 |
| 13 | vasopressor (T) | 0.092 |
| 14 | sofa_scores (T) | 0.084 |
| 15 | PaO2 (U) | 0.077 |
