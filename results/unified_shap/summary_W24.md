# Attribution, unified variable set, W = 24 h

## 1. TabPFN (68 features) — permutation SHAP, mean |SHAP| (probability units), top 10

| rank | feature | mean abs SHAP |
|---|---|---|
| 1 | `gcs_last` | 0.1110 |
| 2 | `age` | 0.0482 |
| 3 | `urine_n` | 0.0206 |
| 4 | `gcs_count` | 0.0202 |
| 5 | `lactate_min` | 0.0193 |
| 6 | `lactate_last` | 0.0182 |
| 7 | `base_excess_last` | 0.0166 |
| 8 | `gcs_min` | 0.0157 |
| 9 | `urine_total` | 0.0147 |
| 10 | `sofa_coag` | 0.0136 |

(figure: `results/fusion_shap/tabpfn_shap_W24.png`)

## 2. UTDE-SOFA (15 channels + notes) — occlusion

| full | without notes | without time series | notes only vs TS only |
|---|---|---|---|
| 0.826 | 0.803 (−0.023) | 0.643 (−0.183) | 0.643 vs 0.803 |

| channel | mean abs Δlogit when occluded | AUROC drop |
|---|---|---|
| GCS | 0.624 | +0.057 |
| Age | 0.475 | +0.020 |
| Urine | 0.408 | +0.005 |
| Vent | 0.333 | +0.003 |
| Lactate | 0.287 | +0.008 |
| MeanBP | 0.274 | +0.005 |
| PaO2 | 0.208 | +0.008 |
| MeanABP | 0.158 | +0.003 |
| BaseExcess | 0.141 | +0.003 |
| FiO2 | 0.135 | +0.004 |
| Bilirubin | 0.113 | +0.007 |
| Platelets | 0.111 | +0.002 |
| Creatinine | 0.086 | -0.001 |
| Sex | 0.071 | -0.000 |
| Vasopressor | 0.058 | +0.002 |

## 3. Late fusion TabPFN + UTDE-SOFA — exact stacker SHAP

stacker coefficients TabPFN 0.55, UTDE-SOFA 0.37; mean |SHAP| TabPFN 0.71, UTDE-SOFA 0.45 (UTDE share 39%); logit correlation 0.91; fusion test AUROC 0.841

| rank | input | mean abs attribution (fused logit) |
|---|---|---|
| 1 | gcs_last (T) | 0.798 |
| 2 | age (T) | 0.315 |
| 3 | notes (U), offset removed | 0.242 |
| 4 | GCS (U) | 0.230 |
| 5 | Age (U) | 0.175 |
| 6 | Urine (U) | 0.150 |
| 7 | urine_n (T) | 0.145 |
| 8 | Vent (U) | 0.123 |
| 9 | gcs_count (T) | 0.108 |
| 10 | Lactate (U) | 0.106 |
| 11 | MeanBP (U) | 0.101 |
| 12 | lactate_last (T) | 0.100 |
| 13 | gcs_min (T) | 0.096 |
| 14 | lactate_min (T) | 0.093 |
| 15 | urine_total (T) | 0.093 |

## 4. TabPFN-in-UTDE, replace (TabPFN tokens + notes) — occlusion

| full | without TabPFN tokens | without notes | without TabPFN tokens and notes |
|---|---|---|---|
| 0.836 | 0.562 (−0.274) | 0.834 (−0.002) | 0.494 |

mean |Δlogit| when removed: TabPFN tokens 2.00, notes 0.65

## 4. TabPFN-in-UTDE, 3 streams (TabPFN tokens + 15-ch time series + notes) — occlusion

| full | without TabPFN tokens | without notes | without TabPFN tokens and notes | without time series |
|---|---|---|---|---|
| 0.856 | 0.765 (−0.091) | 0.840 (−0.016) | 0.752 | 0.852 (−0.005) |

mean |Δlogit| when removed: TabPFN tokens 1.27, notes 0.94, time series 0.29

| channel | mean abs Δlogit | AUROC drop |
|---|---|---|
| Vent | 0.161 | +0.000 |
| Age | 0.108 | -0.002 |
| GCS | 0.099 | +0.006 |
| PaO2 | 0.077 | +0.003 |
| Bilirubin | 0.076 | +0.001 |
| FiO2 | 0.064 | -0.000 |
| Urine | 0.060 | +0.000 |
| Sex | 0.054 | +0.000 |
| BaseExcess | 0.052 | +0.000 |
| MeanABP | 0.050 | +0.002 |
| Lactate | 0.050 | -0.000 |
| Vasopressor | 0.047 | +0.001 |
| Platelets | 0.040 | -0.001 |
| MeanBP | 0.037 | +0.000 |
| Creatinine | 0.037 | -0.001 |

