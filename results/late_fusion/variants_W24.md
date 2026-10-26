# Fusion variants, W = 24 h (test n = 1374, 28-day mortality 0.258); stacker fitted on validation

| model | AUROC | AUPRC | ΔAUROC vs TabPFN [95 % CI] |
|---|---|---|---|
| TabPFN (structured) | 0.837 | 0.673 | +0.000 [+0.000, +0.000] |
| UTDE time series only | 0.827 | 0.643 | -0.009 [-0.029, +0.011] |
| late fusion TabPFN + UTDE time series only | 0.850 | 0.683 | +0.013 [+0.003, +0.024] |
| UTDE time series + notes | 0.844 | 0.680 | +0.007 [-0.013, +0.026] |
| late fusion TabPFN + UTDE time series + notes | 0.860 | 0.707 | +0.023 [+0.010, +0.035] |
| UTDE with the tabular variable set (15 channels: SOFA inputs, lactate, BE, vent, age, sex) + notes | 0.826 | 0.658 | -0.011 [-0.025, +0.002] |
| late fusion TabPFN + UTDE with the tabular variable set (15 channels: SOFA inputs, lactate, BE, vent, age, sex) + notes | 0.841 | 0.680 | +0.003 [-0.002, +0.009] |
| TabPFN-embedding UTDE (TabPFN tokens + notes) | 0.836 | 0.671 | -0.001 [-0.006, +0.004] |
| TabPFN-embedding UTDE, 3 streams (TabPFN tokens + time series + notes) | 0.868 | 0.731 | +0.031 [+0.015, +0.046] |
| TabPFN-embedding UTDE, 3 streams, tabular variable set (TabPFN tokens + 15-ch time series + notes) | 0.856 | 0.709 | +0.019 [+0.008, +0.030] |
