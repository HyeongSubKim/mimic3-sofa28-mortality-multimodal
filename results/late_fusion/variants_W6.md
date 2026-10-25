# Fusion variants, W = 6 h (test n = 853, 28-day mortality 0.290); stacker fitted on validation

| model | AUROC | AUPRC | ΔAUROC vs TabPFN [95 % CI] |
|---|---|---|---|
| TabPFN (structured) | 0.832 | 0.681 | +0.000 [+0.000, +0.000] |
| UTDE time series only | 0.797 | 0.642 | -0.035 [-0.064, -0.006] |
| late fusion TabPFN + UTDE time series only | 0.841 | 0.697 | +0.009 [-0.003, +0.020] |
| UTDE time series + notes | 0.798 | 0.618 | -0.034 [-0.066, -0.003] |
| late fusion TabPFN + UTDE time series + notes | 0.845 | 0.704 | +0.014 [-0.002, +0.029] |
| UTDE with the tabular variable set (15 channels: SOFA inputs, lactate, BE, vent, age, sex) + notes | 0.823 | 0.658 | -0.009 [-0.031, +0.013] |
| late fusion TabPFN + UTDE with the tabular variable set (15 channels: SOFA inputs, lactate, BE, vent, age, sex) + notes | 0.839 | 0.688 | +0.008 [-0.008, +0.023] |
| TabPFN-embedding UTDE, 3 streams (TabPFN tokens + time series + notes) | 0.836 | 0.700 | +0.004 [-0.012, +0.020] |
