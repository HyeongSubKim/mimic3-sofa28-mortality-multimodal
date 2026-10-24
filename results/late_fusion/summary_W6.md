# Late fusion, W = 6 h (test n = 853, 28-day mortality 0.290; stacker weights tabpfn 0.55, utde 0.54)

| model | test AUROC | test AUPRC | ΔAUROC vs TabPFN [95 % CI] | ΔAUROC vs UTDE [95 % CI] |
|---|---|---|---|---|
| TabPFN (tier 4 + age/sex) | 0.832 | 0.681 | +0.000 [+0.000, +0.000] | +0.034 [+0.003, +0.066] |
| UTDE (TS + notes) | 0.798 | 0.618 | -0.034 [-0.066, -0.003] | +0.000 [+0.000, +0.000] |
| late fusion: mean of probabilities | 0.844 | 0.703 | +0.013 [-0.004, +0.029] | +0.047 [+0.030, +0.064] |
| late fusion: logistic stacker (fit on val) | 0.845 | 0.704 | +0.014 [-0.002, +0.029] | +0.048 [+0.030, +0.066] |
