# Late fusion, W = 24 h (test n = 1,374, 28-day mortality 0.258; stacker weights tabpfn 0.41, utde 0.43)

| model | test AUROC | test AUPRC | ΔAUROC vs TabPFN [95 % CI] | ΔAUROC vs UTDE [95 % CI] |
|---|---|---|---|---|
| TabPFN (tier 4 + age/sex) | 0.837 | 0.673 | +0.000 [+0.000, +0.000] | -0.007 [-0.026, +0.013] |
| UTDE (TS + notes) | 0.844 | 0.680 | +0.007 [-0.013, +0.026] | +0.000 [+0.000, +0.000] |
| late fusion: mean of probabilities | 0.860 | 0.707 | +0.023 [+0.011, +0.035] | +0.017 [+0.008, +0.025] |
| late fusion: logistic stacker (fit on val) | 0.860 | 0.707 | +0.023 [+0.010, +0.035] | +0.016 [+0.008, +0.024] |
