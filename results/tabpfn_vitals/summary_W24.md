# TabPFN with the extra vital-sign aggregates, W = 24 h (test n = 1374)

| model | AUROC | AUPRC | Δ vs TabPFN (68) | Δ vs TabPFN + vitals |
|---|---|---|---|---|
| TabPFN (68 structured) | 0.837 | 0.673 | +0.000 [+0.000, +0.000] | -0.015 [-0.023, -0.007] |
| TabPFN + vitals aggregates | 0.852 | 0.684 | +0.015 [+0.007, +0.023] | +0.000 [+0.000, +0.000] |
| UTDE 21 ch + notes | 0.844 | 0.680 | +0.007 [-0.013, +0.026] | -0.008 [-0.026, +0.010] |
| late fusion TabPFN + UTDE | 0.860 | 0.707 | +0.023 [+0.010, +0.035] | +0.008 [-0.004, +0.019] |
| late fusion TabPFN+vitals + UTDE | 0.863 | 0.709 | +0.026 [+0.013, +0.039] | +0.012 [+0.002, +0.021] |
| 3-stream TabPFN-embedding UTDE | 0.868 | 0.731 | +0.031 [+0.015, +0.046] | +0.016 [+0.003, +0.028] |
