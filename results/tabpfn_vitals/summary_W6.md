# TabPFN with the extra vital-sign aggregates, W = 6 h (test n = 853)

| model | AUROC | AUPRC | Δ vs TabPFN (68) | Δ vs TabPFN + vitals |
|---|---|---|---|---|
| TabPFN (68 structured) | 0.832 | 0.681 | +0.000 [+0.000, +0.000] | -0.010 [-0.024, +0.003] |
| TabPFN + vitals aggregates | 0.842 | 0.705 | +0.010 [-0.003, +0.024] | +0.000 [+0.000, +0.000] |
| UTDE 21 ch + notes | 0.798 | 0.618 | -0.034 [-0.066, -0.003] | -0.044 [-0.072, -0.016] |
| late fusion TabPFN + UTDE | 0.845 | 0.704 | +0.014 [-0.002, +0.029] | +0.004 [-0.011, +0.019] |
| late fusion TabPFN+vitals + UTDE | 0.844 | 0.710 | +0.012 [-0.009, +0.032] | +0.002 [-0.012, +0.016] |
| 3-stream TabPFN-embedding UTDE | 0.836 | 0.700 | +0.004 [-0.012, +0.020] | -0.006 [-0.017, +0.004] |
