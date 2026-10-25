# ICU 28-day mortality on MIMIC-III: TabPFN, UTDE and their late fusion

Structured early-ICU data (TabPFN v2), irregular time series + clinical notes (UTDE, ICML 2023), and a late fusion of
the two, on one cohort: adult ICU stays in MIMIC-III v1.4 with a computable SOFA score, lactate and base excess in the
first **W = 6 h / 24 h**, alive at hour W; outcome = death from any cause within 28 days of ICU admission.

| model | input | 6 h AUROC (AUPRC) | 24 h AUROC (AUPRC) | ΔAUROC vs TabPFN, 24 h [95 % CI] |
|---|---|---|---|---|
| TabPFN v2 | 68 structured features (SOFA inputs, lactate, BE, age, sex) | 0.832 (.681) | 0.837 (.673) | – |
| UTDE | 21 hourly channels + last 5 notes (Clinical-Longformer) | 0.798 (.618) | 0.844 (.680) | +0.007 [−0.013, +0.026] |
| **late fusion** (logistic stacker on the two logits, fitted on validation) | both | **0.845 (.704)** | **0.860 (.707)** | **+0.023 [+0.010, +0.035]** |
| TabPFN + vital-sign aggregates (fair structured baseline, 113 features) | structured incl. HR / RR / SpO2 / Temp / BP | 0.842 (.705) | 0.852 (.684) | +0.015 [+0.007, +0.023] |
| TabPFN-in-UTDE, 3 streams (frozen TabPFN tokens + time series + notes) | both, one network | 0.836 (.700) | **0.868 (.731)** | **+0.031 [+0.015, +0.046]** |

Test split: 853 stays (6 h) / 1,374 stays (24 h); patient-level 70/15/15 split shared by all models. Against the fair
structured baseline (TabPFN + the vital signs UTDE sees) the multimodal gain is +0.002 (6 h, ns) / +0.012 (24 h) for the
late fusion and +0.016 (24 h) for the 3-stream model; see `results/late_fusion/variants_W*.md` and
`results/tabpfn_vitals/summary_W*.md` (scripts 09–11).
Full tables with confidence intervals: `results/late_fusion/summary_W{6,24}.md`; attribution of the three models
(stacker SHAP, TabPFN permutation SHAP, UTDE channel occlusion): `results/fusion_shap/`.

## Pipeline

```
MIMIC-III csv.gz ──01──> data/*.parquet ──03──> results/features/features_W{W}.parquet ──04──> TabPFN probabilities ─┐
                  └─02──> data/noteevents_text.parquet ──05──> UTDE pickles ──(utde/ + run_sofa28.sh)──> UTDE probabilities ┴─06──> late fusion
```

| step | script | env | what it does |
|---|---|---|---|
| 1 | `scripts/01_extract_events.py` | base | ICUSTAYS/PATIENTS/ADMISSIONS + the needed CHARTEVENTS / INPUTEVENTS / OUTPUTEVENTS rows → `data/*.parquet` |
| 2 | `scripts/02_extract_notes.py` | base | NOTEEVENTS of those admissions (no discharge summaries, de-id tokens stripped) → `data/noteevents_text.parquet` |
| 3 | `scripts/03_build_features.py` | base | cohort, split, 68 features per stay and window → `results/features/` |
| 4 | `scripts/04_tabpfn.py` | tabpfn | TabPFN v2 (open checkpoint, no tuning) → `results/tabpfn/` |
| 5 | `scripts/05_build_utde_dataset.py` | base | hourly z-scored channels + time-gated notes → `../MultimodalMIMIC/Data/sofa28/W{W}/*.pkl` |
| – | `utde/setup_utde.sh`, then `run_sofa28.sh` | utde | reference UTDE code @ `XZhang97666/MultimodalMIMIC` 8e513ca + our 7-file adapter patch; training with the original hyper-parameters |
| 6 | `scripts/06_late_fusion.py` | tabpfn | stacker on validation predictions, paired bootstrap on test → `results/late_fusion/` |
| 7–8 | `scripts/07_fusion_shap.py`, `08_fusion_shap_figs.py` (+ `utde_attr.py` from the patch) | tabpfn / utde | optional: SHAP / occlusion attribution and figures |
| 9–11 | `scripts/09_tabpfn_embeddings.py`, `10_fusion_variants.py`, `11_tabpfn_vitals.py` (+ `run_tabemb.sh`, `--tab_emb_mode concat` in the patch; `05 --channel-set sofa`) | tabpfn / utde | optional: frozen TabPFN tokens inside UTDE (3-stream), variable-fair controls |

Step-by-step commands, environments, run times and pitfalls: **`docs/REPRODUCE.md`**.

## Requirements

* MIMIC-III v1.4 (PhysioNet, credentialed access, including NOTEEVENTS). No patient-derived file is in this repository.
* `requirements.txt` (base, Python ≥ 3.10), `requirements-tabpfn.txt` (TabPFN 2 / torch 2.6), `requirements-utde.txt`
  (the reference UTDE code needs Python 3.8, torch 1.13, transformers 4.16).
* One GPU with ≥ 24 GB for UTDE training (~3 h per window on an A6000); TabPFN needs a few GB.

## License

Apache License 2.0 (`LICENSE`). MIMIC-III data remain under the PhysioNet Credentialed Health Data Use Agreement. The
UTDE reference implementation is an external dependency applied through the patch in `utde/` and keeps its own terms.
