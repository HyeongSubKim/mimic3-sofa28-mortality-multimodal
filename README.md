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

## Unified variable set: TabPFN / UTDE / late fusion / TabPFN-in-UTDE on the same inputs (`05 --channel-set sofa`, scripts 09–12)

All models restricted to the tabular model's variables — SOFA inputs, lactate, base excess, ventilation, age, sex — i.e.
no HR / RR / SpO2 / temperature / blood-pressure channels for UTDE. Test AUROC (AUPRC), single seed:

| model | 6 h | Δ vs TabPFN | 24 h | Δ vs TabPFN |
|---|---|---|---|---|
| TabPFN (68 features) | 0.832 (.681) | – | 0.837 (.673) | – |
| UTDE-SOFA (15 channels + notes) | 0.823 (.658) | −0.009 [−0.031, +0.013] | 0.826 (.658) | −0.011 [−0.025, +0.002] |
| late fusion TabPFN + UTDE-SOFA | 0.839 (.688) | +0.008 [−0.008, +0.023] | 0.841 (.680) | +0.003 [−0.002, +0.009] |
| TabPFN-in-UTDE, replace (TabPFN tokens + notes) | 0.827 (.677) | −0.005 [−0.012, +0.002] | 0.836 (.671) | −0.001 [−0.006, +0.004] |
| **TabPFN-in-UTDE, 3 streams (TabPFN tokens + 15-ch time series + notes)** | 0.835 (.687) | +0.004 [−0.010, +0.016] | **0.856 (.709)** | **+0.019 [+0.008, +0.030]** |

Attribution on the same models (`results/unified_shap/summary_W{6,24}.md`, figures alongside):

| model | what drives it (24 h) | notes' share (AUROC when removed) |
|---|---|---|
| TabPFN | `gcs_last` ≫ `age` > urine / lactate / BE aggregates (permutation SHAP) | – |
| UTDE-SOFA | GCS (−0.057) > Age (−0.020) > lactate / PaO2 / bilirubin (occlusion); at 6 h Age ≈ GCS lead | −0.023 (6 h −0.018) |
| late fusion | TabPFN logit 61 % of the fused logit at 24 h (39 % at 6 h); GCS and age enter through both paths, logit corr. 0.91 | via the UTDE logit only |
| replace | TabPFN tokens −0.274 when removed; notes −0.002 — the note branch is not used | ≈ 0 |
| 3 streams | TabPFN tokens −0.091, notes −0.016, time-series stream −0.005; channels ≤ 0.006 each | −0.016 |

Reading: with identical variables the late fusion adds nothing significant, and a frozen TabPFN dropped into UTDE in
place of the time series learns to ignore the notes; keeping the time-series stream alongside the TabPFN tokens lets the
note branch train (notes −0.016 vs −0.002) and gives the only significant gain over TabPFN at 24 h (+0.019). At 6 h no
multimodal configuration beats TabPFN. The 21-channel 3-stream (0.868) is 0.012 higher than this one — that is the
contribution of the extra vital-sign channels.

## Requirements

* MIMIC-III v1.4 (PhysioNet, credentialed access, including NOTEEVENTS). No patient-derived file is in this repository.
* `requirements.txt` (base, Python ≥ 3.10), `requirements-tabpfn.txt` (TabPFN 2 / torch 2.6), `requirements-utde.txt`
  (the reference UTDE code needs Python 3.8, torch 1.13, transformers 4.16).
* One GPU with ≥ 24 GB for UTDE training (~3 h per window on an A6000); TabPFN needs a few GB.

## License

Apache License 2.0 (`LICENSE`). MIMIC-III data remain under the PhysioNet Credentialed Health Data Use Agreement. The
UTDE reference implementation is an external dependency applied through the patch in `utde/` and keeps its own terms.
