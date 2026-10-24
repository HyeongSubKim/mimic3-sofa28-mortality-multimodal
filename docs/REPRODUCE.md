# Reproduction guide

## 0. Setup

```bash
export MIMIC3_RAW=/path/to/mimic-iii-clinical-database-1.4      # the PhysioNet csv.gz files
export HF_HOME=$HOME/.cache/huggingface                         # pre-trained weights are cached here

python -m venv .venv-base   && .venv-base/bin/pip   install -r requirements.txt
python -m venv .venv-tabpfn && .venv-tabpfn/bin/pip install -r requirements-tabpfn.txt
# UTDE: python 3.8 environment (conda/micromamba), torch first, then the pins
pip install torch==1.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117 && pip install -r requirements-utde.txt

huggingface-cli download Prior-Labs/TabPFN-v2-clf     # open checkpoint, no token
huggingface-cli download yikuan8/Clinical-Longformer
```

Set `PYTHONNOUSERSITE=1` in the UTDE environment so packages from `~/.local` cannot leak in (transformers 4.16 breaks
with newer tokenizers / huggingface_hub).

## 1. Data and features (base env, ~1 h; CHARTEVENTS is read once, ~30 min)

```bash
python scripts/01_extract_events.py
python scripts/02_extract_notes.py
python scripts/03_build_features.py --windows 6 24
#  -> results/features/features_W6.parquet (5,757 stays), features_W24.parquet (9,229 stays)
#     columns: 63 features, age, sex, SUBJECT_ID, dataset (train/val/test), y28
```

## 2. TabPFN (tabpfn env, ~10 min)

```bash
HF_HOME=... python scripts/04_tabpfn.py --windows 6 24
#  -> results/tabpfn/tabpfn_preds_W{W}.parquet (p_tabpfn for val + test), tabpfn_metrics.json, summary.md
#     expected test AUROC 0.832 (6 h) / 0.837 (24 h)
```

## 3. UTDE (base env for the data, utde env for training; ~3 h per window on one A6000)

```bash
python scripts/05_build_utde_dataset.py --windows 6 24 --out ../MultimodalMIMIC/Data/sofa28
bash utde/setup_utde.sh ../MultimodalMIMIC          # clone reference code @ 8e513ca and apply utde/MultimodalMIMIC_sofa28.patch
cd ../MultimodalMIMIC
GPU=0 W=24 PY=/path/to/utde-env/python bash run_sofa28.sh
GPU=0 W=6  PY=/path/to/utde-env/python bash run_sofa28.sh
#  -> run/sofa28_W{W}/ihm/TS_Text/.../2e-05_2_3_0.0004_6_8_128_1_2/auc/42.pth.tar (best validation-AUROC epoch)
#     and test_logits_42.npz next to it; expected test AUROC 0.798 (6 h) / 0.844 (24 h)
```

Data format (per stay): 21 hourly channels (HR, RR, SpO2, Temp, SBP, DBP, MeanBP, ASBP, ADBP, MeanABP, FiO2, O2 flow,
PaO2, platelets, bilirubin, creatinine, lactate, base excess, GCS, urine, vasopressor), z-scored with train statistics,
rows = hours with ≥ 1 value; notes available by INTIME + W (STORETIME for bedside notes, CHARTTIME for radiology; no
discharge summaries), time-ordered; UTDE uses the last 5, 1,024 tokens each, with their timestamps. Stays without a note
get one placeholder note so the cohort is identical to the tabular one.

What the adapter patch changes in the reference code (model, loss and training loop untouched): channel count as an
argument (`--orig_d_ts 21`), checkpoint selection by validation AUROC (`--select_metric auc`), dumping test
probabilities, loading Clinical-Longformer from a local snapshot (`CLINICAL_LONGFORMER_PATH`), list-typed arrays in the
pickles, an optional term-attention flag (unused here), and the run / attribution scripts.

## 4. Late fusion (tabpfn env, seconds)

```bash
RUN=../MultimodalMIMIC/run/sofa28_W24/ihm/TS_Text/TS_24/Atten/Text_24/bioLongformer/1024/cross_attn3/batch_seq_feature/irregular_TS_64/irregular_Text_64/2e-05_2_3_0.0004_6_8_128_1_2/
python scripts/06_late_fusion.py --window 24 --utde-run $RUN --seed 42      # and --window 6 with the W6 run dir
#  -> results/late_fusion/summary_W{W}.md, metrics_W{W}.json, test_preds_W{W}.parquet
#     expected test AUROC 0.845 (6 h) / 0.860 (24 h)
```

The stacker (logistic regression on `[logit p_TabPFN, logit p_UTDE]`) is fitted on the validation split only — TabPFN
validation probabilities come from a train-only fit, UTDE validation logits are read from the checkpoint — and applied
to the test split; differences are paired-bootstrap ΔAUROC (2,000 resamples).

## 5. Optional: attribution

```bash
HF_HOME=... python scripts/07_fusion_shap.py --window 24         # stacker SHAP (exact) + TabPFN permutation SHAP (~5 h on a shared GPU; --max-cases 300 for a draft)
cd ../MultimodalMIMIC && GPU=0 W=24 PY=... ATTR_OUT=../<this repo>/results/fusion_shap bash run_attr.sh   # UTDE modality / channel occlusion (~1 h)
python scripts/08_fusion_shap_figs.py --window 6 24               # -> results/fusion_shap/*.png
```

## Pitfalls

* The pickles store plain lists on purpose: numpy-2 pickles cannot be read by the numpy 1.21 of the UTDE environment.
* Keep `data/` and every per-stay file (`*.parquet`, `*.npz`, `*.csv` under `results/`) out of any public copy — the
  `.gitignore` does this; only aggregate tables, metrics and figures are tracked.
* `03` caches two raw subsets (`data/labevents_sofa.parquet`, `data/chartevents_gcs.parquet`) on its first run.
