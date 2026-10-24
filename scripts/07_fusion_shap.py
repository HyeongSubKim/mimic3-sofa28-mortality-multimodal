"""SHAP for the best multimodal model (late fusion TabPFN + UTDE) and for its components.

1. Fusion stacker: logit(p_fusion) = b0 + b1 logit(p_tabpfn) + b2 logit(p_utde) -> exact (linear) SHAP per case:
   phi_k = b_k (x_k - mean_k).  Coefficients are recovered from the saved predictions (results/late_fusion/test_preds_W{W}.parquet).
2. TabPFN: refit exactly as in 04_tabpfn.py (context = train + val), then permutation SHAP
   (shap.PermutationExplainer, 10 background rows, 2F+1 evaluations per case) on the test split -> per-feature values.
3. UTDE: modality-level Shapley and channel occlusion are produced by utde_attr.py (in the adapter patch) (deep model, no
   feature-level SHAP); this script only reads them if present.
Outputs results/fusion_shap/: fusion_shap_W{W}.csv, tabpfn_shap_W{W}.npz, summary_W{W}.md
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.special import logit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'fusion_shap'; OUT.mkdir(parents=True, exist_ok=True)
META = {'SUBJECT_ID', 'dataset', 'y28'}
LG = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--background', type=int, default=10)
    ap.add_argument('--max-cases', type=int, default=0, help='explain only the first n test cases (0 = all)')
    args = ap.parse_args(); W = args.window
    hf = Path(os.environ.get('HF_HOME', '/mnt/nas5/AIBL-Research/hskim/.cache/huggingface'))
    ckpt = sorted(glob.glob(str(hf / 'hub/models--Prior-Labs--TabPFN-v2-clf/snapshots/*/tabpfn-v2-classifier.ckpt')))[0]

    # ---- 1. fusion stacker (exact linear SHAP in logit space)
    tp = pd.read_parquet(ROOT / 'results' / 'late_fusion' / f'test_preds_W{W}.parquet')
    Xf = np.c_[LG(tp.tabpfn.values), LG(tp.utde.values)]; yf = LG(tp['late fusion: logistic stacker (fit on val)'].values)
    lr = LinearRegression().fit(Xf, yf); assert lr.score(Xf, yf) > 0.9999, 'stacker is not linear in the logits?'
    phi = lr.coef_ * (Xf - Xf.mean(0))
    fus = pd.DataFrame({'y': tp.y.values, 'logit_tabpfn': Xf[:, 0], 'logit_utde': Xf[:, 1], 'shap_tabpfn': phi[:, 0], 'shap_utde': phi[:, 1],
                        'p_fusion': tp['late fusion: logistic stacker (fit on val)'].values})
    fus.to_csv(OUT / f'fusion_shap_W{W}.csv', index=False)
    agree = np.sign(Xf[:, 0] - np.median(Xf[:, 0])) == np.sign(Xf[:, 1] - np.median(Xf[:, 1]))
    fusion = {'coef_tabpfn': float(lr.coef_[0]), 'coef_utde': float(lr.coef_[1]), 'intercept': float(lr.intercept_),
              'mean_abs_shap_tabpfn': float(np.abs(phi[:, 0]).mean()), 'mean_abs_shap_utde': float(np.abs(phi[:, 1]).mean()),
              'share_tabpfn': float(np.abs(phi[:, 0]).sum() / np.abs(phi).sum()),
              'corr_logits': float(np.corrcoef(Xf[:, 0], Xf[:, 1])[0, 1]), 'disagree_frac': float(1 - agree.mean()),
              'auroc_fusion_agree': float(roc_auc_score(tp.y[agree], fus.p_fusion[agree])), 'auroc_fusion_disagree': float(roc_auc_score(tp.y[~agree], fus.p_fusion[~agree])),
              'auroc_tabpfn_disagree': float(roc_auc_score(tp.y[~agree], tp.tabpfn[~agree])), 'auroc_utde_disagree': float(roc_auc_score(tp.y[~agree], tp.utde[~agree]))}
    print('fusion', json.dumps(fusion, indent=0), flush=True)

    # ---- 2. TabPFN permutation SHAP
    if (OUT / f'tabpfn_shap_W{W}.npz').exists() and not args.max_cases:
        print('tabpfn SHAP already computed, skipping'); return
    df = pd.read_parquet(ROOT / 'results' / 'features' / f'features_W{W}.parquet')
    feats = [c for c in df.columns if c not in META]
    ctx = df.dataset.isin(['train', 'val']).to_numpy(); te = (df.dataset == 'test').to_numpy()
    X = df[feats].to_numpy(dtype='float32'); y = df.y28.to_numpy()
    clf = TabPFNClassifier(device=args.device, model_path=ckpt, n_estimators=8, random_state=0).fit(X[ctx], y[ctx])
    p_te = clf.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], p_te); print(f'TabPFN refit test AUROC {auc:.4f} (expected 0.832 / 0.837)', flush=True)
    f = lambda Z: clf.predict_proba(Z.astype('float32'))[:, 1]
    rng = np.random.RandomState(0); bg = X[ctx][rng.choice(ctx.sum(), args.background, replace=False)]
    expl = shap.PermutationExplainer(f, shap.maskers.Independent(bg, max_samples=args.background), seed=0)
    Xte = X[te] if not args.max_cases else X[te][:args.max_cases]
    sv = expl(Xte, max_evals=2 * len(feats) + 1, silent=True)
    S = np.asarray(sv.values)
    np.savez(OUT / f'tabpfn_shap_W{W}.npz', shap=S, X=Xte, base=np.asarray(sv.base_values), features=np.array(feats), y=y[te][:len(Xte)], p=p_te[:len(Xte)])
    imp = pd.DataFrame({'feature': feats, 'mean_abs_shap': np.abs(S).mean(0), 'mean_shap': S.mean(0)}).sort_values('mean_abs_shap', ascending=False)
    imp.to_csv(OUT / f'tabpfn_shap_W{W}.csv', index=False)
    # group by source variable (prefix before the last _stat) for a compact view
    grp = imp.assign(var=imp.feature.str.replace(r'_(min|max|mean|last|count|n|total|drugs|ml_per_h)$', '', regex=True)).groupby('var').mean_abs_shap.sum().sort_values(ascending=False)
    md = [f'# Late-fusion SHAP, W = {W} h (test n = {len(tp)})', '', '## 1. Fusion stacker (linear in logits): exact SHAP', '',
          f"coefficients: TabPFN {fusion['coef_tabpfn']:.3f}, UTDE {fusion['coef_utde']:.3f}; mean |SHAP| TabPFN {fusion['mean_abs_shap_tabpfn']:.3f}, UTDE {fusion['mean_abs_shap_utde']:.3f} "
          f"(TabPFN share {fusion['share_tabpfn']:.2f}); logit correlation {fusion['corr_logits']:.2f}; cases where the two disagree about the direction (vs their medians): {fusion['disagree_frac']:.2f} "
          f"— fusion AUROC {fusion['auroc_fusion_disagree']:.3f} on those (TabPFN {fusion['auroc_tabpfn_disagree']:.3f}, UTDE {fusion['auroc_utde_disagree']:.3f}) vs {fusion['auroc_fusion_agree']:.3f} where they agree.", '',
          f'## 2. TabPFN (tier 4 + age/sex) permutation SHAP, refit test AUROC {auc:.3f}, explained cases {len(Xte)}', '',
          '| rank | feature | mean abs SHAP (prob.) | mean SHAP |', '|---|---|---|---|']
    md += [f"| {i+1} | {r.feature} | {r.mean_abs_shap:.4f} | {r.mean_shap:+.4f} |" for i, r in enumerate(imp.head(20).itertuples())]
    md += ['', '### grouped by source variable (sum of mean abs SHAP over its aggregates)', '', '| variable | total mean abs SHAP |', '|---|---|']
    md += [f'| {k} | {v:.4f} |' for k, v in grp.head(15).items()]
    open(OUT / f'summary_W{W}.md', 'w').write('\n'.join(md) + '\n'); json.dump(fusion, open(OUT / f'fusion_W{W}.json', 'w'), indent=1)
    print('\n'.join(md), flush=True)


if __name__ == '__main__':
    main()
