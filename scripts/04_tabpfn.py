"""TabPFN v2 on the structured features (all 68 columns = SOFA inputs + lactate + base excess + age/sex), per window.

Context = train + val (no tuning); test probabilities from that fit, validation probabilities from a train-only fit (used
later by the fusion stacker). Uses the openly hosted checkpoint Prior-Labs/TabPFN-v2-clf (no token needed) if it is in
$HF_HOME, otherwise the library default. Run in an environment with `tabpfn` + torch (requirements-tabpfn.txt).

Usage: HF_HOME=... python scripts/04_tabpfn.py [--windows 6 24] [--device cuda]
Output: results/tabpfn/tabpfn_preds_W{W}.parquet (SUBJECT_ID, dataset, y28, p_tabpfn for val + test), tabpfn_metrics.json, summary.md
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
META = ['SUBJECT_ID', 'dataset', 'y28']


def boot_ci(y, p, n=1000, seed=0):
    rng = np.random.RandomState(seed); y = np.asarray(y); p = np.asarray(p); a, b = [], []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a.append(roc_auc_score(y[i], p[i])); b.append(average_precision_score(y[i], p[i]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))], [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, nargs='+', default=[6, 24])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n-estimators', type=int, default=8, help='TabPFN ensemble members (library default)')
    args = ap.parse_args()
    OUT = ROOT / 'results' / 'tabpfn'; OUT.mkdir(parents=True, exist_ok=True)
    hf = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface'))
    ckpts = sorted(glob.glob(str(hf / 'hub/models--Prior-Labs--TabPFN-v2-clf/snapshots/*/tabpfn-v2-classifier.ckpt')))
    model_path = ckpts[0] if ckpts else None
    print('checkpoint:', model_path or 'library default', flush=True)

    results, md = {}, ['# TabPFN v2, structured features (SOFA inputs + lactate + base excess + age/sex)\n\n| W | train+val | test | test AUROC [95 % CI] | test AUPRC [95 % CI] | time |\n|---|---|---|---|---|---|\n']
    for W in args.windows:
        df = pd.read_parquet(ROOT / 'results' / 'features' / f'features_W{W}.parquet')
        feats = [c for c in df.columns if c not in META]
        X = df[feats].to_numpy(dtype='float32'); y = df.y28.to_numpy()
        ctx = df.dataset.isin(['train', 'val']).to_numpy(); trn = (df.dataset == 'train').to_numpy()
        val = (df.dataset == 'val').to_numpy(); te = (df.dataset == 'test').to_numpy()
        t0 = time.time()
        p = TabPFNClassifier(device=args.device, model_path=model_path, n_estimators=args.n_estimators, random_state=0).fit(X[ctx], y[ctx]).predict_proba(X[te])[:, 1]
        pv = TabPFNClassifier(device=args.device, model_path=model_path, n_estimators=args.n_estimators, random_state=0).fit(X[trn], y[trn]).predict_proba(X[val])[:, 1]
        dt = time.time() - t0
        preds = df[META].copy(); col = np.full(len(df), np.nan); col[te] = p; col[val] = pv; preds['p_tabpfn'] = col
        preds.to_parquet(OUT / f'tabpfn_preds_W{W}.parquet')
        auc_ci, ap_ci = boot_ci(y[te], p)
        results[W] = {'n_context': int(ctx.sum()), 'n_test': int(te.sum()), 'n_features': len(feats), 'test_auroc': float(roc_auc_score(y[te], p)),
                      'test_auroc_ci95': auc_ci, 'test_auprc': float(average_precision_score(y[te], p)), 'test_auprc_ci95': ap_ci, 'seconds': dt}
        r = results[W]
        print(f"W={W} h: {len(feats)} features, AUROC {r['test_auroc']:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}], AUPRC {r['test_auprc']:.3f}, {dt:.0f} s", flush=True)
        md.append(f"| {W} h | {r['n_context']:,} | {r['n_test']:,} | {r['test_auroc']:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}] | {r['test_auprc']:.3f} [{ap_ci[0]:.3f}, {ap_ci[1]:.3f}] | {dt:.0f} s |\n")
    json.dump(results, open(OUT / 'tabpfn_metrics.json', 'w'), indent=1); (OUT / 'summary.md').write_text(''.join(md))
    print(''.join(md))


if __name__ == '__main__':
    main()
