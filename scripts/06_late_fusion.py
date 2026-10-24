"""Late fusion of TabPFN (structured) and UTDE (time series + notes) for 28-day mortality.

Inputs
  results/tabpfn/tabpfn_preds_W{W}.parquet                TabPFN val/test probabilities (04_tabpfn.py)
  ../MultimodalMIMIC/Data/sofa28/W{W}/{val,test}p2x_data.pkl   instance order of the UTDE loaders (field `name` = ICUSTAY_ID)
  <utde_run_dir>/test_logits_{seed}.npz and the checkpoint's stored validation logits   UTDE probabilities
Fusion = logistic regression on [logit(p_tabpfn), logit(p_utde)] fitted on the validation split (no test labels touched),
plus the simple average as a reference. Reports test AUROC / AUPRC with paired-bootstrap differences vs each single model.
Usage: python scripts/06_late_fusion.py --window 24 --utde-run ../MultimodalMIMIC/run/sofa28_W24/ihm/TS_Text/... --seed 42
"""
import argparse
import glob
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def paired_boot(y, p_a, p_b, n=2000, seed=0):
    rng = np.random.RandomState(seed); d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    return float(np.mean(d)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, required=True)
    ap.add_argument('--utde-run', required=True, help='UTDE checkpoint directory (args.ck_file_path) containing auc/<seed>.pth.tar and test_logits_<seed>.npz')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data', default=None, help='UTDE data dir (default ../MultimodalMIMIC/Data/sofa28/W{W})')
    args = ap.parse_args()
    W = args.window
    data_dir = Path(args.data or ROOT.parent / 'MultimodalMIMIC' / 'Data' / 'sofa28' / f'W{W}')
    run = Path(args.utde_run)

    tab = pd.read_parquet(ROOT / 'results' / 'tabpfn' / f'tabpfn_preds_W{W}.parquet')
    out = {}
    frames = {}
    for split in ['val', 'test']:
        names = [d['name'] for d in pickle.load(open(data_dir / f'{split}p2x_data.pkl', 'rb'))]
        if split == 'test':
            z = np.load(run / f'test_logits_{args.seed}.npz'); p_utde, y_utde = z['logits'], z['labels']
        else:
            ck = torch.load(glob.glob(str(run / '*' / f'{args.seed}.pth.tar'))[0], map_location='cpu', weights_only=False)   # our own checkpoint
            p_utde = np.array(ck['logits']); y_utde = None
        assert len(p_utde) == len(names), f'{split}: UTDE wrote {len(p_utde)} probabilities for {len(names)} instances'
        t = tab.loc[names]
        if y_utde is not None:
            assert (t.y28.to_numpy() == y_utde).all(), 'label order mismatch between TabPFN table and UTDE output'
        frames[split] = pd.DataFrame({'y': t.y28.to_numpy(), 'tabpfn': t['p_tabpfn'].to_numpy(), 'utde': p_utde}, index=names)
    va, te = frames['val'], frames['test']
    eps = 1e-6
    Zva = np.c_[logit(va.tabpfn.clip(eps, 1 - eps)), logit(va.utde.clip(eps, 1 - eps))]
    Zte = np.c_[logit(te.tabpfn.clip(eps, 1 - eps)), logit(te.utde.clip(eps, 1 - eps))]
    stack = LogisticRegression(C=1.0, max_iter=1000).fit(Zva, va.y)
    preds = {'TabPFN (tier 4 + age/sex)': te.tabpfn.to_numpy(), 'UTDE (TS + notes)': te.utde.to_numpy(),
             'late fusion: mean of probabilities': (te.tabpfn.to_numpy() + te.utde.to_numpy()) / 2,
             'late fusion: logistic stacker (fit on val)': stack.predict_proba(Zte)[:, 1]}
    y = te.y.to_numpy()
    md = [f'# Late fusion, W = {W} h (test n = {len(te):,}, 28-day mortality {y.mean():.3f}; stacker weights '
          f'tabpfn {stack.coef_[0][0]:.2f}, utde {stack.coef_[0][1]:.2f})\n\n| model | test AUROC | test AUPRC | ΔAUROC vs TabPFN [95 % CI] | ΔAUROC vs UTDE [95 % CI] |\n|---|---|---|---|---|\n']
    for name, p in preds.items():
        d_t = paired_boot(y, p, preds['TabPFN (tier 4 + age/sex)']); d_u = paired_boot(y, p, preds['UTDE (TS + notes)'])
        out[name] = {'auroc': float(roc_auc_score(y, p)), 'auprc': float(average_precision_score(y, p)), 'd_vs_tabpfn': d_t, 'd_vs_utde': d_u}
        md.append(f"| {name} | {out[name]['auroc']:.3f} | {out[name]['auprc']:.3f} | {d_t[0]:+.3f} [{d_t[1][0]:+.3f}, {d_t[1][1]:+.3f}] | {d_u[0]:+.3f} [{d_u[1][0]:+.3f}, {d_u[1][1]:+.3f}] |\n")
    od = ROOT / 'results' / 'late_fusion'; od.mkdir(exist_ok=True)
    json.dump(out, open(od / f'metrics_W{W}.json', 'w'), indent=1); (od / f'summary_W{W}.md').write_text(''.join(md))
    te.assign(**{k: v for k, v in preds.items()}).to_parquet(od / f'test_preds_W{W}.parquet')
    print(''.join(md))


if __name__ == '__main__':
    main()
