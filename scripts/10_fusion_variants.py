"""Late-fusion variants on the same test split: TabPFN with UTDE trained on time series only, with the full UTDE
(time series + notes), and with the TabPFN-embedding UTDE when its run exists. Stacker fitted on validation only.
Usage: python scripts/10_fusion_variants.py --window 24
Output: results/late_fusion/variants_W{W}.md / .json
"""
import argparse, glob, json, pickle
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]; M = ROOT.parent / 'MultimodalMIMIC'
LG = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); TIER = 'p_tabpfn'


def boot(y, a, b, n=2000):
    rng = np.random.RandomState(0); d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return float(np.mean(d)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def utde_run(tag, W, seed=42):
    runs = glob.glob(str(M / f'run/sofa28_W{W}{tag}/ihm/**/{"0.0004" if tag == "_tsonly" else "2e-05"}*/'), recursive=True)
    runs = [r for r in runs if Path(r, f'test_logits_{seed}.npz').exists()]
    if not runs:
        return None
    r = runs[0]; z = np.load(Path(r) / f'test_logits_{seed}.npz')
    ck = torch.load(glob.glob(str(Path(r) / '*' / f'{seed}.pth.tar'))[0], map_location='cpu', weights_only=False)
    return z['logits'], z['labels'], np.array(ck['logits'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--window', type=int, required=True); a = ap.parse_args(); W = a.window
    tab = pd.read_parquet(ROOT / 'results/tabpfn' / f'tabpfn_preds_W{W}.parquet')
    names = {s: [d['name'] for d in pickle.load(open(M / f'Data/sofa28/W{W}/{s}p2x_data.pkl', 'rb'))] for s in ('val', 'test')}
    t_te, t_va = tab.loc[names['test'], TIER].values, tab.loc[names['val'], TIER].values
    y, y_va = tab.loc[names['test'], 'y28'].values, tab.loc[names['val'], 'y28'].values
    preds = {'TabPFN (structured)': t_te}
    for tag, label in (('_tsonly', 'UTDE time series only'), ('', 'UTDE time series + notes'), ('_sofavars', 'UTDE with the tabular variable set (15 channels: SOFA inputs, lactate, BE, vent, age, sex) + notes'), ('_tabemb', 'TabPFN-embedding UTDE (TabPFN tokens + notes)'), ('_tabemb3', 'TabPFN-embedding UTDE, 3 streams (TabPFN tokens + time series + notes)')):
        r = utde_run(tag, W)
        if r is None:
            continue
        p_te, yy, p_va = r; assert (yy == y).all() and len(p_va) == len(y_va)
        preds[label] = p_te
        if tag.startswith('_tabemb'):
            continue                                           # the embedding models already contain TabPFN; no stacking
        st = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[LG(t_va), LG(p_va)], y_va)
        preds[f'late fusion TabPFN + {label}'] = st.predict_proba(np.c_[LG(t_te), LG(p_te)])[:, 1]
    rows = []; md = [f'# Fusion variants, W = {W} h (test n = {len(y)}, 28-day mortality {y.mean():.3f}); stacker fitted on validation', '',
                      '| model | AUROC | AUPRC | ΔAUROC vs TabPFN [95 % CI] |', '|---|---|---|---|']
    for k, p in preds.items():
        d, ci = boot(y, p, t_te) if k != 'TabPFN (structured)' else (0.0, [0.0, 0.0])
        rows.append({'model': k, 'auroc': float(roc_auc_score(y, p)), 'auprc': float(average_precision_score(y, p)), 'd_vs_tabpfn': d, 'ci': ci})
        md.append(f"| {k} | {rows[-1]['auroc']:.3f} | {rows[-1]['auprc']:.3f} | {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    od = ROOT / 'results/late_fusion'; json.dump(rows, open(od / f'variants_W{W}.json', 'w'), indent=1); (od / f'variants_W{W}.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
