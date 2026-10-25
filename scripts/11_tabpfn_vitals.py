"""Fair-variable check: give TabPFN the extra vital-sign channels that UTDE sees (HR, RR, SpO2, Temp, SBP, DBP, arterial SBP/DBP,
O2 flow) as window aggregates (min / max / mean / last / count of the hourly means), then re-fit TabPFN and re-do the late
fusion with the 21-channel UTDE. Answers: how much of the fusion gain was "variables UTDE had and TabPFN did not"?
Output: results/tabpfn_vitals/tabpfn_preds_W{W}.parquet, summary_W{W}.md
"""
import argparse, glob, importlib.util, os, pickle
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]; M = ROOT.parent / 'MultimodalMIMIC'
spec = importlib.util.spec_from_file_location('b13', ROOT / 'scripts' / '05_build_utde_dataset.py'); b13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b13)
VITALS = ['HR', 'RR', 'SpO2', 'Temp', 'SBP', 'DBP', 'ASBP', 'ADBP', 'O2flow']
LG = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); TIER = 'p_tabpfn'


def boot(y, a, b, n=2000):
    rng = np.random.RandomState(0); d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return f'{np.mean(d):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--window', type=int, required=True); ap.add_argument('--device', default='cuda'); a = ap.parse_args(); W = a.window
    out = ROOT / 'results' / 'tabpfn_vitals'; out.mkdir(exist_ok=True)
    hf = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface'))
    ckpt = sorted(glob.glob(str(hf / 'hub/models--Prior-Labs--TabPFN-v2-clf/snapshots/*/tabpfn-v2-classifier.ckpt')))[0]
    feat = pd.read_parquet(ROOT / 'results/features' / f'features_W{W}.parquet')
    icu = pd.read_parquet(ROOT / 'data/icustays.parquet', columns=['ICUSTAY_ID', 'HADM_ID', 'INTIME']).set_index('ICUSTAY_ID'); icu['INTIME'] = pd.to_datetime(icu.INTIME)
    st = feat[['SUBJECT_ID', 'dataset', 'y28']].join(icu)
    long = b13.hourly_long(st, W); long = long[long.channel.isin(VITALS)].sort_values(['ICUSTAY_ID', 'hour'])
    g = long.groupby(['ICUSTAY_ID', 'channel']).value
    agg = pd.concat({'min': g.min(), 'max': g.max(), 'mean': g.mean(), 'last': g.last(), 'count': g.size()}, axis=1).unstack('channel')
    agg.columns = [f'{c.lower()}_{s}' for s, c in agg.columns]
    V = agg.reindex(feat.index); V[[c for c in V.columns if c.endswith('_count')]] = V[[c for c in V.columns if c.endswith('_count')]].fillna(0)
    base = [c for c in feat.columns if c not in ('SUBJECT_ID', 'dataset', 'y28')]
    X0 = feat[base].to_numpy('float32'); X1 = np.hstack([X0, V.to_numpy('float32')]); y = feat.y28.to_numpy(); split = feat.dataset.to_numpy()
    ctx, trn, val, te = np.isin(split, ['train', 'val']), split == 'train', split == 'val', split == 'test'
    mk = lambda: TabPFNClassifier(device=a.device, model_path=ckpt, n_estimators=8, random_state=0)
    res = {}
    for name, X in (('TabPFN (68 structured)', X0), ('TabPFN + vitals aggregates', X1)):
        p_te = mk().fit(X[ctx], y[ctx]).predict_proba(X[te])[:, 1]; p_va = mk().fit(X[trn], y[trn]).predict_proba(X[val])[:, 1]
        res[name] = (p_va, p_te); print(f'{name:30s} {X.shape[1]} features  test AUROC {roc_auc_score(y[te], p_te):.3f}  AUPRC {average_precision_score(y[te], p_te):.3f}', flush=True)
    pv = feat[['SUBJECT_ID', 'dataset', 'y28']].copy(); col = np.full(len(feat), np.nan); col[te] = res['TabPFN + vitals aggregates'][1]; col[val] = res['TabPFN + vitals aggregates'][0]; pv['p'] = col
    pv.to_parquet(out / f'tabpfn_preds_W{W}.parquet')
    # fusion with the 21-channel UTDE and comparison with the 3-stream model (pickle order)
    names = {s: [d['name'] for d in pickle.load(open(M / f'Data/sofa28/W{W}/{s}p2x_data.pkl', 'rb'))] for s in ('val', 'test')}
    pos = {sid: i for i, sid in enumerate(feat.index)}
    iv = [pos[n] for n in names['val']]; it = [pos[n] for n in names['test']]; yt = y[it]; yv = y[iv]
    def utde(tag):
        r = [x for x in glob.glob(str(M / f'run/sofa28_W{W}{tag}/ihm/**/2e-05*/'), recursive=True) if os.path.exists(os.path.join(x, 'test_logits_42.npz'))][0]
        z = np.load(os.path.join(r, 'test_logits_42.npz')); ck = torch.load(glob.glob(os.path.join(r, '*', '42.pth.tar'))[0], map_location='cpu', weights_only=False); return np.array(ck['logits']), z['logits']
    u_va, u_te = utde('')
    t0_va, t0_te = res['TabPFN (68 structured)']; t1_va, t1_te = res['TabPFN + vitals aggregates']
    sel = lambda arr, idx_mask, order: pd.Series(arr, index=feat.index[idx_mask]).loc[order].values
    t0v, t0t = sel(t0_va, val, names['val']), sel(t0_te, te, names['test']); t1v, t1t = sel(t1_va, val, names['val']), sel(t1_te, te, names['test'])
    rows = [('TabPFN (68 structured)', t0t), ('TabPFN + vitals aggregates', t1t), ('UTDE 21 ch + notes', u_te)]
    for lab, tv, tt in (('late fusion TabPFN + UTDE', t0v, t0t), ('late fusion TabPFN+vitals + UTDE', t1v, t1t)):
        stk = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[LG(tv), LG(u_va)], yv); rows.append((lab, stk.predict_proba(np.c_[LG(tt), LG(u_te)])[:, 1]))
    try:
        _, e3 = utde('_tabemb3'); rows.append(('3-stream TabPFN-embedding UTDE', e3))
    except IndexError:
        pass
    md = [f'# TabPFN with the extra vital-sign aggregates, W = {W} h (test n = {len(yt)})', '', '| model | AUROC | AUPRC | Δ vs TabPFN (68) | Δ vs TabPFN + vitals |', '|---|---|---|---|---|']
    for lab, p in rows:
        md.append(f'| {lab} | {roc_auc_score(yt, p):.3f} | {average_precision_score(yt, p):.3f} | {boot(yt, p, t0t)} | {boot(yt, p, t1t)} |')
    (out / f'summary_W{W}.md').write_text('\n'.join(md) + '\n'); print('\n'.join(md))


if __name__ == '__main__':
    main()
