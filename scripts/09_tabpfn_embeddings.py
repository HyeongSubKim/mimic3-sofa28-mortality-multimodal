"""Frozen TabPFN v2 embeddings per stay, to be used as the tabular branch of a UTDE-style fusion (REVIEW §9).

Rows are always embedded in *query* mode against a context that excludes them: val / test rows against the train split,
train rows with a 5-fold context inside the train split. One 192-d vector per ensemble member (n_estimators = 8) is kept
(members permute features / classes, so they are not averaged).
Output: <out>/W{W}/tabpfn_emb.npz  with ids (ICUSTAY_ID), emb (n, 8, 192), split, y, and a probe AUROC printed as a check.
Usage: HF_HOME=... python scripts/09_tabpfn_embeddings.py --windows 24 6 [--out ../MultimodalMIMIC/Data/sofa28]
"""
import argparse, glob, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
META = ['SUBJECT_ID', 'dataset', 'y28']


def embed(clf, X):
    E = clf.get_embeddings(X, data_source='test')          # (n_estimators, n, 192)
    return np.transpose(np.asarray(E), (1, 0, 2)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, nargs='+', default=[24, 6])
    ap.add_argument('--out', default='../MultimodalMIMIC/Data/sofa28')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    hf = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface'))
    ckpt = sorted(glob.glob(str(hf / 'hub/models--Prior-Labs--TabPFN-v2-clf/snapshots/*/tabpfn-v2-classifier.ckpt')))[0]
    mk = lambda: TabPFNClassifier(device=args.device, model_path=ckpt, n_estimators=8, random_state=0)
    for W in args.windows:
        df = pd.read_parquet(ROOT / 'results' / 'features' / f'features_W{W}.parquet')
        feats = [c for c in df.columns if c not in META]; X = df[feats].to_numpy('float32'); y = df.y28.to_numpy(); split = df.dataset.to_numpy()
        tr = split == 'train'; E = np.zeros((len(df), 8, 192), np.float32)
        clf = mk().fit(X[tr], y[tr]); E[~tr] = embed(clf, X[~tr])                                   # val + test: context = train
        for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X[tr], y[tr])):
            idx = np.where(tr)[0]; E[idx[b]] = embed(mk().fit(X[idx[a]], y[idx[a]]), X[idx[b]])     # train: 5-fold context
            print(f'W={W} train fold {k} embedded', flush=True)
        te = split == 'test'; va = split == 'val'
        for m in range(8):                                                                           # per-member probe as a sanity check
            p = LogisticRegression(C=0.1, max_iter=2000).fit(E[tr, m], y[tr]).predict_proba(E[te, m])[:, 1]
            print(f'  member {m}: probe train->test AUROC {roc_auc_score(y[te], p):.3f}', flush=True)
        p = LogisticRegression(C=0.1, max_iter=5000).fit(E[tr].reshape(tr.sum(), -1), y[tr]).predict_proba(E[te].reshape(te.sum(), -1))[:, 1]
        print(f'W={W}: concatenated 8x192 probe test AUROC {roc_auc_score(y[te], p):.3f} (TabPFN itself 0.832 / 0.837)', flush=True)
        out = Path(args.out) / f'W{W}'; out.mkdir(parents=True, exist_ok=True)
        np.savez(out / 'tabpfn_emb.npz', ids=df.index.values.astype(np.int64), emb=E, split=split, y=y)
        print('saved', out / 'tabpfn_emb.npz', E.shape, flush=True)


if __name__ == '__main__':
    main()
