"""Occlusion attribution for TabPFN (same method as the UTDE channel occlusion): a variable group's columns are set to NaN
(= 'not measured', which TabPFN handles natively) for the test rows, the model is re-queried, and the per-case Δlogit
(full − occluded) and the AUROC drop are recorded.  Then the late-fusion feature-level attribution is rebuilt with both
sides occlusion: (T) TabPFN group Δlogit × b₁ + (U) UTDE-SOFA channel Δlogit × b₂ (notes offset-centred).
Outputs results/unified_shap/: tabpfn_occlusion_W{W}.npz/.csv, tabpfn_occlusion_beeswarm_W{W}.png,
fusion_sofa_feature_occlusion_W{W}.png, occlusion_table_W{W}.md
Usage: HF_HOME=... python scripts/13_tabpfn_occlusion.py --window 24
"""
import argparse, glob, json, os, pickle, re
from pathlib import Path
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, shap
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'results'; M = ROOT.parent / 'MultimodalMIMIC'; OUT = R / 'unified_shap'
LG = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); TIER = 'p_tabpfn'; PH = 'No clinical note available in this window.'
GROUP_RE = r'^(pao2|fio2|pf|platelets|bilirubin|creatinine|map|gcs|urine|lactate|base_excess)_'
REP = {'pao2': 'pao2_last', 'fio2': 'fio2_last', 'pf': 'pf_last', 'platelets': 'platelets_last', 'bilirubin': 'bilirubin_last', 'creatinine': 'creatinine_last',
       'map': 'map_last', 'gcs': 'gcs_last', 'urine': 'urine_total', 'lactate': 'lactate_last', 'base_excess': 'base_excess_last', 'vasopressor': 'vaso_n',
       'vent': 'vent', 'sofa_scores': 'sofa_total', 'age': 'age', 'sex': 'sex'}


def group_of(c):
    m = re.match(GROUP_RE, c)
    if m: return m.group(1)
    if c.startswith('vaso'): return 'vasopressor'
    if c.startswith('vent'): return 'vent'
    if c.startswith('sofa_'): return 'sofa_scores'
    return c  # age, sex


def beeswarm(A, X, names, title, path, xlabel, h=None):
    plt.figure(figsize=(9, h or 0.32 * len(names) + 1.6)); shap.summary_plot(A, X, feature_names=names, max_display=len(names), show=False, plot_size=None)
    lim = np.percentile(np.abs(A), 99.7); plt.xlim(-lim, lim); plt.title(title, fontsize=8); plt.xlabel(xlabel); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--window', type=int, required=True); ap.add_argument('--device', default='cuda'); a = ap.parse_args(); W = a.window
    hf = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface'))
    ckpt = sorted(glob.glob(str(hf / 'hub/models--Prior-Labs--TabPFN-v2-clf/snapshots/*/tabpfn-v2-classifier.ckpt')))[0]
    feat = pd.read_parquet(ROOT / 'results/features' / f'features_W{W}.parquet'); cols = [c for c in feat.columns if c not in ('SUBJECT_ID', 'dataset', 'y28')]
    X = feat[cols].to_numpy('float32'); y = feat.y28.to_numpy(); ctx = feat.dataset.isin(['train', 'val']).to_numpy(); te = (feat.dataset == 'test').to_numpy()
    groups = {}
    for c in cols: groups.setdefault(group_of(c), []).append(c)
    clf = TabPFNClassifier(device=a.device, model_path=ckpt, n_estimators=8, random_state=0).fit(X[ctx], y[ctx])
    full = clf.predict_proba(X[te])[:, 1]; yt = y[te]; print(f'W={W} full AUROC {roc_auc_score(yt, full):.4f}', flush=True)
    occ = {}
    for g, gc in groups.items():
        Xo = X[te].copy(); Xo[:, [cols.index(c) for c in gc]] = np.nan
        occ[g] = clf.predict_proba(Xo)[:, 1]; print(f'  {g:12s} ({len(gc):2d} cols) AUROC {roc_auc_score(yt, occ[g]):.4f}  mean|Δlogit| {np.abs(LG(full)-LG(occ[g])).mean():.3f}', flush=True)
    te_ids = feat.index[te]
    np.savez(OUT / f'tabpfn_occlusion_W{W}.npz', ids=te_ids.values, y=yt, full=full, groups=np.array(list(groups)), **{f'drop_{g}': v for g, v in occ.items()})
    A_t = pd.DataFrame({g: LG(full) - LG(occ[g]) for g in groups}, index=te_ids)
    X_t = pd.DataFrame({g: feat.loc[te_ids, REP[g]].values for g in groups}, index=te_ids)
    imp = pd.DataFrame({'group': list(groups), 'n_cols': [len(groups[g]) for g in groups], 'mean_abs_dlogit': [A_t[g].abs().mean() for g in groups],
                        'auroc_drop': [roc_auc_score(yt, full) - roc_auc_score(yt, occ[g]) for g in groups]}).sort_values('mean_abs_dlogit', ascending=False)
    imp.to_csv(OUT / f'tabpfn_occlusion_W{W}.csv', index=False)
    order = list(imp.group)
    beeswarm(A_t[order].values, X_t[order].values, order, f'TabPFN (68 features), W = {W} h: variable-group occlusion (logit full − logit with the group set to NaN)\ncolour = representative value (last / total / age / sex)', OUT / f'tabpfn_occlusion_beeswarm_W{W}.png', 'attribution to the TabPFN logit')

    # ---- late fusion feature-level, both sides occlusion
    tab = pd.read_parquet(ROOT / 'results/tabpfn' / f'tabpfn_preds_W{W}.parquet'); data = M / 'Data/sofa28_sofa' / f'W{W}'
    names = {s: [d['name'] for d in pickle.load(open(data / f'{s}p2x_data.pkl', 'rb'))] for s in ('val', 'test')}
    test = pickle.load(open(data / 'testp2x_data.pkl', 'rb')); ch = json.load(open(data / 'channels.json'))['channels']
    r = [x for x in glob.glob(str(M / f'run/sofa28_W{W}_sofavars/ihm/**/2e-05*/'), recursive=True) if Path(x, 'test_logits_42.npz').exists()][0]
    s_te = np.load(Path(r) / 'test_logits_42.npz')['logits']; s_va = np.array(torch.load(glob.glob(str(Path(r) / '*' / '42.pth.tar'))[0], map_location='cpu', weights_only=False)['logits'])
    t_va = tab.loc[names['val'], TIER].values; y_va = tab.loc[names['val'], 'y28'].values; t_te = tab.loc[names['test'], TIER].values
    st = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[LG(t_va), LG(s_va)], y_va); b1, b2 = st.coef_[0]
    u = np.load(R / 'fusion_shap' / f'utde_attr_W{W}_sofavars.npz', allow_pickle=True); fu = LG(u['full']); chans = [c for c in ch if f'drop_{c}' in u]
    A_u = pd.DataFrame({f'{c} (U)': (fu - LG(u[f'drop_{c}'])) * b2 for c in chans}, index=names['test'])
    dn = fu - LG(u['no_notes']); A_u['notes (U)'] = (dn - dn.mean()) * b2
    BIN = {'Vent', 'Vasopressor', 'Sex'}; Xu = np.full((len(test), len(chans) + 1), np.nan)
    for i, d in enumerate(test):
        v = np.asarray(d['irg_ts'], float); m = np.asarray(d['irg_ts_mask']) == 1
        for j, c in enumerate(chans):
            k = ch.index(c); Xu[i, j] = (float(v[m[:, k], k].max()) if m[:, k].any() else 0.0) if c in BIN else (v[m[:, k], k].mean() if m[:, k].any() else np.nan)
        Xu[i, -1] = sum(t != PH for t in d['text_data'])
    X_u = pd.DataFrame(Xu, index=names['test'], columns=list(A_u.columns))
    ids = [i for i in te_ids if i in A_u.index]
    A = pd.concat([(A_t.loc[ids] * b1).add_suffix(' (T)'), A_u.loc[ids]], axis=1); Xc = pd.concat([X_t.loc[ids].add_suffix(' (T)'), X_u.loc[ids]], axis=1)
    rank = A.abs().mean().sort_values(ascending=False); order = list(rank.index[:25])
    beeswarm(A[order].values, Xc[order].values, order, f'Late fusion TabPFN + UTDE-SOFA, W = {W} h: feature-level attribution of the fused logit — both sides OCCLUSION\n(T) TabPFN variable-group occlusion × {b1:.2f}; (U) UTDE-SOFA channel occlusion × {b2:.2f}; notes offset ({dn.mean():+.2f}) removed; colour = value',
             OUT / f'fusion_sofa_feature_occlusion_W{W}.png', 'attribution to the fused logit')
    # ---- apples-to-apples table: AUROC drop per variable, TabPFN vs UTDE-SOFA
    pair = {'gcs': 'GCS', 'age': 'Age', 'urine': 'Urine', 'lactate': 'Lactate', 'base_excess': 'BaseExcess', 'bilirubin': 'Bilirubin', 'creatinine': 'Creatinine',
            'platelets': 'Platelets', 'pao2': 'PaO2', 'fio2': 'FiO2', 'map': 'MeanBP', 'vasopressor': 'Vasopressor', 'vent': 'Vent', 'sex': 'Sex'}
    aur_t = {g: roc_auc_score(yt, full) - roc_auc_score(yt, occ[g]) for g in groups}; aur_u = {c: roc_auc_score(u['y'], u['full']) - roc_auc_score(u['y'], u[f'drop_{c}']) for c in chans}
    md = [f'# Occlusion attribution, both models, W = {W} h (AUROC drop when the variable is removed; mean |Δlogit| in brackets)', '',
          '| variable | TabPFN (group of columns → NaN) | UTDE-SOFA (channel → 0) |', '|---|---|---|']
    for g, c in sorted(pair.items(), key=lambda kv: -aur_t[kv[0]]):
        md.append(f"| {c} | {-aur_t[g]:+.3f} [{A_t[g].abs().mean():.2f}] | {-aur_u.get(c, float('nan')):+.3f} [{np.abs(fu - LG(u[f'drop_{c}'])).mean() if c in aur_u else float('nan'):.2f}] |")
    md += [f"| SOFA component scores (7 cols) | {-aur_t['sofa_scores']:+.3f} [{A_t['sofa_scores'].abs().mean():.2f}] | – |", f"| P/F ratio (3 cols) | {-aur_t['pf']:+.3f} [{A_t['pf'].abs().mean():.2f}] | – |",
           f"| notes | – | {roc_auc_score(u['y'], u['no_notes']) - roc_auc_score(u['y'], u['full']):+.3f} [{np.abs(dn).mean():.2f}] |", '',
           f'stacker coefficients b₁ (TabPFN) {b1:.2f}, b₂ (UTDE-SOFA) {b2:.2f}. MeanBP in UTDE = MeanBP + MeanABP channels (shown: MeanBP).', '',
           '## fused-logit attribution, both sides occlusion (top 15)', '', '| rank | input | mean abs attribution |', '|---|---|---|']
    md += [f'| {i+1} | {k} | {v:.3f} |' for i, (k, v) in enumerate(rank.head(15).items())]
    (OUT / f'occlusion_table_W{W}.md').write_text('\n'.join(md) + '\n'); print('\n'.join(md))


if __name__ == '__main__':
    main()
