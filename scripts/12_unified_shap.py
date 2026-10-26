"""Attribution for the models of the unified-variable table (SOFA inputs + lactate/BE + age/sex only):
  TabPFN (68)                      -> permutation SHAP (from 07_fusion_shap.py: results/fusion_shap/tabpfn_shap_W{W}.npz)
  UTDE-SOFA (15 ch + notes)        -> occlusion (utde_attr_W{W}_sofavars.npz): modality AUROC drops, channel beeswarm
  late fusion TabPFN + UTDE-SOFA   -> exact stacker SHAP (stacker refitted on validation) + feature-level beeswarm
                                      ((T) TabPFN SHAP x b1 in logit units, (U) UTDE-SOFA occlusion x b2, notes centred)
  TabPFN-in-UTDE replace / 3-stream (15 ch) -> occlusion of the TabPFN tokens / notes / channels when the npz exists
Outputs results/unified_shap/: *.png, summary_W{W}.md
Usage: python scripts/12_unified_shap.py --window 24
"""
import argparse, glob, json, pickle
from pathlib import Path
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, shap
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]; M = ROOT.parent / 'MultimodalMIMIC'; D = ROOT / 'results' / 'fusion_shap'
OUT = ROOT / 'results' / 'unified_shap'; OUT.mkdir(exist_ok=True)
LG = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); TIER = 'p_tabpfn'; PH = 'No clinical note available in this window.'


def utde_run(tag, W, seed=42):
    r = [x for x in glob.glob(str(M / f'run/sofa28_W{W}{tag}/ihm/**/2e-05*/'), recursive=True) if Path(x, f'test_logits_{seed}.npz').exists()][0]
    z = np.load(Path(r) / f'test_logits_{seed}.npz'); ck = torch.load(glob.glob(str(Path(r) / '*' / f'{seed}.pth.tar'))[0], map_location='cpu', weights_only=False)
    return z['logits'], np.array(ck['logits'])


def beeswarm(A, X, names, title, path, xlabel):
    plt.figure(figsize=(9, 0.32 * len(names) + 1.6))
    shap.summary_plot(A, X, feature_names=names, max_display=len(names), show=False, plot_size=None)
    lim = np.percentile(np.abs(A), 99.7); plt.xlim(-lim, lim); plt.title(title, fontsize=8); plt.xlabel(xlabel); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--window', type=int, required=True); a = ap.parse_args(); W = a.window
    md = [f'# Attribution, unified variable set, W = {W} h', '']
    tab = pd.read_parquet(ROOT / 'results/tabpfn' / f'tabpfn_preds_W{W}.parquet')
    feat = pd.read_parquet(ROOT / 'results/features' / f'features_W{W}.parquet')
    data = M / 'Data/sofa28_sofa' / f'W{W}'
    names = {s: [d['name'] for d in pickle.load(open(data / f'{s}p2x_data.pkl', 'rb'))] for s in ('val', 'test')}
    test = pickle.load(open(data / 'testp2x_data.pkl', 'rb')); ch = json.load(open(data / 'channels.json'))['channels']
    y = tab.loc[names['test'], 'y28'].values; y_va = tab.loc[names['val'], 'y28'].values
    t_te, t_va = tab.loc[names['test'], TIER].values, tab.loc[names['val'], TIER].values

    # ---- 1. TabPFN: already explained (permutation SHAP); reproduce the top-10 here for the record
    z = np.load(D / f'tabpfn_shap_W{W}.npz', allow_pickle=True); S, feats = z['shap'], list(z['features'])
    te_ids = feat.index[(feat.dataset == 'test').values]; St = pd.DataFrame(S, index=te_ids, columns=feats)
    imp = St.abs().mean().sort_values(ascending=False)
    md += ['## 1. TabPFN (68 features) — permutation SHAP, mean |SHAP| (probability units), top 10', '', '| rank | feature | mean abs SHAP |', '|---|---|---|']
    md += [f'| {i+1} | `{k}` | {v:.4f} |' for i, (k, v) in enumerate(imp.head(10).items())]
    md += ['', f'(figure: `results/fusion_shap/tabpfn_shap_W{W}.png`)', '']

    # ---- 2. UTDE-SOFA occlusion
    u = np.load(D / f'utde_attr_W{W}_sofavars.npz', allow_pickle=True); assert (u['y'] == y).all()
    full = LG(u['full']); auc = lambda p: roc_auc_score(y, p)
    md += ['## 2. UTDE-SOFA (15 channels + notes) — occlusion', '',
           f"| full | without notes | without time series | notes only vs TS only |", '|---|---|---|---|',
           f"| {auc(u['full']):.3f} | {auc(u['no_notes']):.3f} (−{auc(u['full'])-auc(u['no_notes']):.3f}) | {auc(u['no_ts']):.3f} (−{auc(u['full'])-auc(u['no_ts']):.3f}) | {auc(u['no_ts']):.3f} vs {auc(u['no_notes']):.3f} |", '']
    chans = [c for c in ch if f'drop_{c}' in u]
    dl = pd.Series({c: np.abs(full - LG(u[f'drop_{c}'])).mean() for c in chans}).sort_values(ascending=False)
    da = pd.Series({c: auc(u['full']) - auc(u[f'drop_{c}']) for c in chans}).reindex(dl.index)
    md += ['| channel | mean abs Δlogit when occluded | AUROC drop |', '|---|---|---|'] + [f'| {c} | {dl[c]:.3f} | {da[c]:+.3f} |' for c in dl.index] + ['']
    BINARY = {'Vent', 'Vasopressor', 'Sex'}                       # presence channels: colour = present in the window (1) or not (0)
    Xm = np.full((len(test), len(chans)), np.nan)
    for i, d in enumerate(test):
        v = np.asarray(d['irg_ts'], float); m = np.asarray(d['irg_ts_mask']) == 1
        for j, c in enumerate(chans):
            k = ch.index(c)
            if c in BINARY:
                Xm[i, j] = float(v[m[:, k], k].max()) if m[:, k].any() else 0.0
            elif m[:, k].any():
                Xm[i, j] = v[m[:, k], k].mean()
    Au = np.column_stack([full - LG(u[f'drop_{c}']) for c in chans])
    beeswarm(Au, Xm, chans, f'UTDE-SOFA W = {W} h: channel occlusion attribution (logit full − logit without channel); colour = channel mean (z); 0/1 channels: present in window', OUT / f'utde_sofa_beeswarm_W{W}.png', 'attribution to the UTDE-SOFA logit')

    # ---- 3. late fusion TabPFN + UTDE-SOFA: stacker SHAP + feature-level beeswarm
    s_te, s_va = utde_run('_sofavars', W)
    st = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[LG(t_va), LG(s_va)], y_va); b1, b2 = st.coef_[0]
    Xf = np.c_[LG(t_te), LG(s_te)]; phi = st.coef_[0] * (Xf - Xf.mean(0)); p_f = st.predict_proba(Xf)[:, 1]
    md += ['## 3. Late fusion TabPFN + UTDE-SOFA — exact stacker SHAP', '',
           f'stacker coefficients TabPFN {b1:.2f}, UTDE-SOFA {b2:.2f}; mean |SHAP| TabPFN {np.abs(phi[:,0]).mean():.2f}, UTDE-SOFA {np.abs(phi[:,1]).mean():.2f} '
           f'(UTDE share {np.abs(phi[:,1]).sum()/np.abs(phi).sum():.0%}); logit correlation {np.corrcoef(Xf[:,0], Xf[:,1])[0,1]:.2f}; fusion test AUROC {auc(p_f):.3f}', '']
    plt.figure(figsize=(8, 2.8)); shap.summary_plot(phi, Xf, feature_names=['TabPFN logit', 'UTDE-SOFA logit'], show=False, plot_size=None)
    plt.title(f'Late fusion TabPFN + UTDE-SOFA, W = {W} h: stacker SHAP', fontsize=10); plt.xlabel('SHAP on the fused logit'); plt.tight_layout(); plt.savefig(OUT / f'fusion_sofa_stacker_W{W}.png', dpi=150); plt.close()
    p = np.clip(z['p'], 0.03, 0.97); StL = pd.DataFrame(S / (p * (1 - p))[:, None] * b1, index=te_ids, columns=[f'{c} (T)' for c in feats]); Xt = pd.DataFrame(z['X'], index=te_ids, columns=StL.columns)
    dn = full - LG(u['no_notes']); AuF = pd.DataFrame(Au * b2, index=names['test'], columns=[f'{c} (U)' for c in chans]); AuF['notes (U), offset removed'] = (dn - dn.mean()) * b2
    XuF = pd.DataFrame(Xm, index=names['test'], columns=[f'{c} (U)' for c in chans]); XuF['notes (U), offset removed'] = [sum(t != PH for t in d['text_data']) for d in test]
    ids = [i for i in te_ids if i in AuF.index]
    A = pd.concat([StL.loc[ids], AuF.loc[ids]], axis=1); X = pd.concat([Xt.loc[ids], XuF.loc[ids]], axis=1)
    order = A.abs().mean().sort_values(ascending=False).index[:25]
    beeswarm(A[order].values, X[order].values, list(order), f'Late fusion TabPFN + UTDE-SOFA, W = {W} h: feature-level attribution of the fused logit (top 25)\n(T) TabPFN SHAP x {b1:.2f} (prob->logit per case, p clipped); (U) UTDE-SOFA occlusion x {b2:.2f}; notes offset ({dn.mean():+.2f}) removed', OUT / f'fusion_sofa_feature_beeswarm_W{W}.png', 'attribution to the fused logit')
    md += ['| rank | input | mean abs attribution (fused logit) |', '|---|---|---|'] + [f'| {i+1} | {k} | {v:.3f} |' for i, (k, v) in enumerate(A.abs().mean().sort_values(ascending=False).head(15).items())] + ['']

    # ---- 4. TabPFN-in-UTDE (replace / 3-stream on the 15 channels) — occlusion of the TabPFN tokens and notes
    for tag, label in (('_tabemb', 'TabPFN-in-UTDE, replace (TabPFN tokens + notes)'), ('_tabemb3_sofavars', 'TabPFN-in-UTDE, 3 streams (TabPFN tokens + 15-ch time series + notes)')):
        f = D / f'utde_attr_W{W}{tag}.npz'
        if not f.exists():
            md += [f'## 4. {label}: not available yet', '']; continue
        e = np.load(f, allow_pickle=True); fe = LG(e['full'])
        md += [f'## 4. {label} — occlusion', '', '| full | without TabPFN tokens | without notes | without TabPFN tokens and notes |' + (' without time series |' if 'no_ts' in e and tag.endswith('sofavars') else ''), '|---|---|---|---|' + ('---|' if tag.endswith('sofavars') else '')]
        row = f"| {auc(e['full']):.3f} | {auc(e['no_tab']):.3f} (−{auc(e['full'])-auc(e['no_tab']):.3f}) | {auc(e['no_notes']):.3f} (−{auc(e['full'])-auc(e['no_notes']):.3f}) | {auc(e['no_tab_no_notes']):.3f} |"
        if tag.endswith('sofavars'):
            row += f" {auc(e['no_ts']):.3f} (−{auc(e['full'])-auc(e['no_ts']):.3f}) |"
        md += [row, '', f"mean |Δlogit| when removed: TabPFN tokens {np.abs(fe-LG(e['no_tab'])).mean():.2f}, notes {np.abs(fe-LG(e['no_notes'])).mean():.2f}" + (f", time series {np.abs(fe-LG(e['no_ts'])).mean():.2f}" if tag.endswith('sofavars') else ''), '']
        chans_e = [c for c in e['channels'] if f'drop_{c}' in e]
        if chans_e:
            Ae = np.column_stack([fe - LG(e[f'drop_{c}']) for c in chans_e]); dle = pd.Series(np.abs(Ae).mean(0), index=chans_e).sort_values(ascending=False)
            md += ['| channel | mean abs Δlogit | AUROC drop |', '|---|---|---|'] + [f"| {c} | {dle[c]:.3f} | {auc(e['full'])-auc(e[f'drop_{c}']):+.3f} |" for c in dle.index] + ['']
            beeswarm(Ae, Xm[:, [chans.index(c) for c in chans_e]], chans_e, f'{label}, W = {W} h: channel occlusion attribution', OUT / f'tabemb3_sofa_beeswarm_W{W}.png', 'attribution to the logit')
    (OUT / f'summary_W{W}.md').write_text('\n'.join(md) + '\n'); print('\n'.join(md))


if __name__ == '__main__':
    main()
