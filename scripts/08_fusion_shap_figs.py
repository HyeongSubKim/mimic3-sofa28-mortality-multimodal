"""Figures for the late-fusion SHAP analysis (whatever inputs exist in results/fusion_shap/):
  fusion_W{W}.png        stacker: per-case contributions of TabPFN vs UTDE (logit space), coloured by outcome; bar of mean |SHAP|
  tabpfn_shap_W{W}.png   TabPFN beeswarm (top 20) + grouped-by-variable bar
  utde_attr_W{W}.png     UTDE modality Shapley (TS vs notes, logit) + per-channel occlusion (mean |delta logit| and AUROC drop)
Usage: python scripts/08_fusion_shap_figs.py --window 24
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'results' / 'fusion_shap'
LG = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))


def fig_fusion(W):
    f = D / f'fusion_shap_W{W}.csv'
    if not f.exists():
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={'width_ratios': [3, 1.3]})
    for y, c, lab in ((0, 'tab:blue', 'alive at 28 d'), (1, 'tab:red', 'died ≤ 28 d')):
        m = d.y == y; ax[0].scatter(d.shap_tabpfn[m], d.shap_utde[m], s=9, alpha=.45, c=c, label=lab)
    lim = np.abs(d[['shap_tabpfn', 'shap_utde']].values).max() * 1.05
    ax[0].plot([-lim, lim], [lim, -lim], 'k:', lw=.8); ax[0].axhline(0, c='grey', lw=.5); ax[0].axvline(0, c='grey', lw=.5)
    ax[0].set_xlim(-lim, lim); ax[0].set_ylim(-lim, lim)
    ax[0].set_xlabel('SHAP of TabPFN logit (structured)'); ax[0].set_ylabel('SHAP of UTDE logit (time series + notes)')
    ax[0].set_title(f'Late fusion, W = {W} h: per-case contributions (test n = {len(d)})'); ax[0].legend(frameon=False, fontsize=8)
    ax[0].text(lim * .55, -lim * .95, 'dotted: contributions cancel', fontsize=7)
    mabs = [d.shap_tabpfn.abs().mean(), d.shap_utde.abs().mean()]
    ax[1].barh(['TabPFN', 'UTDE'], mabs, color=['tab:green', 'tab:purple'])
    for i, v in enumerate(mabs):
        ax[1].text(v, i, f' {v:.2f}', va='center', fontsize=9)
    ax[1].set_xlabel('mean |SHAP| (logit)'); ax[1].set_title('share of the fused logit'); ax[1].set_xlim(0, max(mabs) * 1.3)
    fig.tight_layout(); fig.savefig(D / f'fusion_W{W}.png', dpi=150); plt.close(fig)


def fig_tabpfn(W):
    f = D / f'tabpfn_shap_W{W}.npz'
    if not f.exists():
        return
    import shap
    z = np.load(f, allow_pickle=True); S, X, feats = z['shap'], z['X'], list(z['features'])
    imp = pd.Series(np.abs(S).mean(0), index=feats).sort_values(ascending=False)
    fig = plt.figure(figsize=(13, 6.5))
    ax1 = fig.add_subplot(1, 2, 1)
    shap.summary_plot(S, X, feature_names=feats, max_display=20, show=False, plot_size=None)
    ax1 = plt.gca(); ax1.set_title(f'TabPFN (tier 4 + age/sex), W = {W} h: SHAP beeswarm, top 20', fontsize=10); ax1.set_xlabel('SHAP (probability of 28-day death)')
    ax2 = fig.add_subplot(1, 2, 2)
    var = imp.index.str.replace(r'_(min|max|mean|last|count|n|total|drugs|ml_per_h)$', '', regex=True)
    grp = imp.groupby(var).sum().sort_values(ascending=False).head(15)[::-1]
    ax2.barh(grp.index, grp.values, color='tab:green'); ax2.set_xlabel('sum of mean |SHAP| over the variable\'s aggregates'); ax2.set_title('grouped by source variable', fontsize=10)
    fig.tight_layout(); fig.savefig(D / f'tabpfn_shap_W{W}.png', dpi=150); plt.close(fig)


def fig_utde(W):
    f = D / f'utde_attr_W{W}.npz'
    if not f.exists():
        return
    z = np.load(f, allow_pickle=True); y = z['y']; ch = list(z['channels'])
    full, nn, nts, none = LG(z['full']), LG(z['no_notes']), LG(z['no_ts']), LG(z['none'])
    # exact 2-player Shapley on the logit: phi_ts = 1/2[(full - no_ts) + (no_notes - none)], phi_notes = 1/2[(full - no_notes) + (no_ts - none)]
    phi_ts = .5 * ((full - nts) + (nn - none)); phi_notes = .5 * ((full - nn) + (nts - none))
    occ = {c: LG(z[f'drop_{c}']) for c in ch if f'drop_{c}' in z}
    dl = pd.Series({c: np.abs(full - v).mean() for c, v in occ.items()}).sort_values()
    da = pd.Series({c: roc_auc_score(y, z['full']) - roc_auc_score(y, z[f'drop_{c}']) for c in occ}).reindex(dl.index)
    fig, ax = plt.subplots(1, 3, figsize=(14, 5), gridspec_kw={'width_ratios': [1.1, 1.6, 1.6]})
    for yy, c, lab in ((0, 'tab:blue', 'alive'), (1, 'tab:red', 'died')):
        m = y == yy; ax[0].scatter(phi_ts[m], phi_notes[m], s=8, alpha=.45, c=c, label=lab)
    ax[0].axhline(0, c='grey', lw=.5); ax[0].axvline(0, c='grey', lw=.5); ax[0].set_xlabel('Shapley: time series (logit)'); ax[0].set_ylabel('Shapley: notes (logit)')
    ax[0].set_title(f'UTDE W = {W} h: modality Shapley\nmean |φ| TS {np.abs(phi_ts).mean():.2f}, notes {np.abs(phi_notes).mean():.2f}', fontsize=9); ax[0].legend(frameon=False, fontsize=8)
    ax[1].barh(dl.index, dl.values, color='tab:purple'); ax[1].set_xlabel('mean |Δ logit| when the channel is occluded'); ax[1].set_title('channel occlusion: effect on the prediction', fontsize=9)
    ax[2].barh(da.index, da.values, color=['tab:red' if v > 0 else 'tab:grey' for v in da.values]); ax[2].axvline(0, c='k', lw=.6)
    ax[2].set_xlabel('AUROC drop when the channel is occluded'); ax[2].set_title(f'channel occlusion: effect on discrimination (full {roc_auc_score(y, z["full"]):.3f})', fontsize=9)
    fig.tight_layout(); fig.savefig(D / f'utde_attr_W{W}.png', dpi=150); plt.close(fig)
    # beeswarm form: per-case occlusion attribution (full - channel removed, logit) per channel, coloured by the channel's mean observed value
    import pickle, shap
    test = pickle.load(open(ROOT.parent / 'MultimodalMIMIC' / 'Data' / 'sofa28' / f'W{W}' / 'testp2x_data.pkl', 'rb'))
    assert len(test) == len(y), (len(test), len(y))
    chans = list(dl.index[::-1])
    Xm = np.full((len(test), len(chans)), np.nan)
    for i, d in enumerate(test):
        v = np.asarray(d['irg_ts'], dtype=float); m = np.asarray(d['irg_ts_mask']) == 1
        for j, c in enumerate(chans):
            k = ch.index(c)
            if m[:, k].any():
                Xm[i, j] = v[m[:, k], k].mean()
    A = np.column_stack([full - occ[c] for c in chans])
    plt.figure(figsize=(9, 7))
    shap.summary_plot(A, Xm, feature_names=chans, max_display=21, show=False, plot_size=None)
    plt.title(f'UTDE W = {W} h: channel occlusion attribution (logit of full model − logit without the channel)\ncolour = channel mean in the window (z-scored); grey = never measured', fontsize=9)
    plt.xlabel('attribution of the channel to the UTDE logit'); plt.tight_layout(); plt.savefig(D / f'utde_beeswarm_W{W}.png', dpi=150); plt.close()
    return {'phi_ts_mean_abs': float(np.abs(phi_ts).mean()), 'phi_notes_mean_abs': float(np.abs(phi_notes).mean()),
            'auroc_full': float(roc_auc_score(y, z['full'])), 'auroc_no_notes': float(roc_auc_score(y, z['no_notes'])), 'auroc_no_ts': float(roc_auc_score(y, z['no_ts'])),
            'channel_delta_logit': dl.to_dict(), 'channel_auroc_drop': da.to_dict()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--window', type=int, nargs='+', default=[6, 24]); a = ap.parse_args()
    for W in a.window:
        fig_fusion(W); fig_tabpfn(W); r = fig_utde(W)
        if r:
            import json; json.dump(r, open(D / f'utde_attr_W{W}.json', 'w'), indent=1)
        print(W, sorted(p.name for p in D.glob(f'*_W{W}.png')))
