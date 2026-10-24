"""Cohort, patient split and structured features for the first W hours of the ICU stay (shared by TabPFN, UTDE and the fusion).

Cohort per window W: adults (>= 18 y), SOFA computable within W h (all six organ systems measured), lactate and base
excess measured within W h, alive at hour W. Outcome: death from any cause within 28 days of ICU admission (PATIENTS.DOD).
Patient-level 70/15/15 split (seed 0). Features: min / max / mean / last / count of the SOFA inputs (PaO2, FiO2, PaO2/FiO2,
platelets, bilirubin, creatinine, MAP, GCS), vasopressors, urine output, ventilation, lactate, base excess, the six SOFA
component scores and the total, plus age and sex (68 columns).
Needs data/*.parquet from 01_extract_events.py; the lab / GCS subsets it additionally needs are extracted from LABEVENTS /
CHARTEVENTS on the first run and cached in data/.

Usage: MIMIC3_RAW=/path/to/mimic-iii-clinical-database-1.4 python scripts/03_build_features.py [--windows 6 24]
Output: results/features/features_W{W}.parquet  (index ICUSTAY_ID; columns: features..., age, sex, SUBJECT_ID, dataset, y28)
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(os.environ.get('MIMIC3_RAW', 'data/mimic-iii-clinical-database-1.4'))
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
H = pd.Timedelta(1, 'h')
SEED = 0

MAP_ITEMS = {456, 52, 6702, 443, 220052, 220181, 225312}
FIO2_ITEMS = {190, 3420, 3422, 223835}
VENT_ITEMS = {720, 223849, 223848, 445, 448, 449, 450, 1340, 1486, 1600, 224687, 639, 654, 681, 682, 683, 684, 224685,
              224684, 224686, 218, 436, 535, 444, 224697, 224695, 224696, 224746, 224747, 221, 1, 1211, 1655, 2000,
              226873, 224738, 224419, 224750, 227187, 543, 5865, 5866, 224707, 224709, 224705, 224706, 60, 437, 505,
              506, 686, 220339, 224700, 3459, 501, 502, 503, 224702, 223, 667, 668, 669, 670, 671, 672, 224701}
VASO = {30047: 'norepinephrine', 30120: 'norepinephrine', 221906: 'norepinephrine',
        30044: 'epinephrine', 30119: 'epinephrine', 30309: 'epinephrine', 221289: 'epinephrine',
        30043: 'dopamine', 30307: 'dopamine', 221662: 'dopamine',
        30042: 'dobutamine', 30306: 'dobutamine', 221653: 'dobutamine',
        30127: 'phenylephrine', 30128: 'phenylephrine', 221749: 'phenylephrine',
        30125: 'vasopressin', 222315: 'vasopressin'}
GCS_EYE, GCS_MOTOR, GCS_VERBAL = {184, 220739}, {454, 223901}, {723, 223900}
LAB = {50821: 'pao2', 50816: 'fio2_lab', 51265: 'platelets', 50885: 'bilirubin', 50912: 'creatinine',
       50813: 'lactate', 50802: 'base_excess'}



# ----------------------------------------------------------------------------------------------- cohort
def load_cohort_tables():
    icu = pd.read_csv(RAW / 'ICUSTAYS.csv.gz', parse_dates=['INTIME', 'OUTTIME'])
    pat = pd.read_csv(RAW / 'PATIENTS.csv.gz', usecols=['SUBJECT_ID', 'DOB', 'DOD', 'GENDER'], parse_dates=['DOB', 'DOD'])
    adm = pd.read_csv(RAW / 'ADMISSIONS.csv.gz', usecols=['HADM_ID', 'DEATHTIME'], parse_dates=['DEATHTIME'])
    st = icu.merge(pat, on='SUBJECT_ID').merge(adm, on='HADM_ID')
    st['age'] = (st.INTIME.dt.year - st.DOB.dt.year) - ((st.INTIME.dt.month * 100 + st.INTIME.dt.day) < (st.DOB.dt.month * 100 + st.DOB.dt.day))
    st['age'] = st.age.astype(float); st.loc[st.age > 120, 'age'] = 91.4
    st = st[(st.age >= 18) & st.OUTTIME.notna() & (st.OUTTIME > st.INTIME)].copy()
    d0 = st.INTIME.dt.floor('D')
    st['y28'] = ((st.DOD >= d0) & (st.DOD <= d0 + pd.Timedelta(28, 'D'))).astype(int)
    st['sex'] = (st.GENDER == 'M').astype(int)
    # patient-level split, fixed across windows and tiers
    rng = np.random.RandomState(SEED)
    subj = st.SUBJECT_ID.unique(); rng.shuffle(subj)
    n = len(subj)
    split = {s: 'train' for s in subj[:int(.7 * n)]}
    split.update({s: 'val' for s in subj[int(.7 * n):int(.85 * n)]})
    split.update({s: 'test' for s in subj[int(.85 * n):]})
    st['dataset'] = st.SUBJECT_ID.map(split)
    return st.set_index('ICUSTAY_ID')


def load_events(st):
    ids = set(st.index)
    lab = pd.read_parquet(DATA / 'labevents_sofa.parquet').merge(st.reset_index()[['HADM_ID', 'ICUSTAY_ID']], on='HADM_ID')
    lab['t'] = pd.to_datetime(lab.CHARTTIME); lab['name'] = lab.ITEMID.map(LAB)
    ce = pd.read_parquet(DATA / 'chartevents.parquet', columns=['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'VALUE'])
    ce = ce[ce.ICUSTAY_ID.isin(ids)]; ce['t'] = pd.to_datetime(ce.CHARTTIME)
    gcs = pd.read_parquet(DATA / 'chartevents_gcs.parquet'); gcs = gcs[gcs.ICUSTAY_ID.isin(ids)]; gcs['t'] = pd.to_datetime(gcs.CHARTTIME)
    icv = pd.read_parquet(DATA / 'inputevents_cv.parquet', columns=['ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'AMOUNT']).rename(columns={'CHARTTIME': 't'})
    imv = pd.read_parquet(DATA / 'inputevents_mv.parquet', columns=['ICUSTAY_ID', 'STARTTIME', 'ITEMID', 'AMOUNT']).rename(columns={'STARTTIME': 't'})
    vaso = pd.concat([icv, imv]); vaso = vaso[vaso.ITEMID.isin(VASO) & (vaso.AMOUNT > 0) & vaso.ICUSTAY_ID.isin(ids)].copy()
    vaso['t'] = pd.to_datetime(vaso.t); vaso['drug'] = vaso.ITEMID.map(VASO)
    out = pd.read_parquet(DATA / 'outputevents.parquet', columns=['ICUSTAY_ID', 'CHARTTIME', 'VALUE'])
    out = out[out.VALUE.notna() & out.ICUSTAY_ID.isin(ids)].copy(); out['t'] = pd.to_datetime(out.CHARTTIME)
    for df in (lab, ce, gcs, vaso, out):          # chart/input/output parquets carry ICUSTAY_ID as float
        df['ICUSTAY_ID'] = df.ICUSTAY_ID.astype('int64')
    return lab, ce, gcs, vaso, out


def window(df, st, W, lab_window=False):
    it = df.ICUSTAY_ID.map(st.INTIME)
    lo = it - 6 * H if lab_window else it
    return df[(df.t >= lo) & (df.t <= it + W * H)]


def agg(df, col, name, stats=('min', 'max', 'mean', 'last', 'count')):
    g = df.sort_values('t').groupby('ICUSTAY_ID')[col]
    return pd.concat({f'{name}_{s}': getattr(g, s)() for s in stats}, axis=1)


# ----------------------------------------------------------------------------------------------- features + SOFA
def build_features(st, ev, W):
    lab, ce, gcs, vaso, out = ev
    labw = window(lab, st, W, True)
    labw = labw.assign(VALUENUM=np.where(labw.name.eq('fio2_lab') & (labw.VALUENUM <= 1), labw.VALUENUM * 100, labw.VALUENUM))
    cew = window(ce[ce.VALUENUM.notna()], st, W)
    fio2 = cew[cew.ITEMID.isin(FIO2_ITEMS)].copy()
    fio2['VALUENUM'] = np.where(fio2.VALUENUM <= 1, fio2.VALUENUM * 100, fio2.VALUENUM)
    fio2 = pd.concat([fio2[['ICUSTAY_ID', 't', 'VALUENUM']], labw.loc[labw.name == 'fio2_lab', ['ICUSTAY_ID', 't', 'VALUENUM']]])
    fio2 = fio2[(fio2.VALUENUM >= 21) & (fio2.VALUENUM <= 100)]
    pao2 = labw[labw.name == 'pao2'][['ICUSTAY_ID', 't', 'VALUENUM']]
    # PaO2/FiO2: each PaO2 paired with the most recent FiO2 charted up to 4 h before it (21 % if none)
    pf = pd.merge_asof(pao2.sort_values('t'), fio2.sort_values('t').rename(columns={'VALUENUM': 'fio2'}),
                       on='t', by='ICUSTAY_ID', direction='backward', tolerance=4 * H)
    pf['fio2'] = pf.fio2.fillna(21.0); pf['pf'] = pf.VALUENUM / (pf.fio2 / 100)
    mapw = cew[cew.ITEMID.isin(MAP_ITEMS) & cew.VALUENUM.between(20, 200)]
    vent = window(ce[ce.ITEMID.isin(VENT_ITEMS)], st, W).groupby('ICUSTAY_ID').size().rename('vent_n')
    # GCS total per charting time (sum of the three components when charted together)
    g = window(gcs, st, W).copy()
    g['comp'] = np.select([g.ITEMID.isin(GCS_EYE), g.ITEMID.isin(GCS_MOTOR), g.ITEMID.isin(GCS_VERBAL)], ['e', 'm', 'v'], default='x')
    g = g.pivot_table(index=['ICUSTAY_ID', 't'], columns='comp', values='VALUENUM', aggfunc='first').dropna().reset_index()
    g['VALUENUM'] = g.e + g.m + g.v
    vw = window(vaso, st, W); ow = window(out, st, W)

    X = pd.concat([
        agg(pao2, 'VALUENUM', 'pao2'), agg(fio2, 'VALUENUM', 'fio2'), agg(pf, 'pf', 'pf', ('min', 'mean', 'last')),
        agg(labw[labw.name == 'platelets'], 'VALUENUM', 'platelets'),
        agg(labw[labw.name == 'bilirubin'], 'VALUENUM', 'bilirubin'),
        agg(labw[labw.name == 'creatinine'], 'VALUENUM', 'creatinine'),
        agg(mapw, 'VALUENUM', 'map'), agg(g, 'VALUENUM', 'gcs', ('min', 'last', 'count')),
        vw.groupby('ICUSTAY_ID').agg(vaso_n=('AMOUNT', 'size'), vaso_drugs=('drug', 'nunique')),
        pd.get_dummies(vw.drop_duplicates(['ICUSTAY_ID', 'drug']).set_index('ICUSTAY_ID').drug, prefix='vaso').groupby(level=0).max().astype(int),
        ow.groupby('ICUSTAY_ID').agg(urine_total=('VALUE', 'sum'), urine_n=('VALUE', 'size')),
        vent,
        agg(labw[labw.name == 'lactate'], 'VALUENUM', 'lactate'),
        agg(labw[labw.name == 'base_excess'], 'VALUENUM', 'base_excess'),
    ], axis=1).reindex(st.index)
    for c in ['vaso_n', 'vaso_drugs', 'vent_n', 'urine_n'] + [c for c in X.columns if c.startswith('vaso_') and c not in ('vaso_n', 'vaso_drugs')]:
        X[c] = X[c].fillna(0)
    X['vent'] = (X.vent_n > 0).astype(int)
    X['urine_ml_per_h'] = X.urine_total / W

    # SOFA component scores from the worst value in the window (standard cut-offs; vasopressor dose not available,
    # so dopamine/dobutamine -> 2, norepinephrine/epinephrine -> 3; urine criterion only used for the 24 h window)
    def cut(v, bins, scores):
        return pd.cut(v, bins=bins, labels=scores, right=False).astype(float)
    s = pd.DataFrame(index=X.index)
    resp = cut(X.pf_min, [-np.inf, 100, 200, 300, 400, np.inf], [4, 3, 2, 1, 0])
    resp = np.where((resp >= 3) & (X.vent == 0), 2, resp)                     # 3-4 require ventilation
    s['sofa_resp'] = resp
    s['sofa_coag'] = cut(X.platelets_min, [-np.inf, 20, 50, 100, 150, np.inf], [4, 3, 2, 1, 0])
    s['sofa_liver'] = cut(X.bilirubin_max, [-np.inf, 1.2, 2.0, 6.0, 12.0, np.inf], [0, 1, 2, 3, 4])
    cardio = np.where(X.map_min < 70, 1, 0).astype(float)
    cardio = np.where(X.get('vaso_dopamine', 0).astype(bool) | X.get('vaso_dobutamine', 0).astype(bool), np.maximum(cardio, 2), cardio)
    cardio = np.where(X.get('vaso_norepinephrine', 0).astype(bool) | X.get('vaso_epinephrine', 0).astype(bool), np.maximum(cardio, 3), cardio)
    s['sofa_cardio'] = cardio
    s['sofa_cns'] = cut(X.gcs_min, [-np.inf, 6, 10, 13, 15, np.inf], [4, 3, 2, 1, 0])
    renal = cut(X.creatinine_max, [-np.inf, 1.2, 2.0, 3.5, 5.0, np.inf], [0, 1, 2, 3, 4])
    if W >= 24:
        renal = np.maximum(renal, np.where(X.urine_total < 200, 4, np.where(X.urine_total < 500, 3, 0)))
    s['sofa_renal'] = renal
    s['sofa_total'] = s.sum(axis=1, min_count=1)
    return pd.concat([X, s], axis=1)


def cohort_mask(X, st, W):
    organs = [X.pao2_count > 0, X.fio2_count > 0, X.platelets_count > 0, X.bilirubin_count > 0,
              (X.map_count > 0) | (X.vaso_n > 0), X.gcs_count > 0, (X.creatinine_count > 0) | (X.urine_n > 0)]
    sofa_ok = np.logical_and.reduce(organs)
    died_in_w = (st.DEATHTIME >= st.INTIME) & (st.DEATHTIME <= st.INTIME + W * H)
    return sofa_ok & (X.lactate_count > 0) & (X.base_excess_count > 0) & ~died_in_w.to_numpy()

# ----------------------------------------------------------------------------------------------- cached raw subsets
LAB_ITEMS = set(LAB)
GCS_ITEMS = GCS_EYE | GCS_MOTOR | GCS_VERBAL


def ensure_raw_caches():
    if not (DATA / 'labevents_sofa.parquet').exists():
        t0, parts = time.time(), []
        for ch in pd.read_csv(RAW / 'LABEVENTS.csv.gz', usecols=['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM'], chunksize=5_000_000):
            parts.append(ch[ch.ITEMID.isin(LAB_ITEMS) & ch.VALUENUM.notna() & ch.HADM_ID.notna()])
        pd.concat(parts, ignore_index=True).to_parquet(DATA / 'labevents_sofa.parquet', index=False)
        print(f'labevents_sofa.parquet written ({time.time()-t0:.0f}s)', flush=True)
    if not (DATA / 'chartevents_gcs.parquet').exists():
        t0, parts = time.time(), []
        for i, ch in enumerate(pd.read_csv(RAW / 'CHARTEVENTS.csv.gz', usecols=['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'ERROR'],
                                           chunksize=5_000_000, low_memory=False)):
            parts.append(ch[ch.ITEMID.isin(GCS_ITEMS) & (ch.ERROR.fillna(0) != 1) & ch.VALUENUM.notna()].drop(columns='ERROR'))
            if i % 10 == 0:
                print(f'  CHARTEVENTS chunk {i}, {time.time()-t0:.0f}s', flush=True)
        pd.concat(parts, ignore_index=True).to_parquet(DATA / 'chartevents_gcs.parquet', index=False)
        print(f'chartevents_gcs.parquet written ({time.time()-t0:.0f}s)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, nargs='+', default=[6, 24])
    args = ap.parse_args()
    OUT = ROOT / 'results' / 'features'; OUT.mkdir(parents=True, exist_ok=True)
    ensure_raw_caches()
    st = load_cohort_tables()
    ev = load_events(st)
    for W in args.windows:
        X = build_features(st, ev, W)
        keep = cohort_mask(X, st, W)
        Xc, sc = X[keep], st[keep]
        Xc = Xc.join(sc[['age', 'sex']]).join(sc[['SUBJECT_ID', 'dataset', 'y28']])
        Xc.to_parquet(OUT / f'features_W{W}.parquet')
        n = sc.dataset.value_counts().to_dict()
        print(f"W={W} h: cohort {len(Xc):,} stays (train/val/test {n['train']}/{n['val']}/{n['test']}), "
              f"28-day mortality {sc.y28.mean():.3f}, {Xc.shape[1]-5} features -> {OUT / f'features_W{W}.parquet'}", flush=True)


if __name__ == '__main__':
    main()
