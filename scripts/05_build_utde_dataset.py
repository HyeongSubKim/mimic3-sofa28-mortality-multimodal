"""Build the UTDE (MultimodalMIMIC, ICML 2023) input pickles for the cohort of 03_build_features.py.

For every stay in results/features/features_W{W}.parquet (same cohort, same patient split, same 28-day label):
  name              ICUSTAY_ID
  irg_ts            l_ts x d_m   hourly means of the observed channels (z-scored with train statistics), rows = hours with >= 1 value
  irg_ts_mask       l_ts x d_m   1 where observed
  ts_tt             hours since ICU admission of each row (labs drawn in the 6 h before admission are placed at t = 0)
  reg_ts            tt_max x 2 d_m  forward-filled hourly grid + masks (recomputed by the UTDE loader anyway)
  text_data         notes available by INTIME + W (STORETIME for bedside notes, CHARTTIME for radiology), time-ordered;
                    categories: Nursing/other, Nursing, Physician, Respiratory, General, Radiology; de-identified tokens removed
  text_time_to_end  W - hours(note availability - INTIME), clipped to [0, W]  (UTDE uses 1 - t / tt_max as the note time)
  label             28-day mortality
Stays without any note get one placeholder note so that the cohort stays identical to the tabular experiments (the UTDE
loader drops instances with an empty note list).
Output: <out>/W{W}/{train,val,test}p2x_data.pkl + channels.json
Usage: MIMIC3_RAW=... python scripts/05_build_utde_dataset.py --windows 6 24 --out ../MultimodalMIMIC/Data/sofa28
"""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RAW = Path(os.environ.get('MIMIC3_RAW', 'data/mimic-iii-clinical-database-1.4'))
H = pd.Timedelta(1, 'h')
NOTE_CATS = ['Nursing/other', 'Nursing', 'Physician', 'Respiratory', 'General', 'Radiology']
PLACEHOLDER = 'No clinical note available in this window.'

CHART = {'HR': {211, 220045}, 'RR': {615, 618, 220210, 224690}, 'SpO2': {646, 220277},
         'TempC': {223762, 676}, 'TempF': {223761, 678},
         'SBP': {442, 455, 220179}, 'DBP': {8440, 8441, 220180}, 'MeanBP': {456, 443, 220181},
         'ASBP': {51, 6701, 220050}, 'ADBP': {8368, 8555, 220051}, 'MeanABP': {52, 6702, 220052, 225312},
         'FiO2': {190, 3420, 3422, 223835}, 'O2flow': {470, 471, 223834}}
LAB = {50821: 'PaO2', 51265: 'Platelets', 50885: 'Bilirubin', 50912: 'Creatinine', 50813: 'Lactate', 50802: 'BaseExcess'}
VASO = {30047, 30120, 30044, 30119, 30309, 30127, 30128, 30042, 30043, 30306, 30307, 30125,
        221906, 221289, 221662, 221653, 221749, 222315}
GCS = {184, 220739, 454, 223901, 723, 223900}
RANGES = {'HR': (0, 300), 'RR': (0, 100), 'SpO2': (0, 100), 'Temp': (25, 45), 'SBP': (0, 400), 'DBP': (0, 300), 'MeanBP': (0, 300),
          'ASBP': (0, 400), 'ADBP': (0, 300), 'MeanABP': (0, 300), 'FiO2': (21, 100), 'O2flow': (0, 100), 'PaO2': (0, 800),
          'Platelets': (0, 2000), 'Bilirubin': (0, 100), 'Creatinine': (0, 50), 'Lactate': (0, 40), 'BaseExcess': (-50, 50),
          'GCS': (3, 15), 'Urine': (0, 5000)}
CHANNELS = ['HR', 'RR', 'SpO2', 'Temp', 'SBP', 'DBP', 'MeanBP', 'ASBP', 'ADBP', 'MeanABP', 'FiO2', 'O2flow', 'PaO2', 'Platelets',
            'Bilirubin', 'Creatinine', 'Lactate', 'BaseExcess', 'GCS', 'Urine', 'Vasopressor']
# --channel-set sofa: the same variables as the tabular model (SOFA inputs + lactate + BE + ventilation + age/sex), no extra vitals
SOFA_CHANNELS = ['FiO2', 'PaO2', 'Platelets', 'Bilirubin', 'Creatinine', 'MeanBP', 'MeanABP', 'GCS', 'Urine', 'Vasopressor',
                 'Lactate', 'BaseExcess', 'Vent', 'Age', 'Sex']
VENT_ITEMS = {720, 223849, 223848, 445, 448, 449, 450, 1340, 1486, 1600, 224687, 639, 654, 681, 682, 683, 684, 224685,
              224684, 224686, 218, 436, 535, 444, 224697, 224695, 224696, 224746, 224747, 221, 1, 1211, 1655, 2000,
              226873, 224738, 224419, 224750, 227187, 543, 5865, 5866, 224707, 224709, 224705, 224706, 60, 437, 505,
              506, 686, 220339, 224700, 3459, 501, 502, 503, 224702, 223, 667, 668, 669, 670, 671, 672, 224701}


def hourly_long(st, W):
    """long table (ICUSTAY_ID, hour, channel, value) for hours 0..W-1 (labs from -6 h placed at hour 0)."""
    ids = set(st.index)
    frames = []
    ce = pd.read_parquet(DATA / 'chartevents.parquet', columns=['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM'])
    ce = ce[ce.ICUSTAY_ID.isin(ids) & ce.VALUENUM.notna()].copy(); ce['ICUSTAY_ID'] = ce.ICUSTAY_ID.astype('int64')
    item2ch = {i: c for c, s in CHART.items() for i in s}
    ce = ce[ce.ITEMID.isin(item2ch)]; ce['channel'] = ce.ITEMID.map(item2ch)
    ce.loc[ce.channel == 'TempF', 'VALUENUM'] = (ce.loc[ce.channel == 'TempF', 'VALUENUM'] - 32) * 5 / 9
    ce['channel'] = ce.channel.replace({'TempF': 'Temp', 'TempC': 'Temp'})
    m = ce.channel.eq('FiO2') & (ce.VALUENUM <= 1); ce.loc[m, 'VALUENUM'] *= 100
    frames.append(ce[['ICUSTAY_ID', 'CHARTTIME', 'channel', 'VALUENUM']].rename(columns={'CHARTTIME': 't', 'VALUENUM': 'value'}))
    cev = pd.read_parquet(DATA / 'chartevents.parquet', columns=['ICUSTAY_ID', 'ITEMID', 'CHARTTIME'])
    cev = cev[cev.ITEMID.isin(VENT_ITEMS) & cev.ICUSTAY_ID.isin(ids)].copy(); cev['ICUSTAY_ID'] = cev.ICUSTAY_ID.astype('int64')
    cev['channel'] = 'Vent'; cev['value'] = 1.0
    frames.append(cev[['ICUSTAY_ID', 'CHARTTIME', 'channel', 'value']].rename(columns={'CHARTTIME': 't'}))
    lab = pd.read_parquet(DATA / 'labevents_sofa.parquet').merge(st.reset_index()[['HADM_ID', 'ICUSTAY_ID']], on='HADM_ID')
    lab = lab[lab.ITEMID.isin(LAB)].copy(); lab['channel'] = lab.ITEMID.map(LAB)
    frames.append(lab[['ICUSTAY_ID', 'CHARTTIME', 'channel', 'VALUENUM']].rename(columns={'CHARTTIME': 't', 'VALUENUM': 'value'}))
    g = pd.read_parquet(DATA / 'chartevents_gcs.parquet'); g = g[g.ICUSTAY_ID.isin(ids)].copy(); g['ICUSTAY_ID'] = g.ICUSTAY_ID.astype('int64')
    g['comp'] = np.select([g.ITEMID.isin({184, 220739}), g.ITEMID.isin({454, 223901})], ['e', 'm'], default='v')
    g = g.pivot_table(index=['ICUSTAY_ID', 'CHARTTIME'], columns='comp', values='VALUENUM', aggfunc='first').dropna().reset_index()
    g['value'] = g.e + g.m + g.v; g['channel'] = 'GCS'
    frames.append(g[['ICUSTAY_ID', 'CHARTTIME', 'channel', 'value']].rename(columns={'CHARTTIME': 't'}))
    out = pd.read_parquet(DATA / 'outputevents.parquet', columns=['ICUSTAY_ID', 'CHARTTIME', 'VALUE'])
    out = out[out.ICUSTAY_ID.isin(ids) & out.VALUE.notna()].copy(); out['ICUSTAY_ID'] = out.ICUSTAY_ID.astype('int64')
    out['channel'] = 'Urine'
    frames.append(out.rename(columns={'CHARTTIME': 't', 'VALUE': 'value'})[['ICUSTAY_ID', 't', 'channel', 'value']])
    icv = pd.read_parquet(DATA / 'inputevents_cv.parquet', columns=['ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'AMOUNT']).rename(columns={'CHARTTIME': 't'})
    imv = pd.read_parquet(DATA / 'inputevents_mv.parquet', columns=['ICUSTAY_ID', 'STARTTIME', 'ITEMID', 'AMOUNT']).rename(columns={'STARTTIME': 't'})
    v = pd.concat([icv, imv]); v = v[v.ITEMID.isin(VASO) & (v.AMOUNT > 0) & v.ICUSTAY_ID.isin(ids)].copy()
    v['ICUSTAY_ID'] = v.ICUSTAY_ID.astype('int64'); v['channel'] = 'Vasopressor'; v['value'] = 1.0
    frames.append(v[['ICUSTAY_ID', 't', 'channel', 'value']])
    L = pd.concat(frames, ignore_index=True); L['t'] = pd.to_datetime(L.t)
    L['h'] = (L.t - L.ICUSTAY_ID.map(st.INTIME)) / H
    is_lab = L.channel.isin(list(LAB.values()))
    L = L[((L.h >= 0) | (is_lab & (L.h >= -6))) & (L.h < W)].copy()
    L['hour'] = np.floor(L.h.clip(lower=0)).astype(int)
    for ch, (lo, hi) in RANGES.items():
        m = L.channel.eq(ch) & ((L.value < lo) | (L.value > hi)); L.loc[m, 'value'] = np.nan
    L = L.dropna(subset=['value'])
    agg = L.groupby(['ICUSTAY_ID', 'hour', 'channel']).value.agg(lambda s: s.sum() if s.name == 'Urine' else s.mean())
    # urine: sum per hour, vasopressor: any; everything else: mean  (done channel-wise below)
    sums = L[L.channel == 'Urine'].groupby(['ICUSTAY_ID', 'hour', 'channel']).value.sum()
    anys = L[L.channel.isin(['Vasopressor', 'Vent'])].groupby(['ICUSTAY_ID', 'hour', 'channel']).value.max()
    means = L[~L.channel.isin(['Urine', 'Vasopressor', 'Vent'])].groupby(['ICUSTAY_ID', 'hour', 'channel']).value.mean()
    return pd.concat([means, sums, anys]).reset_index()


def notes_for(st, W):
    nt = pd.read_parquet(DATA / 'noteevents_text.parquet')
    nt = nt[nt.CATEGORY.isin(NOTE_CATS) & nt.HADM_ID.isin(set(st.HADM_ID))].merge(st.reset_index()[['HADM_ID', 'ICUSTAY_ID', 'INTIME']], on='HADM_ID')
    avail = pd.to_datetime(nt.STORETIME).fillna(pd.to_datetime(nt.CHARTTIME))
    nt['h'] = (avail - nt.INTIME) / H
    nt = nt[(nt.h > -48) & (nt.h <= W)].sort_values(['ICUSTAY_ID', 'h'])
    nt['tte'] = (W - nt.h).clip(lower=0, upper=W)
    nt['TEXT'] = nt.CATEGORY + ' note: ' + nt.TEXT
    return {k: (g.TEXT.tolist(), g.tte.tolist()) for k, g in nt.groupby('ICUSTAY_ID')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, nargs='+', default=[6, 24])
    ap.add_argument('--features-dir', default='results/features')
    ap.add_argument('--out', default='../MultimodalMIMIC/Data/sofa28')
    ap.add_argument('--channel-set', default='full', choices=['full', 'sofa'], help='full = 21 benchmark channels; sofa = the tabular model\'s variables')
    args = ap.parse_args()
    chans = SOFA_CHANNELS if args.channel_set == 'sofa' else CHANNELS
    icu = pd.read_parquet(DATA / 'icustays.parquet', columns=['ICUSTAY_ID', 'HADM_ID', 'INTIME']).set_index('ICUSTAY_ID')
    icu['INTIME'] = pd.to_datetime(icu.INTIME)
    for W in args.windows:
        f = pd.read_parquet(ROOT / args.features_dir / f'features_W{W}.parquet', columns=['SUBJECT_ID', 'dataset', 'y28', 'age', 'sex'])
        st = f.join(icu)
        long = hourly_long(st, W)
        long = long[long.channel.isin(chans)]
        if args.channel_set == 'sofa':                     # age / sex as constant channels on every observed hour of the stay
            hrs = long[['ICUSTAY_ID', 'hour']].drop_duplicates()
            for name, col in (('Age', 'age'), ('Sex', 'sex')):
                long = pd.concat([long, hrs.assign(channel=name, value=hrs.ICUSTAY_ID.map(st[col]).astype(float))], ignore_index=True)
        notes = notes_for(st, W)
        # z-score with train statistics (Vasopressor stays binary)
        tr_ids = set(st.index[st.dataset == 'train'])
        binary = ['Vasopressor', 'Vent', 'Sex']
        stats = long[long.ICUSTAY_ID.isin(tr_ids) & ~long.channel.isin(binary)].groupby('channel').value.agg(['mean', 'std'])
        long = long.merge(stats, left_on='channel', right_index=True, how='left')
        long['z'] = np.where(long.channel.isin(binary), long.value, (long.value - long['mean']) / long['std'].replace(0, 1))
        ch_idx = {c: i for i, c in enumerate(chans)}
        out_dir = Path(args.out) / f'W{W}'; out_dir.mkdir(parents=True, exist_ok=True)
        counts = {}
        for split in ['train', 'val', 'test']:
            inst = []
            ids = st.index[st.dataset == split]
            grp = {k: g for k, g in long[long.ICUSTAY_ID.isin(set(ids))].groupby('ICUSTAY_ID')}
            for sid in ids:
                g = grp.get(sid)
                if g is None or g.empty:
                    continue
                hours = np.sort(g.hour.unique())
                X = np.zeros((len(hours), len(chans)), dtype=np.float32); M = np.zeros_like(X, dtype=np.int64)
                hrow = {h: i for i, h in enumerate(hours)}
                for r in g.itertuples(index=False):
                    X[hrow[r.hour], ch_idx[r.channel]] = r.z; M[hrow[r.hour], ch_idx[r.channel]] = 1
                reg = np.zeros((W, 2 * len(chans)), dtype=np.float32)
                last = np.zeros(len(chans), dtype=np.float32)
                for h in range(W):
                    if h in hrow:
                        m = M[hrow[h]] == 1; last[m] = X[hrow[h]][m]; reg[h, len(chans):] = m
                    reg[h, :len(chans)] = last
                txt, tte = notes.get(sid, ([PLACEHOLDER], [0.0]))
                # plain lists: the UTDE environment runs numpy 1.21, which cannot unpickle numpy-2 arrays (numpy._core)
                inst.append({'name': int(sid), 'irg_ts': X.tolist(), 'irg_ts_mask': M.tolist(), 'ts_tt': hours.astype(float).tolist(), 'reg_ts': reg.tolist(),
                             'text_data': txt, 'text_time_to_end': [float(t) for t in tte], 'label': int(st.loc[sid, 'y28'])})
            pickle.dump(inst, open(out_dir / f'{split}p2x_data.pkl', 'wb'))
            counts[split] = {'n': len(inst), 'pos': int(sum(i['label'] for i in inst)),
                             'with_note': int(sum(i['text_data'][0] != PLACEHOLDER for i in inst)),
                             'notes_per_stay_median': float(np.median([len(i['text_data']) for i in inst])),
                             'ts_rows_median': float(np.median([len(i['ts_tt']) for i in inst]))}
            print(W, split, counts[split], flush=True)
        json.dump({'channels': chans, 'tt_max': W, 'train_stats': stats.to_dict(), 'counts': counts}, open(out_dir / 'channels.json', 'w'), indent=1)
    print('saved', args.out)


if __name__ == '__main__':
    main()
