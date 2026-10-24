"""Extract adult-ICU (non-NICU) rows for the ITEMIDs we need from the big MIMIC-III tables.

Usage: MIMIC3_RAW=/path/to/mimic-iii-clinical-database-1.4 python scripts/01_extract_events.py [TABLE ...]

Output (data/):
  icustays.parquet            adult ICU stays + PATIENTS/ADMISSIONS static columns
  chartevents.parquet         selected ITEMIDs only
  labevents_be.parquet        base excess (50802)
  inputevents.parquet         CV + MV inputs (amount)
  outputevents.parquet
  procedureevents_intub.parquet  MetaVision intubation procedure (224385)
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd

RAW = Path(os.environ.get('MIMIC3_RAW', 'data/mimic-iii-clinical-database-1.4'))   # folder with the PhysioNet csv.gz files
OUT = Path(__file__).resolve().parents[1] / 'data'
OUT.mkdir(exist_ok=True)

# concept -> itemids (carevue + metavision). Same concepts as the original NICU notebook where an adult analog exists.
ITEMS = {
    'HR': [211, 220045],
    'SBP': [51, 442, 455, 6701, 220179, 220050],
    'DBP': [8368, 8440, 8441, 8555, 220180, 220051],
    'MeanBP': [456, 52, 6702, 443, 220052, 220181, 225312],
    'RR': [615, 618, 220210, 224690],
    'TempF': [223761, 678], 'TempC': [223762, 676],
    'SpO2': [646, 220277],
    'FiO2': [190, 3420, 3422, 223835],
    'Weight': [762, 763, 3723, 3580, 224639, 226512],
    'Height': [920, 1394, 4187, 3486, 226707, 226730],
    'O2Flow': [470, 471, 223834],
    # mechanical-ventilation settings (MIMIC-code ventilation_classification)
    'VENT': [720, 223849, 223848, 445, 448, 449, 450, 1340, 1486, 1600, 224687, 639, 654, 681, 682, 683, 684, 224685,
             224684, 224686, 218, 436, 535, 444, 224697, 224695, 224696, 224746, 224747, 221, 1, 1211, 1655, 2000,
             226873, 224738, 224419, 224750, 227187, 543, 5865, 5866, 224707, 224709, 224705, 224706, 60, 437, 505,
             506, 686, 220339, 224700, 3459, 501, 502, 503, 224702, 223, 667, 668, 669, 670, 671, 672, 224701],
    'O2Device': [467, 226732],
    'Extubated': [640],
}
TEXT_ITEMS = {720, 223849, 223848, 467, 226732, 640}  # keep VALUE string for these only
ALL_ITEMS = {i for v in ITEMS.values() for i in v}


def build_static():
    icu = pd.read_csv(RAW / 'ICUSTAYS.csv.gz')
    icu = icu[icu.FIRST_CAREUNIT != 'NICU']
    pat = pd.read_csv(RAW / 'PATIENTS.csv.gz', usecols=['SUBJECT_ID', 'GENDER', 'DOB'])
    adm = pd.read_csv(RAW / 'ADMISSIONS.csv.gz', usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'ADMISSION_TYPE'])
    st = icu.merge(pat, on='SUBJECT_ID').merge(adm, on=['SUBJECT_ID', 'HADM_ID'])
    st.to_parquet(OUT / 'icustays.parquet', index=False)
    print(f'adult ICU stays {len(st)}  subjects {st.SUBJECT_ID.nunique()}', flush=True)
    return st


def extract(table, out_name, keep_fn, usecols, chunksize=5_000_000):
    t0 = time.time()
    parts, n_in = [], 0
    for i, chunk in enumerate(pd.read_csv(RAW / f'{table}.csv.gz', usecols=usecols, chunksize=chunksize, low_memory=False)):
        n_in += len(chunk)
        parts.append(keep_fn(chunk))
        if i % 10 == 0:
            print(f'  {table}: {n_in/1e6:.0f}M read, {sum(map(len, parts))/1e6:.2f}M kept, {time.time()-t0:.0f}s', flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(OUT / f'{out_name}.parquet', index=False)
    print(f'{table}: kept {len(df)} rows -> {out_name}.parquet ({time.time()-t0:.0f}s)', flush=True)


def main():
    st = build_static()
    stays, hadms = set(st.ICUSTAY_ID), set(st.HADM_ID)
    tables = sys.argv[1:] or ['CHARTEVENTS', 'LABEVENTS', 'INPUTEVENTS_CV', 'INPUTEVENTS_MV', 'OUTPUTEVENTS', 'PROCEDUREEVENTS_MV']

    def keep_chart(c):
        c = c[c.ITEMID.isin(ALL_ITEMS) & c.ICUSTAY_ID.isin(stays) & (c.ERROR.fillna(0) != 1)]
        c = c.drop(columns='ERROR')
        c.loc[~c.ITEMID.isin(TEXT_ITEMS), 'VALUE'] = None
        return c

    for t in tables:
        if t == 'CHARTEVENTS':
            extract(t, 'chartevents', keep_chart, ['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'VALUENUM', 'VALUEUOM', 'ERROR'])
        elif t == 'LABEVENTS':
            extract(t, 'labevents_be', lambda c: c[(c.ITEMID == 50802) & c.HADM_ID.isin(hadms)],
                    ['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM'])
        elif t == 'INPUTEVENTS_CV':
            extract(t, 'inputevents_cv', lambda c: c[c.ICUSTAY_ID.isin(stays) & c.AMOUNT.notna()],
                    ['ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'AMOUNT', 'AMOUNTUOM'])
        elif t == 'INPUTEVENTS_MV':
            extract(t, 'inputevents_mv', lambda c: c[c.ICUSTAY_ID.isin(stays) & c.AMOUNT.notna() & (c.STATUSDESCRIPTION != 'Rewritten')],
                    ['ICUSTAY_ID', 'STARTTIME', 'ENDTIME', 'ITEMID', 'AMOUNT', 'AMOUNTUOM', 'STATUSDESCRIPTION'])
        elif t == 'OUTPUTEVENTS':
            extract(t, 'outputevents', lambda c: c[c.ICUSTAY_ID.isin(stays) & c.VALUE.notna()],
                    ['ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM'])
        elif t == 'PROCEDUREEVENTS_MV':
            extract(t, 'procedureevents_intub', lambda c: c[(c.ITEMID == 224385) & c.ICUSTAY_ID.isin(stays)],
                    ['ICUSTAY_ID', 'STARTTIME', 'ENDTIME', 'ITEMID'])
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
