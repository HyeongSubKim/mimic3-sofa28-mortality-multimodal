"""Extract the full text of NOTEEVENTS rows that belong to adult ICU admissions (HADM_IDs of data/icustays.parquet),
excluding discharge summaries, ISERROR rows and date-only notes. Cached to data/noteevents_text.parquet
(ROW_ID, SUBJECT_ID, HADM_ID, CHARTTIME, STORETIME, CATEGORY, DESCRIPTION, TEXT). ~5 min, ~1.5 GB parquet.
"""
import os, re, time
from pathlib import Path
import pandas as pd

RAW = Path(os.environ.get('MIMIC3_RAW', 'data/mimic-iii-clinical-database-1.4'))
ROOT = Path(__file__).resolve().parents[1]
hadm = set(pd.read_parquet(ROOT / 'data' / 'icustays.parquet', columns=['HADM_ID']).HADM_ID)
DEID = re.compile(r'\[\*\*[^\]]*\*\*\]')
t0, parts, n = time.time(), [], 0
for i, ch in enumerate(pd.read_csv(RAW / 'NOTEEVENTS.csv.gz', chunksize=100_000, low_memory=False)):
    ch = ch[ch.HADM_ID.isin(hadm) & ch.ISERROR.isna() & (ch.CATEGORY.str.strip() != 'Discharge summary') & ch.CHARTTIME.notna()]
    ch = ch[['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'STORETIME', 'CATEGORY', 'DESCRIPTION', 'TEXT']].copy()
    ch['CATEGORY'] = ch.CATEGORY.str.strip()
    ch['TEXT'] = ch.TEXT.str.replace(DEID, ' ', regex=True).str.replace(r'[ \t]+', ' ', regex=True).str.replace(r'\n{3,}', '\n\n', regex=True).str.strip()
    parts.append(ch); n += len(ch)
    if i % 5 == 0:
        print(f'chunk {i}: kept {n:,} notes, {time.time()-t0:.0f}s', flush=True)
df = pd.concat(parts, ignore_index=True)
df['HADM_ID'] = df.HADM_ID.astype('int64')
df.to_parquet(ROOT / 'data' / 'noteevents_text.parquet', index=False)
print(f'saved {len(df):,} notes for {df.HADM_ID.nunique():,} admissions ({time.time()-t0:.0f}s)')
