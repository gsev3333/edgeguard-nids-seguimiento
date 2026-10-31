import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

RAW_DIR = sys.argv[sys.argv.index('--input-dir')+1] if '--input-dir' in sys.argv else 'data/raw/cicids2017'
OUT_DIR = sys.argv[sys.argv.index('--output-dir')+1] if '--output-dir' in sys.argv else 'data/processed_real'

os.makedirs(OUT_DIR, exist_ok=True)
import glob
csvs = glob.glob(os.path.join(RAW_DIR, '*.csv'))
print('Found CSVs:', len(csvs))
for c in csvs:
    print(' -', os.path.basename(c))
if not csvs:
    print('NO_CSV_FOUND')
    sys.exit(1)

dfs = []
for c in csvs:
    try:
        df = pd.read_csv(c, low_memory=False)
        dfs.append(df)
    except Exception as e:
        print('ERR', c, e)
df = pd.concat(dfs, ignore_index=True)
df.columns = [str(c).strip() for c in df.columns]
if 'Label' in df.columns:
    df['Label'] = df['Label'].astype(str).str.strip()
    df['y'] = df['Label'].apply(lambda x: 0 if x.upper() == 'BENIGN' else 1)
else:
    print('NO_LABEL')
    sys.exit(1)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
feat_cols = [c for c in df.columns if c not in ['Label','y']]
Xdf = df[feat_cols].select_dtypes(include=[np.number]).copy()
X = Xdf.values.astype(np.float32)
y = df['y'].values.astype(np.int32)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
np.save(os.path.join(OUT_DIR,'X_train.npy'), X_train)
np.save(os.path.join(OUT_DIR,'X_test.npy'), X_test)
np.save(os.path.join(OUT_DIR,'y_train.npy'), y_train)
np.save(os.path.join(OUT_DIR,'y_test.npy'), y_test)
joblib.dump(scaler, os.path.join(OUT_DIR,'scaler.joblib'))
with open(os.path.join(OUT_DIR,'feature_names.txt'),'w') as f:
    for c in Xdf.columns:
        f.write(c+'\n')
print('DONE', X_train.shape, X_test.shape, np.bincount(y_train), np.bincount(y_test))
