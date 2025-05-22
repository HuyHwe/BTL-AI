import pandas as pd, numpy as np, pathlib, sys

CSV     = 'dummies.csv'
WEIGHTS = 'weights.dat'
OUT     = 'feature_names.csv'

n_w = np.fromfile(WEIGHTS, dtype=np.float64).size
print(f'→ weights length  = {n_w}')

cols = pd.read_csv(CSV, nrows=0).columns.tolist()
print(f'→ header columns  = {len(cols)}')

if len(cols) < n_w:
    sys.exit('Header ít cột hơn weight – sai file dummies.csv?')
if len(cols) > n_w:
    print(f'Header dư {len(cols)-n_w} cột; tự cắt đuôi để khớp weight.')
    cols = cols[:n_w]

pd.Series(cols).to_csv(OUT, index=False, header=False)
print(f'✓ Đã tạo {OUT} với {len(cols)} feature – khớp EXACT weights!')
