from flask import Flask, render_template, request, jsonify
import pandas as pd, numpy as np, re, pathlib

RAW_CSV = 'car_detail_cleaned.csv'   # file gốc (chưa encode)
ENC_CSV = 'dataset.csv'              # file đã target-encode

raw_df = pd.read_csv(RAW_CSV)
enc_df = pd.read_csv(ENC_CSV)

FEATURES = [c for c in enc_df.columns if c not in ['Price', 'log_price']]
WEIGHTS  = np.fromfile('weights.dat', dtype=np.float64)
BIAS     = np.fromfile('bias.dat',   dtype=np.float64)[0]
assert len(FEATURES) == len(WEIGHTS), 'Chiều FEATURES ≠ WEIGHTS!'

MEAN = enc_df[FEATURES].mean().to_numpy(float)
STD  = enc_df[FEATURES].std(ddof=0).replace(0, 1).to_numpy(float)

mapping, mapping_mean, num_mapping = {}, {}, {}
def _num(txt):
    m = re.search(r'[\d.]+', str(txt))
    return float(m.group()) if m else None

for col in FEATURES:
    if col.endswith('_target_encoded'):
        raw_col = col[:-15]
        pair = raw_df[raw_col].astype(str).to_frame().join(enc_df[col])
        mp = dict(zip(pair[raw_col], pair[col]))
        mapping[raw_col] = mp
        mapping_mean[raw_col] = float(enc_df[col].mean())
        if raw_col in {'fuel_consumption', 'seating_capacity'}:       # 2 cột nhập số
            num_mapping[raw_col] = {_num(k): v for k, v in mp.items() if _num(k) is not None}

NUM_FIELDS = {'mileage', 'year_of_manufacture', 'fuel_consumption', 'seating_capacity'}
OPTIONS = {c: sorted(v) for c, v in mapping.items() if c not in NUM_FIELDS}
PLACEHOLDER = {
    'mileage': float(raw_df['mileage'].mean()),
    'year_of_manufacture': float(raw_df['year_of_manufacture'].mean()),
    'fuel_consumption': np.mean(list(num_mapping['fuel_consumption'].keys())),
    'seating_capacity': np.mean(list(num_mapping['seating_capacity'].keys())),
}

# ---------- FLASK ----------
app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/metadata')
def metadata():
    return jsonify({**OPTIONS, **PLACEHOLDER})

@app.route('/predict', methods=['POST'])
def predict():
    from encoder import encode_row
    try:
        data = request.json or {}
        x = encode_row(data, FEATURES, mapping, mapping_mean, num_mapping)
        x_scaled = (x - MEAN) / STD
        log_p = float(np.dot(x_scaled, WEIGHTS) + BIAS)
        return jsonify({'ok': True, 'price': round(np.exp(log_p))})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
