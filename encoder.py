import numpy as np

def _nearest(val, table: dict[float, float]):
    if val is None or not table:
        return None
    key = min(table, key=lambda k: abs(k - val))
    return table[key]

def encode_row(form, feat_list, mapping, mapping_mean, num_mapping):
    vec = []
    for col in feat_list:
        if col.endswith('_target_encoded'):
            raw = col[:-15]
            if raw in num_mapping:                      # 2 cột nhập số
                try:
                    num = float(form.get(raw, ''))
                except ValueError:
                    num = None
                enc = _nearest(num, num_mapping[raw]) or mapping_mean[raw]
            else:
                txt = str(form.get(raw, ''))
                enc = mapping[raw].get(txt, mapping_mean[raw])
            vec.append(enc)
        else:                                          # mileage, year
            try:
                vec.append(float(form.get(col, 0)))
            except ValueError:
                vec.append(0.0)
    return np.array(vec, dtype=np.float64)
