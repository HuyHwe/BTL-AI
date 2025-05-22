import pandas as pd, numpy as np
cols4738 = pd.read_csv('feature_names.csv', header=None)[0].tolist()
df = pd.read_csv('dummies.csv', usecols=cols4738)
np.save('scaler_mean.npy', df.mean().to_numpy(float))
np.save('scaler_scale.npy', df.std(ddof=0).replace(0,1).to_numpy(float))

