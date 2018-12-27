import pandas as pd
import time
import numpy as np

feature_file = '../../data/LX_features/20180611_223038_[284 columns].csv'

df = pd.DataFrame(pd.read_csv(feature_file))
columns = df.columns.values.tolist()

scale_list = [0, 0, 0]

start = time.time()
for c in columns[3:]:
    series = np.array(df[c])
    mean = np.mean(series)
    median = np.median(series)
    scale = 0.5 / (mean*0.8 + median*0.2)
    adjusted = series * scale
    df[c] = pd.Series(adjusted)
    scale_list.append(scale)
print('Adjust finished in %.3f s.' % (time.time() - start))

df_col = pd.DataFrame([scale_list], columns=columns)
df.to_csv(feature_file[:-4] + '_adjusted.csv')
df_col.to_csv(feature_file[:-4] + '_scale.csv')
# print(df.head())
