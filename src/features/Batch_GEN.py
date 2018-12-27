import pandas as pd
from src.features.Main_Feature import MainFeatures
from src.features.History import SaveWithHistory
import time

df = pd.DataFrame(pd.read_csv('../../data/original.csv'))

lines = df.shape[0]
e = df['Entry'].tolist()
n = df['Entry name'].tolist()
s = df['Sequence'].tolist()
c = df['Subcellular location [CC]'].tolist()

sh = SaveWithHistory('../../data/features/', MainFeatures().generate_keys())

for i in range(40):
    start = time.time()
    print('Entry %d [%s] started.' % (i+1, e[i]))
    mf = MainFeatures(e[i], n[i], s[i])
    sh.add_line(e[i], i, mf.generate_features())
    print('Finished in [%.1f]s.' % (time.time()-start))
