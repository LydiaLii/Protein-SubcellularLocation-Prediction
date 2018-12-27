import pandas as pd

df = pd.DataFrame(pd.read_csv('./data/original.csv'))
coi = df['Sequence']

coi.to_csv('./data/sequence.csv')
