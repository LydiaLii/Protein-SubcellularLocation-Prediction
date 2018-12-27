import pandas as pd
import re

df = pd.DataFrame(pd.read_csv('./data/original.csv'))
# print(df[['Sequence', 'Function [CC]']].head(10))
coi = df['Function [CC]'].str.replace('FUNCTION: ', '')
# coi = df['Sequence']
coi = coi.str.split('\. ')

func_set = []
filter_rules = [
    '\.$',
    ' \(Probable\)$',
    ' \(By similarity\)$',
    ' \(PubMed.*?\)$'
]
for row in range(coi.shape[0]):  # coi.shape[0]
    func_list = coi.iloc[row]
    try:
        func_clean = []
        for item in func_list:
            for rule in filter_rules:
                r = re.search(rule, item)
                if r != None:
                    item = item[:r.span()[0]]
            if (item[0] != '{') and (len(item)>6):
                func_clean.append(item)
        func_set += func_clean
    except TypeError:
        continue

func_set = sorted(set(func_set))
# print(type(func_set))
func_df = pd.Series(list(func_set))
func_df.to_csv('./data/functions.csv')

# for item in func_set:
#     if item[0] != '{':
#         print(item)
