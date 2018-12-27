import pandas as pd
import re
from time import time


def if_dups(func_name):
    for rule in filter_rules:
        r = re.search(rule, func_name)
        if r is not None:
            func_name = func_name[:r.span()[0]]
            if func_clean.setdefault(func_name, 'NotInDict') != 'NotInDict':
                return 0
    return 1


df = pd.DataFrame(pd.read_csv('./data/original.csv'))
coi = df['Function [CC]'].str.replace('FUNCTION: ', '')
coi = coi.str.split('\. ')
coi = pd.DataFrame(coi)
coi.reset_index(inplace=True)

filter_rules = [
    '\.$',
    ' \(Probable\)$',
    ' \(By similarity\)$',
    ' \(PubMed.*?\)$'
]

func_clean = {}
for row in range(coi.shape[0]):
    if row % 4000 == 0:
        print('Processing row %d...' % row)
    func_list = coi.iloc[row]['Function [CC]']
    try:
        for item in func_list:
            # for rule in filter_rules:
            #     r = re.search(rule, item)
            #     if r is not None:
            #         item = item[:r.span()[0]]
            if (item[0] != '{') and (len(item) > 6):
                if item not in func_clean.keys():
                    func_clean[item] = [row]
                else:
                    func_clean[item].append(row)
    except TypeError:
        continue

keys = pd.Series(list(func_clean.keys()))
values = pd.Series(list(func_clean.values()))

pair = pd.DataFrame()

pair['function'] = keys
pair['sequence'] = values

pair.sort_values(by='function', inplace=True)
pair.reset_index(drop=True, inplace=True)

marker = -1
mk = [0 for _ in range(pair.shape[0])]
for row in range(pair.shape[0]):
    if row % 4000 == 0:
        print('Processing row %d...' % row)
    mk[row] = marker
    func = pair.iloc[row]['function']
    marker += if_dups(func)

pair['marker'] = pd.Series(mk[1:])

# print(pair.head(100))
pair.to_csv('./data/pair_marker.csv')
