import pandas as pd
import re
import datetime

df = pd.DataFrame(pd.read_csv('../../data/original.csv'))

lines = df.shape[0]
c = df['Subcellular location [CC]'].tolist()
e = df['Entry'].tolist()

target_loc = ['Endoplasmic reticulum membrane', 'Cell projection', 'Cell membrane', 'Cell junction', 'Cytoplasm',
              'Nucleus', 'Secreted', 'Cytoskeleton', 'Mitochondrion', 'Membrane']


def categorize(loc_str):
    for loc in target_loc:
        if loc in loc_str:
            return loc, target_loc.index(loc)
    return ''


now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

with open('../../data/label/label_'+now_str+'.csv', 'w') as f:
    f.write('Index,Entry,Location,Category,Label\n')
    for i in range(lines):
        flag = ''
        item = c[i]
        if not isinstance(item, str) or (len(item) < 10):
            continue
        try:
            first = re.findall('SUBCELLULAR LOCATION: (.*?: )?(\w*?\s?\w*?\s?\w*?)(,|\.|;| {ECO)', item)[0]
            category = categorize(first[1])
            # f.write('%d,%s,%s,%s,\n' % (i, e[i], first[1], category))
            if category == '':
                continue
            f.write('%d,%s,%s,%s,%d\n' % (i, e[i], first[1], category[0], category[1]))
        except IndexError:
            continue
            # f.write('%d,%s,"%s",%d\n' % (i, e[i], item, 1))
