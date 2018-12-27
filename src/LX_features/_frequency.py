import re
import pandas as pd
import json


class PolypeptideFrequency:

    def __init__(self, content):
        self.content = content

    def find(self, percentage):
        result_dict = {}
        for key, value in target_dict.items():
            tmp = []
            for _class in value:
                partial_con = self.content[:int(len(self.content) * percentage)]
                x = re.findall(_class, partial_con)
                tmp.append(len(x) * len(_class) / len(partial_con))
            result_dict[key] = tmp
        return result_dict


target_dict = {
    'hydrophobicity': ['RKEDQN', 'GASTPHY', 'CVLIMFW'],
    'normalized Van der Waals volume': ['GASCTPD', 'NVEQIL', 'MHKFRYW'],
    'polarity': ['LIFWCMVY', 'PATGS', 'HQRKNED'],
    'polarizability': ['GASDT', 'CPNVEQIL', 'KMHFRYW'],
    'charge': ['KR', 'ANCQGHILMFPSTWYV', 'DE'],
    'surface tension': ['GQDNAHR', 'KTSEC', 'ILMFPWYV'],
    'secondary structure': ['EALMQKRH', 'VIYCWFT', 'GNPSD'],
    'solvent accessibility': ['ALFCGIVW', 'RKQEND', 'MPSTHY']
}


if __name__ == '__main__':

    df = pd.DataFrame(pd.read_csv('../../data/original.csv'))

    lines = df.shape[0]
    s = df['Sequence'].tolist()

    for sequence in s[:10]:
        pf = PolypeptideFrequency(content=sequence)
        print(json.dumps(pf.find(1.0), indent=4))
