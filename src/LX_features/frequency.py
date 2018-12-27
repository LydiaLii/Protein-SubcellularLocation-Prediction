import re
import pandas as pd
from collections import Counter
import time
from src.LX_features.History import SaveWithHistory


class NoneHit(Warning):
    pass


class PolypeptideFrequency:

    def __init__(self, content=''):
        self.content = content
        self.seq_len = len(self.content)
        self.stepwise = (len(content) + 1) // 4
        self.aa_freq = []

    def frequency(self):
        # Split sequence into 4 parts.
        steps = []
        for i in range(5):
            steps.append(i * self.stepwise)
        c = Counter()
        DH = []
        for i in range(4):
            cut_part = self.content[steps[i]:steps[i + 1]]
            tmp_c = Counter(cut_part)
            c += tmp_c
            DH.append(tmp_c)
        DH.append(c)
        self.aa_freq = [c[x]/self.seq_len for x in amino_acids]

        # Counting AA frequency in each part, and whole sequence.
        result_dict = {}
        for key, value in target_dict.items():
            tmp = {}
            for index, _class in enumerate(value):
                pos_frequency = []
                for p in DH:
                    hits = 0
                    for aa in _class:
                        hits += p[aa] / self.stepwise
                    pos_frequency.append(hits)
                pos_frequency[4] /= 4
                tmp['class_' + str(index + 1)] = pos_frequency
            result_dict[key] = tmp
        return result_dict

    def position(self):
        result_dict = {}
        pos_mark = [0., .25, .5, .75, 1.]
        for key, value in target_dict.items():
            tmp = {}
            for index, _class in enumerate(value):
                pattern = '[' + '|'.join(list(_class)) + ']'
                hit_pos = [m.start() for m in re.finditer(pattern, self.content)]
                marked_pos = []
                hit_num = len(hit_pos) - 1
                for i in pos_mark:
                    if len(hit_pos) == 0:
                        raise NoneHit('No hit found!')
                    else:
                        marked_pos.append(hit_pos[int(i * hit_num)] / self.seq_len)
                tmp['class_' + str(index + 1)] = marked_pos
            result_dict[key] = tmp
        return result_dict

    def double_aa(self):
        result_dict = {}
        for key, value in target_dict.items():
            tmp = {}
            for class_A, class_B in [(0, 1), (0, 2), (1, 2)]:
                pattern = '[' + '|'.join(list(value[class_A])) + '][' + '|'.join(list(value[class_B])) + ']'
                hit_pos = re.findall(pattern, self.content)
                hit_length = len(hit_pos) * 2

                pattern = '[' + '|'.join(list(value[class_B])) + '][' + '|'.join(list(value[class_A])) + ']'
                hit_pos = re.findall(pattern, self.content)
                hit_length += len(hit_pos) * 2

                tmp['class_' + str(class_A + 1) + str(class_B + 1)] = [hit_length / (self.seq_len - 1)]
            result_dict[key] = tmp
        return result_dict

    @staticmethod
    def flatten_result(d):
        empty = []
        for key, value in d.items():
            for k, v in value.items():
                empty += v
        return empty

    @staticmethod
    def get_header():
        header = ['AA-FREQ_'+x for x in amino_acids]
        freq_header = []
        pos_header = []
        doub_header = []
        for key in target_dict.keys():
            for _class in range(3):
                for usage in ['.00-.25', '.25-.50', '.50-.75', '.75-1.0', 'whole']:
                    freq_header.append('FREQ_'+key[:3]+'_c'+str(_class+1)+'_'+usage)
                for usage in ['.00', '.25', '.50', '.75', '1.0']:
                    pos_header.append('POS_'+key[:3]+'_c'+str(_class+1)+'_'+usage)
                doub_header.append('DOUB_'+key[:3]+'_c'+str(_class)+str((_class+1) % 3))
        header += freq_header
        header += pos_header
        header += doub_header
        return header

    def get_feature(self):
        frequency = self.flatten_result(self.frequency())
        position = self.flatten_result(self.position())
        double_aa = self.flatten_result(self.double_aa())
        return self.aa_freq+frequency + position + double_aa


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

amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K',
               'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

if __name__ == '__main__':

    df = pd.DataFrame(pd.read_csv('../../data/original.csv'))
    labels = pd.DataFrame(pd.read_csv('../../data/label/label_20180611_164434.csv'))

    lines = df.shape[0]
    s = df['Sequence'].tolist()
    e = df['Entry'].tolist()
    label = labels['Label'].tolist()
    ii = labels['Index'].tolist()
    label = {ii[i]: label[i] for i in range(len(label))}

    sh = SaveWithHistory('../../data/LX_features/', PolypeptideFrequency().get_header())

    start = time.time()
    for i in range(lines):
        if i not in ii:
            continue
        pf = PolypeptideFrequency(content=s[i])
        try:
            sh.add_line(i, e[i], label[i], pf.get_feature())
        except NoneHit:
            print('No hit found in Entry %d [%s]' % (i, e[i]))
        if i % 1000 == 999:
            print('Entry [ %5d - %5d ] finished in [%.1f]s.' % (i - 999, i, time.time() - start))
            start = time.time()
