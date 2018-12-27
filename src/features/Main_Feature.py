from src.features.every_aa import SiteFeature
from src.features.protein_property import ProteinProperty
from src.features.feature_uniprot import UniprotFeatures
from src.features.feature_config import FeatureConfig
from src.features.History import SaveWithHistory
import json

fc = FeatureConfig()


def length_correction(seq):
    seq_len = len(seq)

    if seq_len == fc.STANDARD_SEQ_LEN:
        pass
    elif seq_len < fc.STANDARD_SEQ_LEN:
        orig_seq = seq
        index = 0
        while len(seq) < fc.STANDARD_SEQ_LEN:
            seq += [orig_seq[index % len(orig_seq)]]
            index += 1
    else:
        if fc.CUT_METHOD == 'cut_both':
            redun_half = (seq_len - fc.STANDARD_SEQ_LEN) // 2
            cut_start = redun_half
            cut_end = cut_start + fc.STANDARD_SEQ_LEN
            seq = seq[cut_start:cut_end]
        elif fc.CUT_METHOD == 'cut_tail':
            cut_start = 0
            cut_end = cut_start + fc.STANDARD_SEQ_LEN
            seq = seq[cut_start:cut_end]
        elif fc.CUT_METHOD == 'cut_head':
            cut_start = seq_len - fc.STANDARD_SEQ_LEN
            seq = seq[cut_start:]

    return seq


class MainFeatures:
    def __init__(self, entry='', name='', seq=''):
        self.entry = entry
        self.name = name
        self.sequence = seq
        self.col_list = []
        for index, item in enumerate(fc.PROPERTIES):
            if item['check']:
                self.col_list.append(index)
        self.global_property = None
        self.global_key = None
        self.verified_property = None
        self.seq_length_property = []
        self.seq_length_key = []
        self.feature_list = []
        self.key_list = []

    def generate_keys(self):
        if fc.USING_FEATURES['global_property']:
            pp = ProteinProperty(self.sequence, logger='silent', mode='blank')
            self.global_key = pp.key_list
            self.key_list += self.global_key
        if fc.USING_FEATURES['seq_length_property']:
            for col in self.col_list:
                self.seq_length_key += [fc.PROPERTIES[col]['name']+'_'+str(x) for x in range(fc.STANDARD_SEQ_LEN)]
            self.key_list += self.seq_length_key
        if fc.USING_FEATURES['verified_property']:
            self.key_list += ['Site_Modify_' + str(x) for x in range(fc.STANDARD_SEQ_LEN)]
        return self.key_list

    def generate_features(self):
        # global_property from https://web.expasy.org/
        if fc.USING_FEATURES['global_property']:
            pp = ProteinProperty(self.sequence, logger=fc.LOG_MODE)
            self.global_property = pp.value_list
            self.feature_list += self.global_property

        # seq_length_property with AA chemical properties
        if fc.USING_FEATURES['seq_length_property']:
            sf = SiteFeature(self.sequence, logger=fc.LOG_MODE)
            for col in self.col_list:
                self.seq_length_property.append(sf.get_property_list(use_col=col))
                self.seq_length_key += [fc.PROPERTIES[col]['name']+'_'+str(x) for x in range(fc.STANDARD_SEQ_LEN)]

            for index, item in enumerate(self.seq_length_property):
                self.feature_list += length_correction(item)

        # verified_property queried from https://www.ebi.ac.uk/
        if fc.USING_FEATURES['verified_property']:
            uf = UniprotFeatures(self.entry, logger=fc.LOG_MODE)
            IDENTICAL_FLAG = uf.entry_check(name=self.name, seq=self.sequence)
            if IDENTICAL_FLAG:
                self.verified_property = length_correction(uf.seq_feature)
            else:
                raise Warning('Sequence DIFFERENT from downloaded!')
            self.feature_list += self.verified_property

        return self.feature_list


if __name__ == '__main__':
    i = 0
    e = 'P62258'
    n = '1433E_HUMAN'
    s = 'MDDREDLVYQAKLAEQAERYDEMVESMKKVAGMDVELTVEERNLLSVAYKNVIGARRASWRIISSIEQKEENKGGEDKLKMIREYRQMVETELKLICCDILDVLD' \
        'KHLIPAANTGESKVFYYKMKGDYHRYLAEFATGNDRKEAAENSLVAYKAASDIAMTELPPTHPIRLGLALNFSVFYYEILNSPDRACRLAKAAFDDAIAELDTLS' \
        'EESYKDSTLIMQLLRDNLTLWTSDMQGDGEEQNKEALQDVEDENQ'

    mf = MainFeatures()
    sh = SaveWithHistory('../../data/features/', mf.generate_keys())
    # print(mf.generate_keys())

    mf = MainFeatures(e, n, s)
    sh.add_line(e, 0, mf.generate_features())
    mf = MainFeatures(e, n, s)
    sh.add_line(e, 1, mf.generate_features())
