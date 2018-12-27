import requests
import json
import time
from src.features.Logger import create_logger


class UniprotFeatures:

    def __init__(self, entry, logger='full'):
        self.COI = [
            # 'DOMAINS_AND_SITES',
            'PTM',
            # 'STRUCTURAL'
        ]
        self.entry = entry
        self.seq_feature = None
        self.features = []
        self.sequence = None
        self.entry_name = None

        logger = create_logger(mode=logger)
        logger.name = 'Uniprot Features'
        logger.header()

        logger.sec_start('Retrieving data')
        self.request()
        logger.sec_end()

        logger.sec_start('Parsing info')
        self.info_parse()
        logger.sec_end()

        logger.footer()

    def entry_check(self, name, seq):
        if (name == self.entry_name) and (seq == self.sequence):
            return True
        else:
            return False

    def request(self):
        r = requests.get('https://www.ebi.ac.uk/proteins/api/features/'+self.entry)
        feature_dict = eval(r.text)
        self.sequence = feature_dict['sequence']
        self.entry_name = feature_dict['entryName']
        for feature in feature_dict['features']:
            try:
                del feature['evidences']
            except KeyError:
                pass
            if feature['category'] in self.COI:
                self.features.append(feature)

        # print(json.dumps({"": self.features}, indent=4))

    def info_parse(self):
        modify = [0] * len(self.sequence)
        for item in self.features:
            if item['category'] == 'PTM':
                for i in range(int(item['begin'])-1, int(item['end'])):
                    modify[i] += 1
        # self.seq_feature = '['+str(max(modify))+']'+''.join([str(x) for x in modify])
        self.seq_feature = modify


if __name__ == '__main__':
    uf = UniprotFeatures('P62258')
    IDENTICAL_FLAG = uf.entry_check(
        name='1433E_HUMAN',
        seq='MDDREDLVYQAKLAEQAERYDEMVESMKKVAGMDVELTVEERNLLSVAYKNVIGARRASWRIISSIEQKEENKGGEDKLKMIREYRQMVETELKLICCDILDVLD'
            'KHLIPAANTGESKVFYYKMKGDYHRYLAEFATGNDRKEAAENSLVAYKAASDIAMTELPPTHPIRLGLALNFSVFYYEILNSPDRACRLAKAAFDDAIAELDTLS'
            'EESYKDSTLIMQLLRDNLTLWTSDMQGDGEEQNKEALQDVEDENQ'
    )
    if IDENTICAL_FLAG:
        print('Sequence consistency verified!')
        print(uf.seq_feature)
    else:
        raise Warning('Sequence DIFFERENT from downloaded!')
