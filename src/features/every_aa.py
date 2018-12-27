import pandas as pd
import time
from src.features.Logger import create_logger


class SiteFeature:
    def __init__(self, seq, logger='full'):
        self.sequence = seq
        self.hydro_list = None

        logger = create_logger(mode=logger)
        logger.name = 'Site Feature'
        logger.header()

        logger.sec_start('Reading files')
        aa_hydro = pd.DataFrame(pd.read_csv('../../res/Amino_Acid_Hydrophobicity.csv'))
        abbr = pd.DataFrame(pd.read_csv('../../res/AA_abbr.csv'))
        aa_pro_1 = pd.DataFrame(pd.read_csv('../../res/AA_property.csv'))
        aa_pro_2 = pd.DataFrame(pd.read_csv('../../res/AA_property_2.csv'))
        logger.sec_end()

        logger.sec_start('Merging csv')
        df = pd.merge(abbr, aa_hydro, how='outer', left_on='Residue', right_on='Residue Type')
        df = pd.merge(df, aa_pro_1, how='outer', on='Code')
        df = pd.merge(df, aa_pro_2, how='outer', on='Code')
        df.drop(['Residue Type', 'Residue', '_Charge', '_Side Chain Polarity', '_Side Chain Acidity'],
                axis=1, inplace=True)
        logger.sec_end()

        # columns = df.columns.values.tolist()
        # for item in columns:
        #     print(item)
        # exit(0)

        self.col_num = df.shape[1]-1

        logger.sec_start('DataFrame to dict')
        self.property = {}
        for i in range(df.shape[0]):
            info = df.iloc[i, :].tolist()
            key = info.pop(0)
            self.property[key] = info
        logger.sec_end()

        logger.footer()

    def get_property_list(self, use_col=0):
        if use_col not in range(self.col_num):
            raise Warning('Column index out of range!')
        self.hydro_list = []
        for bit in self.sequence:
            self.hydro_list.append(self.property[bit][use_col])
        return self.hydro_list


if __name__ == '__main__':
    sequence = 'MDDREDLVYQAKLAEQAERYDEMVESMKKVAGMDVELTVEERNLLSVAYKNVIGARRASWRIISSIEQKEENKGGEDKLKMIREYRQMVETELKLICCDI' \
               'LDVLDKHLIPAANTGESKVFYYKMKGDYHRYLAEFATGNDRKEAAENSLVAYKAASDIAMTELPPTHPIRLGLALNFSVFYYEILNSPDRACRLAKAAFD' \
               'DAIAELDTLSEESYKDSTLIMQLLRDNLTLWTSDMQGDGEEQNKEALQDVEDENQ'
    sf = SiteFeature(sequence)
    print(sf.get_property_list(use_col=10))
