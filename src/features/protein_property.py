import requests
import re
import json
import time
from src.features.Logger import create_logger


class ProteinProperty:
    def __init__(self, seq, mode='request', logger='full'):

        self.mode = mode
        self.sequence = seq
        self.aa_composition = None
        self.atom_composition = None
        self.total_atoms = None
        self.protein_len = len(seq)
        self.molecular_weight = None
        self.theoretical_pi = None
        self.instability_index = None
        self.extinction_coefficients = None
        self.aliphatic_index = None
        self.GRAVY = None
        self.residue_charge = {}
        self.content = None
        self.info_dict = {}
        self.value_list = []
        self.key_list = []

        logger = create_logger(mode=logger)
        logger.name = 'Protein Property'
        logger.header()

        logger.sec_start('Retrieving data')
        self.request()
        logger.sec_end()

        logger.sec_start('Extracting info')
        self.extract_info()
        logger.sec_end()

        logger.sec_start('Integrating gathered info')
        self.info_integrate()
        logger.sec_end()

        logger.footer()

    def request(self):
        if self.mode == 'request':
            data = {"sequence": self.sequence}
            r = requests.post('https://web.expasy.org/cgi-bin/protparam/protparam', data=data)
            self.content = r.text
        elif self.mode == 'local':
            with open('./source.txt') as f:
                self.content = f.read()
        elif self.mode == 'blank':
            pass
        else:
            raise SyntaxError('Unrecognizable mode!')

    def extract_info(self):
        def real():
            self.molecular_weight = float(re.findall('<B>Molecular weight:</B> (.*?)\n', self.content)[0])
            self.theoretical_pi = float(re.findall('<B>Theoretical pI:</B> (.*?)\n', self.content)[0])
            pattern = "name='total_(.)' value='(\d*?)'><input type='hidden' name='percent_.' value='(.*?)'>"
            percentage = re.findall(pattern, self.content)
            # for item in percentage:
            #     print("'"+item[0]+"'", end=',')
            self.aa_composition = {
                x[0]: {
                    'num': int(x[1]),
                    'per': float(x[2])
                }
                for x in percentage
            }
            self.total_atoms = int(re.findall('<B>Total number of atoms:</B> (\d*?)\n', self.content)[0])
            pattern = "\w*?\s*?([CHNOS])\s*?(\d*?)\n"
            percentage = re.findall(pattern, self.content)
            self.atom_composition = {
                x[0]: {
                    'num': int(x[1]),
                    'per': int(x[1]) / self.total_atoms
                }
                for x in percentage
            }
            self.residue_charge['negative'] = {'num': int(re.findall('\(Asp \+ Glu\):</B> (\d*?)\n', self.content)[0])}
            self.residue_charge['negative']['per'] = self.residue_charge['negative']['num'] / self.protein_len
            self.residue_charge['positive'] = {'num': int(re.findall('\(Arg \+ Lys\):</B> (\d*?)\n', self.content)[0])}
            self.residue_charge['positive']['per'] = self.residue_charge['negative']['num'] / self.protein_len

            coeff = re.findall('Ext\. coefficient\s*?(\d*?)\nAbs 0\.1% \(=1 g/l\)\s*?(\S*?), assuming all',
                               self.content)
            self.extinction_coefficients = {
                'kept': {'ext_cof': int(coeff[0][0]), 'abs': float(coeff[0][1])},
                'reduced': {'ext_cof': int(coeff[1][0]), 'abs': float(coeff[1][1])}
            }

            self.instability_index = float(re.findall('index \(II\) is computed to be (.*?)\n', self.content)[0])
            self.aliphatic_index = float(re.findall('<B>Aliphatic index:</B> (.*?)\n', self.content)[0])
            self.GRAVY = float(re.findall('hydropathicity \(GRAVY\):</B> (.*?)\n', self.content)[0])

        def blank():
            self.molecular_weight = 0
            self.theoretical_pi = 0
            self.aa_composition = {x: {'num': 0, 'per': 0} for x in
                                   ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F',
                                    'P', 'S', 'T', 'W', 'Y', 'V', 'O', 'U', 'B', 'Z', 'X']}
            self.total_atoms = 0
            self.atom_composition = {x: {'num': 0, 'per': 0} for x in ['C', 'H', 'N', 'O', 'S']}
            self.residue_charge['negative'] = {'num': 0}
            self.residue_charge['negative']['per'] = 0
            self.residue_charge['positive'] = {'num': 0}
            self.residue_charge['positive']['per'] = 0
            self.extinction_coefficients = {
                'kept': {'ext_cof': 0, 'abs': 0},
                'reduced': {'ext_cof': 0, 'abs': 0}
            }
            self.instability_index = 0
            self.aliphatic_index = 0
            self.GRAVY = 0

        if self.mode == 'blank':
            blank()
        else:
            real()

    def info_integrate(self):
        self.info_dict = {
            "protein_len": self.protein_len,
            "total_atoms": self.total_atoms,
            "molecular_weight": self.molecular_weight,

            "aa_composition": self.aa_composition,
            "atom_composition": self.atom_composition,

            "theoretical_pi": self.theoretical_pi,
            "instability_index": self.instability_index,
            "extinction_coefficients": self.extinction_coefficients,
            "aliphatic_index": self.aliphatic_index,
            "GRAVY": self.GRAVY,
            "residue_charge": self.residue_charge
        }
        self.value_list = []
        self.key_list = []
        self.traverse_dict(self.info_dict, [])

    def traverse_dict(self, target_dict, father_key):
        if isinstance(target_dict, dict):
            for item in target_dict.items():
                father_key.append(item[0])
                self.traverse_dict(item[1], father_key)
                father_key.pop()
        else:
            self.value_list.append(target_dict)
            self.key_list.append('_'.join(father_key))


if __name__ == '__main__':
    sequence = 'MNESKPGDSQNLACVFCRKHDDCPNKYGEKKTKEKWNLTVHYYCLLMSSGIWQRGKEEEGVYGFLIEDIRKEVNRASKLKCCVCKKNGASIGCVAPRCKRSYH' \
               'FPCGLQRECIFQFTGNFASFCWDHRPVQIITSNNYRESLPCTICLEFIEPIPSYNILRSPCCKNAWFHRDCLQVQAINAGVFFFRCTICNNSDIFQKEMLRMG' \
               'GITDCLLEESSPKLPRQSPGSQSKDLLRQGSKFRRNVSTLLIELGFQIKKKTKRLYINKANIWNSALDAFRNRNFNPSYAIEVAYVIENDNFGSEHPGSKQEF' \
               'LSLLMQHLENSSLFEGSLSKNLSLNSQALKENLYYEAGKMLAISLVHGGPSPGFFSKTLFNCLVYGPENTQPILDDVSDFDVAQIIIRINTATTVADLKSIIN' \
               'ECYNYLELIGCLRLITTLSDKYMLVKDILGYHVIQRVHTPFESFKQGLKTLGVLEKIQAYPEAFCSILCHKPESLSAKILSELFTVHTLPDVKALGFWNSYLQ' \
               'AVEDGKSTTTMEDILIFATGCSSIPPAGFKPTPSIECLHVDFPVGNKCNNCLAIPITNTYKEFQENMDFTIRNTLRLEKEESSHYIGH'

    pp = ProteinProperty(sequence, mode='blank')
    print(pp.key_list)

    # pp = ProteinProperty(sequence)
    # print(json.dumps(pp.info_dict, indent=4))
    # print(pp.value_list)
    # print(pp.key_list)
