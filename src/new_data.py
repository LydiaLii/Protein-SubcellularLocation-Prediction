
from src.LX_features.frequency import PolypeptideFrequency, NoneHit

import numpy as np
from src.simple_ann.reload_test import get_answer
import random


class NewData:
    loc = ['Endoplasmic reticulum membrane', 'Cell projection', 'Cell membrane', 'Cell junction',
           'Cytoplasm', 'Nucleus', 'Secreted', 'Cytoskeleton', 'Mitochondrion', 'Membrane']

    def __init__(self, test_seq_list):
        self.test_seq_list = test_seq_list
        with open('../data/LX_features/20180611_223038_[284 columns]_scale.csv') as f:
            scales = f.read().split('\n')[1].split(',')[4:]
        self.scales = np.array([float(x) for x in scales])

        # Dev_Sec below
        self.real_labels = []
        self.answers = None

    def predict(self):
        test_seqs = self.test_seq_list

        real_fea_list = []
        fail_indexes = []
        for ind, seq in enumerate(test_seqs):
            pf = PolypeptideFrequency(content=seq)
            try:
                features = np.array(pf.get_feature())
            except NoneHit:
                # print('No expected hit found in entry.')
                fail_indexes.append(ind)
                continue
            real_fea_list.append((features * self.scales).tolist())
        real_fea_list = np.array(real_fea_list)
        answer_list, confidences = get_answer(real_fea_list, './simple_ann/')

        confidences = [max(x) / np.mean(x) / 15 for x in confidences]

        answer_list = [NewData.loc[x] for x in answer_list]
        for i in fail_indexes:
            answer_list.insert(i, 'UNEXPECTED')
            confidences.insert(i, 0.)
        return zip(answer_list, confidences)

    def dev_test(self):
        with open('../../data/label/label_20180611_164434.csv') as f:
            labels = f.read().split('\n')[1:-1]
        chosen_labels = random.sample(labels, 1000)
        chosen_labels = [x.split(',') for x in chosen_labels]
        chosen_labels = {x[4] + '-' + x[1]: x[0] for x in chosen_labels}

        with open('../../data/sequence.csv') as f:
            sequences = f.read().split('\n')[:-1]
        sequences = [x.split(',') for x in sequences]
        sequences = {x[0]: x[1] for x in sequences}

        test_seqs = []
        for label, index in chosen_labels.items():
            self.real_labels.append(label.split('-')[0])
            test_seqs.append(sequences[index])

        self.test_seq_list = test_seqs
        self.answers = self.predict()
        self.dev_eval()

    def dev_eval(self):
        correct = {x: 0 for x in range(10)}
        tested = {x: 0 for x in range(10)}
        for i, (index, conf) in enumerate(self.answers):
            if index == 'UNEXPECTED':
                break
            predict_type = NewData.loc.index(index)
            # print(predict_type, real_labels[i], end=' | ')
            # if i % 8 == 7:
            #     print()
            tested[predict_type] += 1
            if predict_type == int(self.real_labels[i]):
                correct[predict_type] += 1
                print('[ √ ] %.1f%%' % (conf * 100))
            else:
                print('[   ] %.1f%%' % (conf * 100))

        # for key, value in correct.items():
        #     try:
        #         accuracy = value / tested[key]
        #     except ZeroDivisionError:
        #         print('No [ %s ] selected!' % NewData.loc[key])
        #         continue
        #     print('Correction [ %s ] = %.1f%%' % (NewData.loc[key], accuracy * 100))
        # print(correct, tested)
        pass


if __name__ == '__main__':
    nd = NewData([
        'MFRLWLLLAGLCGLLASRPGFQNSLLQIVIPEKIQTNTNDSSEIEYEQISYIIPIDEKLYTVHLKQRYFLADNFMI'
        'YLYNQGSMNTYSSDIQTQCYYQGNIEGYPDSMVTLSTCSGLRGILQFENVSYGIEPLESAVEFQHVLYKLKNEDND'
        'IAIFIDRSLKEQPMDDNIFISEKSEPAVPDLFPLYLEMHIVVDKTLYDYWGSDSMIVTNKVIEIVGLANSMFTQFK'
        'VTIVLSSLELWSDENKISTVGEADELLQKFLEWKQSYLNLRPHDIAYLLIYMDYPRYLGAVFPGTMCITRYSAGVA'
        'LYPKEITLEAFAVIVTQMLALSLGISYDDPKKCQCSESTCIMNPEVVQSNGVKTFSSCSLRSFQNFISNVGVKCLQ'
        'NKPQMQKKSPKPVCGNGRLEGNEICDCGTEAQCGPASCCDFRTCVLKDGAKCYKGLCCKDCQILQSGVECRPKAHP'
        'ECDIAENCNGTSPECGPDITLINGLSCKNNKFICYDGDCHDLDARCESVFGKGSRNAPFACYEEIQSQSDRFGNCG'
        'RDRNNKYVFCGWRNLICGRLVCTYPTRKPFHQENGDVIYAFVRDSVCITVDYKLPRTVPDPLAVKNGSQCDIGRVC'
        'VNRECVESRIIKASAHVCSQQCSGHGVCDSRNKCHCSPGYKPPNCQIRSKGFSIFPEEDMGSIMERASGKTENTWL'
        'LGFLIALPILIVTTAIVLARKQLKKWFAKEEEFPSSESKSEGSTQTYASQSSSEGSTQTYASQTRSESSSQADTSK'
        'SKSEDSAEAYTSRSKSQDSTQTQSSSN',
        'MDDREDLVYQAKLAEQAERYDEMVESMKKVAGMDVELTVEERNLLSVAYKNVIGARRASWRIISSIEQKEENKGGE'
        'DKLKMIREYRQMVETELKLICCDILDVLDKHLIPAANTGESKVFYYKMKGDYHRYLAEFATGNDRKEAAENSLVAY'
        'KAASDIAMTELPPTHPIRLGLALNFSVFYYEILNSPDRACRLAKAAFDDAIAELDTLSEESYKDSTLIMQLLRDNL'
        'TLWTSDMQGDGEEQNKEALQDVEDENQ',
        'MERASLIQKAKLAEQAERYEDMAAFMKGAVEKGEELSCEERNLLSVAYKNVVGGQRAAWRVLSSIEQKSNEEGSEE'
        'KGPEVREYREKVETELQGVCDTVLGLLDSHLIKEAGDAESRVFYLKMKGDYYRYLAEVATGDDKKRIIDSARSAYQ'
        'EAMDISKKEMPPTNPIRLGLALNFSVFHYEIANSPEEAISLAKTTFDEAMADLHTLSEDSYKDSTLIMQLLRDNLT'
        'LWTADNAGEEGGEAPQEPQS'
    ])
    # nd.dev_test()
    i = 1
    for predict, conf in nd.predict():
        if predict == 'UNEXPECTED':
            print('Invalid input!')
            continue
        print('Sequence %d supposed to be [%s] at the confidence of %.1f%%' % (i, predict, conf * 100))
        i += 1
