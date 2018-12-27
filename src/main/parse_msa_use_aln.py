import re
import pandas as pd
from make_image import monochrome, resized_mono
import math
import pyprind
import time
import os
import shutil

cata_list = [
    'apoptosis',
    'autophagy',
    'cancer',
    'growth',
    'proliferation',
    'repair',
    'transcription',
]

for category in cata_list:
    print('\n\n\n %s starts!!\n' % category)

    with open('./data/fasta/original/'+category+'.aln') as f:
        content = f.read()

    s = re.findall(category+'_(\d{3,4}) hello', content)
    num_list = [int(x) for x in set(s)]
    seq_num = sorted(num_list, reverse=True)[0]

    s = re.findall(category+'_0 hello\n(.*?)\n', content)
    aligned_seq = s[0]
    aligned_len = len(aligned_seq)
    df = pd.DataFrame(columns=range(aligned_len))

    print('---------------------- Constructing full sequence ----------------------')
    bar = pyprind.ProgBar(seq_num, stream=1, bar_char="▓")
    for i in range(seq_num):  # seq_num
        bar.update()
        s = re.findall(category+'_'+str(i+1)+' hello\n(.*?)\n', content)
        try:
            aligned_seq = s[0]
        except IndexError:
            continue
        new_df = pd.DataFrame([list(aligned_seq)], columns=range(len(aligned_seq)))
        df = pd.concat([df, new_df], ignore_index=True)

    def remove_site(series):
        site_list = series.tolist()
        blank = 0
        for char in site_list:
            if char == '-':
                blank += 1
        if blank > len(site_list)*0.75:
            return True
        return False


    print('---------------------- Deleting blanks ----------------------')
    bar = pyprind.ProgBar(aligned_len, stream=1, bar_char="▓")
    for column in range(aligned_len):
        bar.update()
        if remove_site(df[column]):
            df.drop([column], axis=1, inplace=True)

    df.to_csv(category+'_full_seq.csv')


    def remove_seq(seq):
        empty = 0
        for char in seq:
            if char != '-':
                empty += 1
        if empty <= 6:
            return True
        return False


    df = pd.DataFrame(pd.read_csv(category+'_full_seq.csv', index_col=0))
    cata_root = './data/fasta/aligned/'
    # category = 'apoptosis'

    for data_set in ['train', 'test']:
        if os.path.exists(cata_root + data_set + '/' + category + '/'):
            shutil.rmtree(cata_root + data_set + '/' + category + '/')
        while os.path.exists(cata_root + data_set + '/' + category + '/'):
            time.sleep(0.1)
        os.makedirs(cata_root + data_set + '/' + category + '/')
        with open('./data/fasta/aligned/'+data_set+'/apoptosis.txt', 'w') as f:
            pass

    size = int(math.ceil(math.sqrt(df.shape[1])))

    print('---------------------- Generating pics ----------------------')
    bar = pyprind.ProgBar(df.shape[0], stream=1, bar_char="▓")
    for i in range(df.shape[0]):
        bar.update()
        clean_seq = df.iloc[i, :]
        if remove_seq(clean_seq):
            continue
        clean_seq = ''.join(clean_seq.tolist())

        if i < 0.8 * df.shape[0]:
            monochrome(clean_seq, size, cata_root + 'train/' + category + '/' + str(i) + '.jpg', content_type='dash_contained')
            with open(cata_root + 'train/' + category + '.txt', 'a') as f:
                f.write('>%s_%s hello\n' % (category, i))
                f.write('%s\n\n' % clean_seq)
        else:
            monochrome(clean_seq, size, cata_root + 'test/' + category + '/' + str(i) + '.jpg', content_type='dash_contained')
            with open(cata_root + 'test/' + category + '.txt', 'a') as f:
                f.write('>%s_%s hello\n' % (category, i))
                f.write('%s\n\n' % clean_seq)
