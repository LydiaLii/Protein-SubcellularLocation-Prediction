import pandas as pd
import os
from make_image import monochrome, resized_mono, fixed_size
import math
import matplotlib.pyplot as plt
import pyprind
import shutil
import time


def plot(length_dict):
    i = 1
    fig = plt.figure(figsize=(10, 5))
    for key, value in length_dict.items():
        ax = fig.add_subplot(2, 4, i)
        ax.hist(value, bins=200)
        plt.title(key)
        plt.xlim(0, 2000)
        plt.ylim(0, 150)
        i += 1
    plt.tight_layout()
    plt.show()


SCHEME = 'fixed_28'

sequence = pd.DataFrame(pd.read_csv('./data/sequence.csv', header=None, index_col=0))
func_sort = pd.DataFrame(pd.read_csv('./data/function_sort.csv'))


gp = func_sort.groupby('catalog')

categories = gp.count().index.tolist()

# length_dict = {}

for category in categories:
    cata_root = './data/fasta/'+SCHEME+'/'
    print('------------ Processing [%s] ------------' % category.upper())

    # length_dict[category] = []
    cat_df = gp.get_group(category)
    test_talbe = cat_df.copy()
    test_talbe.reset_index(inplace=True, drop=True)

    cat_seq = []
    for i in range(test_talbe.shape[0]):
        seq_list = eval(test_talbe['sequence'][i])
        for item in seq_list:
            cat_seq.append(item)

    cat_seq = list(set(cat_seq))

    for dataset in ['train', 'test']:
        if os.path.exists(cata_root + dataset+'/'+category+'/'):
            shutil.rmtree(cata_root + dataset+'/'+category+'/')
        while os.path.exists(cata_root + dataset+'/'+category+'/'):
            time.sleep(0.1)
        os.makedirs(cata_root + dataset+'/'+category+'/')

        with open(cata_root + dataset + '/' + category + '.txt', 'w'):
            pass

    bar = pyprind.ProgBar(len(cat_seq), stream=1, bar_char="▓")
    for index, seq in enumerate(cat_seq):
        # length_dict[category].append(len(sequence[1][seq]))
        source_seq = sequence[1][seq]
        bar.update()
        if len(source_seq) < 30:
            continue
        size = int(math.ceil(math.sqrt(len(source_seq))))

        if index < len(cat_seq)*0.80:
            fixed_size(source_seq, size, cata_root + 'train/'+category+'/'+str(index)+'.jpg')
            with open(cata_root + '/train/' + category + '.txt', 'a') as f:
                f.write('>%s_%s hello\n' % (category, index))
                f.write('%s\n\n' % source_seq)
        else:
            fixed_size(source_seq, size, cata_root + 'test/'+category+'/'+str(index)+'.jpg')
            with open(cata_root + '/test/' + category + '.txt', 'a') as f:
                f.write('>%s_%s hello\n' % (category, index))
                f.write('%s\n\n' % source_seq)

    print('\n')

# plot(length_dict)
