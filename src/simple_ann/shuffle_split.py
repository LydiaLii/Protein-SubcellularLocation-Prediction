import random
import os, shutil
import hashlib
import datetime


def retrieve_data(data_dir, data_file):
    def get_file_md5(filename):
        if not os.path.isfile(filename):
            return
        myhash = hashlib.md5()
        with open(filename, 'rb') as f:
            while True:
                b = f.read(8096)
                if not b:
                    break
                myhash.update(b)
        return myhash.hexdigest()

    def create_files(otype, fn):
        if os.path.exists(data_dir + otype + '.csv'):
            base_md5 = get_file_md5(data_dir + otype + '.csv')
            dest_md5 = get_file_md5(fn)
            if base_md5 == dest_md5:
                print('File already exist! [ HASH: %s ]' % base_md5)
                return 0
        shutil.copy(fn, data_dir + otype + '.csv')

    if not os.path.exists(data_dir):
        print('Creating %s...' % data_dir)
        os.makedirs(data_dir)
    create_files('data', data_file)
    with open(data_dir + 'log.info', 'w') as f:
        f.write('[Data] %s\n' % data_file)


def shuffle_split(data_dir, test_ratio=0.25):
    with open(data_dir+'data.csv') as f:
        header = f.readline()
        content = f.read()
    records = content.split('\n')
    random.shuffle(records)
    record_index = range(len(records))
    test_list = random.sample(record_index, int(len(records)*test_ratio))
    train_list = [x for x in record_index if x not in test_list]

    with open(data_dir+'train.csv', 'w') as f:
        f.write(header)
        f.write('\n'.join([records[x] for x in train_list]))
    with open(data_dir+'test.csv', 'w') as f:
        f.write(header)
        f.write('\n'.join([records[x] for x in test_list]))

    with open(data_dir + 'log.info', 'a') as f:
        f.write('[Test set] %s\n' % str(test_list))

    print('Shuffle complete. Test data: [%d, %.1f%%]' % (len(test_list), len(test_list)/len(records)*100))


class ShuffleSplit:
    def __init__(self, target_dir, source_feature, test_ratio=0.25):
        self.time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        retrieve_data(
            data_dir=target_dir + self.time_stamp + '/',
            data_file=source_feature
        )
        shuffle_split(data_dir=target_dir+self.time_stamp+'/', test_ratio=test_ratio)


if __name__ == '__main__':
    ss = ShuffleSplit(
        target_dir='../../data/ANN_data/',
        source_feature='../../data/LX_features/20180611_223038_[284 columns]_adjusted.csv',
        test_ratio=0.25
    )
