import numpy as np
import tensorflow as tf


def extract_data(filename):
    labels = []
    fvecs = []

    with open(filename) as f:
        content = f.read().split('\n')[1:]
    for row in content:
        row = row.split(',')
        if row == ['']:
            continue
        labels.append(row[2])
        fvecs.append(row[3:])

    fvecs_np = np.matrix(fvecs).astype(np.float32)
    labels_np = np.array(labels).astype(dtype=np.uint8)
    labels_onehot = (np.arange(10) == labels_np[:, None]).astype(np.float32)

    ds_fvecs = tf.data.Dataset.from_tensor_slices(fvecs_np)
    ds_lab_oh = tf.data.Dataset.from_tensor_slices(labels_onehot)

    zipped_data = tf.data.Dataset.zip((ds_fvecs, ds_lab_oh))
    return zipped_data


class ProteinData:
    def __init__(self, tf_dataset):
        self.pos = 0
        self.features = None
        self.labels = None
        tf_dataset = tf_dataset.batch(100)
        tf_dataset = tf_dataset.repeat(1)
        features, labels = tf_dataset.make_one_shot_iterator().get_next()
        # features = tf.reshape(features, [284, -1])
        with tf.Session() as sess:
            while True:
                try:
                    feats, labs = sess.run([features, labels])
                    self.features = feats if self.features is None else np.concatenate([self.features, feats])
                    self.labels = labs if self.labels is None else np.concatenate([self.labels, labs])
                except tf.errors.OutOfRangeError:
                    break
        self.num_examples = len(self.labels)

    def next_batch(self, batch_size):
        if self.pos + batch_size > len(self.features) or self.pos + batch_size > len(self.labels):
            self.pos = 0
        res = (self.features[self.pos:self.pos + batch_size], self.labels[self.pos:self.pos + batch_size])
        self.pos += batch_size
        return res


class Protein:
    def __init__(self, data_dir):
        def data_set(file_name):
            return extract_data(data_dir+file_name)
        self.train = ProteinData(data_set('train.csv'))
        self.test = ProteinData(data_set('test.csv'))


def read_data_sets(data_dir):
    return Protein(data_dir=data_dir)


if __name__ == '__main__':
    protein = read_data_sets()
    # print(protein.train.labels)
