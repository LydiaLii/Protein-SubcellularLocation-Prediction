import numpy as np
import tensorflow as tf
import os, shutil
import math
# from src.simply_ann.mnistdata import read_data_sets
from src.simple_ann.dataset import read_data_sets

feature_length = 284
learning_rate = 0.07
train_iteration = 100
batch_size = 100
display_step = 10
time_stamp = '20180613_160255'  # 20180612_211120, 20180613_160255
data_dir = '../../data/ANN_data/'+time_stamp+'/'
log_dir = data_dir + 'logs'
# if os.path.exists(log_dir):
#     shutil.rmtree(log_dir)

# mnist = read_data_sets("data", one_hot=True, reshape=False)
mnist = read_data_sets(data_dir)

x = tf.placeholder('float', shape=[None, feature_length])
y = tf.placeholder('float', shape=[None, 10])

# train/test selector for batch normalisation
tst = tf.placeholder(tf.bool)
# training iteration
iter = tf.placeholder(tf.int32)

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 160
M = 90
N = 50
P = 20

W1 = tf.Variable(tf.truncated_normal([feature_length, L], stddev=0.1))  # 784 = 28 * 28
S1 = tf.Variable(tf.ones([L]))
O1 = tf.Variable(tf.zeros([L]))
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
S2 = tf.Variable(tf.ones([M]))
O2 = tf.Variable(tf.zeros([M]))
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
S3 = tf.Variable(tf.ones([N]))
O3 = tf.Variable(tf.zeros([N]))
W4 = tf.Variable(tf.truncated_normal([N, P], stddev=0.1))
S4 = tf.Variable(tf.ones([P]))
O4 = tf.Variable(tf.zeros([P]))
W5 = tf.Variable(tf.truncated_normal([P, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))


def batchnorm(Ylogits, Offset, Scale, is_test, iteration):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.998, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, Offset, Scale, bnepsilon)
    return Ybn, update_moving_averages


def no_batchnorm(Ylogits, Offset, Scale, is_test, iteration):
    return Ylogits, tf.no_op()


# The model
XX = tf.reshape(x, [-1, feature_length])

Y1l = tf.matmul(XX, W1)
Y1bn, update_ema1 = batchnorm(Y1l, O1, S1, tst, iter)
Y1 = tf.nn.sigmoid(Y1bn)

Y2l = tf.matmul(Y1, W2)
Y2bn, update_ema2 = batchnorm(Y2l, O2, S2, tst, iter)
Y2 = tf.nn.sigmoid(Y2bn)

Y3l = tf.matmul(Y2, W3)
Y3bn, update_ema3 = batchnorm(Y3l, O3, S3, tst, iter)
Y3 = tf.nn.sigmoid(Y3bn)

Y4l = tf.matmul(Y3, W4)
Y4bn, update_ema4 = batchnorm(Y4l, O4, S4, tst, iter)
Y4 = tf.nn.sigmoid(Y4bn)

Ylogits = tf.matmul(Y4, W5) + B5


update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)


with tf.name_scope("Wx_b") as scope:
    model = tf.nn.softmax(Ylogits)

with tf.name_scope("cost_function"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=y)
    cost_function = tf.reduce_mean(cross_entropy) * 100
    # cost_function = -tf.reduce_sum(y * tf.log(tf.clip_by_value(model, 1e-20, 1.0)))
    tf.summary.scalar("cost_function", cost_function)

with tf.name_scope("train"):
    lr = 0.0001 + tf.train.exponential_decay(learning_rate, iter, 1000, 1 / math.e)
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost_function)

init = tf.global_variables_initializer()
merge_summary_op = tf.summary.merge_all()

predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(predictions, "float"))

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    for iteration in range(train_iteration):
        avg_cost = 0.
        accu = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = np.reshape(batch_xs, (-1, feature_length))
            o, c, accu, summary_str = sess.run([optimizer, cost_function, accuracy, merge_summary_op],
                                               feed_dict={x: batch_xs, y: batch_ys, iter: i, tst: False})
            avg_cost += c / total_batch
            summary_writer.add_summary(summary_str, iteration * total_batch + i)
        if iteration % display_step == 0:
            print('Iteration: %04d, loss = %.3f, accuracy = %.1f%%' % (iteration + 1, avg_cost, accu*100))
    print('Complete successfully!')

    test_images = np.reshape(mnist.test.features, (-1, feature_length))
    test_labels = mnist.test.labels
    accu = accuracy.eval({x: test_images, y: test_labels, tst: True})
    print('Accuracy: %.2f%%' % (accu * 100))
