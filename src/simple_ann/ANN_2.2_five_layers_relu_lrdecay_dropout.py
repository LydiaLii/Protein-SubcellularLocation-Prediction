import numpy as np
import tensorflow as tf
import os, shutil
import math
# from src.simply_ann.mnistdata import read_data_sets
from src.simple_ann.dataset import read_data_sets


feature_length = 284
learning_rate = 0.003
ini_keep_rate = 0.90
train_iteration = 120
batch_size = 100
display_step = 10

time_stamp = '20180612_211120'  # 20180612_211120, 20180612_221825, 20180613_164908, 20180613_164924
data_dir = '../../data/ANN_data/'+time_stamp+'/'
log_dir = data_dir + 'logs'
# if os.path.exists(log_dir):
#     shutil.rmtree(log_dir)

# mnist = read_data_sets("data", one_hot=True, reshape=False)
mnist = read_data_sets(data_dir)

x = tf.placeholder('float', shape=[None, feature_length])
y = tf.placeholder('float', shape=[None, 10])
step = tf.placeholder(tf.int32)
pkeep = tf.placeholder(tf.float32)

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 160
M = 90
N = 50
O = 20

W1 = tf.Variable(tf.truncated_normal([feature_length, L], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.ones([L])/10)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M])/10)
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.ones([N])/10)
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B4 = tf.Variable(tf.ones([O])/10)
W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))


# The model
XX = tf.reshape(x, [-1, feature_length])
Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)
Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)
Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)
Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(Y4d, W5) + B5

with tf.name_scope("Wx_b") as scope:
    model = tf.nn.softmax(Ylogits)

with tf.name_scope("cost_function"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=y)
    cost_function = tf.reduce_mean(cross_entropy) * 100
    # cost_function = -tf.reduce_sum(y * tf.log(tf.clip_by_value(model, 1e-20, 1.0)))
    tf.summary.scalar("cost_function", cost_function)

with tf.name_scope("train"):
    lr = 0.0001 + tf.train.exponential_decay(learning_rate, step, 100, 1 / math.e)
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost_function)

init = tf.global_variables_initializer()
merge_summary_op = tf.summary.merge_all()

predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(predictions, "float"))

first_flag = True
saver = tf.train.Saver()
max_acc = 0.

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    for iteration in range(train_iteration):
        if first_flag:
            keep_rate = 1.0
            first_flag = False
        else:
            keep_rate = ini_keep_rate
        avg_cost = 0.
        accu = 0.
        test_accu = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = np.reshape(batch_xs, (-1, feature_length))
            o, c, accu, summary_str = sess.run([optimizer, cost_function, accuracy, merge_summary_op],
                                               feed_dict={x: batch_xs, y: batch_ys, pkeep: keep_rate, step: i})
            avg_cost += c / total_batch
            summary_writer.add_summary(summary_str, iteration * total_batch + i)

        test_images = np.reshape(mnist.test.features, (-1, feature_length))
        test_labels = mnist.test.labels
        test_accu = sess.run(accuracy, feed_dict={x: test_images, y: test_labels, pkeep: keep_rate})

        if test_accu > max_acc:
            print('Step [%d] overwrites, accuracy = %.1f%%' % (iteration + 1, test_accu*100))
            max_acc = test_accu
            saver.save(sess, 'save/' + time_stamp + '.ckpt', global_step=iteration + 1)

        if iteration % display_step == 0:
            print('Iteration: %04d, loss = %.3f, train_accuracy = %.1f%%, test_accuracy = %.1f%%' %
                  (iteration + 1, avg_cost, accu*100, test_accu*100))
    print('Complete successfully!')

    test_images = np.reshape(mnist.test.features, (-1, feature_length))
    test_labels = mnist.test.labels
    accu = accuracy.eval({x: test_images, y: test_labels, pkeep: ini_keep_rate})
    print('Accuracy: %.2f%%' % (accu * 100))
