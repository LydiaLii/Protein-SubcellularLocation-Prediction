import numpy as np
import tensorflow as tf
import os, shutil
import math
# from src.simply_ann.mnistdata import read_data_sets
from src.simple_ann.dataset import read_data_sets

feature_length = 284
learning_rate = 0.003
ini_keep_rate = 0.9
train_iteration = 100
batch_size = 100
display_step = 10
time_stamp = '20180612_211120'  # 20180612_211120, 20180612_221825
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

K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)

W4 = tf.Variable(tf.truncated_normal([71 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)


# The model
stride = 1  # output is 284
Y1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, stride], padding='SAME') + B1)
stride = 2  # output is 142
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride], padding='SAME') + B2)
stride = 2  # output is 71
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 71 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5


with tf.name_scope("Wx_b") as scope:
    model = tf.nn.softmax(Ylogits)

with tf.name_scope("cost_function"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y)
    cost_function = tf.reduce_mean(cross_entropy) * 100
    # cost_function = -tf.reduce_sum(y * tf.log(tf.clip_by_value(model, 1e-20, 1.0)))
    tf.summary.scalar("cost_function", cost_function)

with tf.name_scope("train"):
    lr = 0.0001 + tf.train.exponential_decay(learning_rate, step, 100, 1 / math.e)
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost_function)

init = tf.initialize_all_variables()
merge_summary_op = tf.summary.merge_all()

predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(predictions, "float"))

first_flag = True
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
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = np.reshape(batch_xs, (-1, feature_length))
            o, c, accu, summary_str = sess.run([optimizer, cost_function, accuracy, merge_summary_op],
                                               feed_dict={x: batch_xs, y: batch_ys, pkeep: keep_rate, step: i})
            avg_cost += c / total_batch
            summary_writer.add_summary(summary_str, iteration * total_batch + i)
        if iteration % display_step == 0:
            print('Iteration: %04d, loss = %.3f, accuracy = %.1f%%' % (iteration + 1, avg_cost, accu*100))
    print('Complete successfully!')

    test_images = np.reshape(mnist.test.features, (-1, feature_length))
    test_labels = mnist.test.labels
    accu = accuracy.eval({x: test_images, y: test_labels, pkeep: ini_keep_rate})
    print('Accuracy: %.2f%%' % (accu * 100))
