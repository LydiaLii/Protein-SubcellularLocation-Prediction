import numpy as np
import tensorflow as tf
import os, shutil
# from src.simply_ann.mnistdata import read_data_sets
from src.simple_ann.dataset import read_data_sets

feature_length = 284
learning_rate = 0.01
train_iteration = 200
batch_size = 100
display_step = 10
time_stamp = '20180612_211120'  # 20180612_211120
data_dir = '../../data/ANN_data/'+time_stamp+'/'
log_dir = data_dir + 'logs'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

# mnist = read_data_sets("data", one_hot=True, reshape=False)
mnist = read_data_sets(data_dir)

x = tf.placeholder('float', shape=[None, feature_length])
y = tf.placeholder('float', shape=[None, 10])

W = tf.Variable(tf.zeros([feature_length, 10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
    model = tf.nn.softmax(tf.matmul(x, W) + b)

w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)

with tf.name_scope("cost_function"):
    cost_function = -tf.reduce_sum(y * tf.log(tf.clip_by_value(model, 1e-20, 1.0)))
    tf.summary.scalar("cost_function", cost_function)

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

init = tf.initialize_all_variables()
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
                                               feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += c / total_batch
            summary_writer.add_summary(summary_str, iteration * total_batch + i)
        if iteration % display_step == 0:
            print('Iteration: %04d, loss = %.3f, accuracy = %.1f%%' % (iteration + 1, avg_cost, accu*100))
    print('Complete successfully!')

    test_images = np.reshape(mnist.test.features, (-1, feature_length))
    test_labels = mnist.test.labels
    accu = accuracy.eval({x: test_images, y: test_labels})
    print('Accuracy: %.2f%%' % (accu * 100))
