import tensorflow as tf
import numpy as np


def get_answer(test_list, dir_prefix=''):
    feature_length = 284
    keep_rate = 0.9
    x = tf.placeholder('float', shape=[None, feature_length])
    pkeep = tf.placeholder(tf.float32)

    # five layers and their number of neurons (tha last layer has 10 softmax neurons)
    L = 160
    M = 90
    N = 50
    O = 20

    W1 = tf.Variable(tf.truncated_normal([feature_length, L], stddev=0.1))  # 784 = 28 * 28
    B1 = tf.Variable(tf.ones([L]) / 10)
    W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
    B2 = tf.Variable(tf.ones([M]) / 10)
    W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
    B3 = tf.Variable(tf.ones([N]) / 10)
    W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
    B4 = tf.Variable(tf.ones([O]) / 10)
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

    model = tf.nn.softmax(Ylogits)
    predictions = tf.argmax(model, 1)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        model_file = tf.train.latest_checkpoint(dir_prefix + 'save/')
        saver.restore(sess, model_file)

        test_images = np.reshape(test_list, (-1, feature_length))
        pre, confidence = sess.run([predictions, model], feed_dict={x: test_images, pkeep: keep_rate})

        return pre, confidence


if __name__ == '__main__':
    get_answer([
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
        'SKSEDSAEAYTSRSKSQDSTQTQSSSN'
    ])
    pass
