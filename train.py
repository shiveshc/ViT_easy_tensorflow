import tensorflow as tf
from arch import *
import os
import shutil
from tensorflow import keras
import random


if __name__ == '__main__':

    #load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()


    num_heads = 8
    d_model = 512
    d_proj = d_model/num_heads
    d_inner = 1024
    num_blocks = 6
    num_classes = y_train.shape[1]

    lr = 0.001
    batch_size = 50
    iters = 100


    x = tf.placeholder('float', [None, 256, 256, 3])
    y = tf.placeholder('float', [None, num_classes])

    x_patch = tf.extract_image_patches(x, [1, 64, 64, 1], [1, 64, 64, 1], [1, 1, 1, 1], 'SAME')
    x_patch = tf.reshape(x_patch, [-1, x_patch.shape[1]*x_patch.shape[2], x_patch.shape[3]])
    logits = model.net(x_patch, num_blocks, d_model, num_heads, d_proj, d_inner, num_classes)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= logits))

    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(loss)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    # make folder where all results will be saved
    results_dir = 'D:/Shivesh/ViT/Results/ViT_train'
    # results_dir = '/storage/scratch1/0/schaudhary9/SinGAN/SinGan_train_balloons_' + str(run)
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)

    os.mkdir(results_dir)

    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter(results_dir, sess.graph)

        idx = random.sample(range(x_train.shape[3]), 2000)
        curr_batch_x = x_train[idx, :, :, :]
        curr_batch_y = y_train[idx, :]

        for n in range(iters):
            for batch in range(curr_batch_x.shape[0]//batch_size):
                batch_x = curr_batch_x[batch * batch_size:min((batch + 1) * batch_size, curr_batch_x.shape[0])]
                batch_y = curr_batch_y[batch * batch_size:min((batch + 1) * batch_size, curr_batch_y.shape[0])]
                op = sess.run(opt, feed_dict= {x: batch_x, y: batch_y})

            idx = random.sample(range(x_train.shape[3]), 50)
            sample_train_x = x_train[idx, :, :, :]
            sample_train_y = y_train[idx, :]
            train_loss = sess.run(loss, feed_dict= {x: sample_train_x, y: sample_train_y})

            idx = random.sample(range(x_test.shape[3]), 50)
            sample_test_x = x_test[idx, :, :, :]
            sample_test_y = y_test[idx, :]
            test_loss = sess.run(loss, feed_dict={x: sample_test_x, y: sample_test_y})

            print('Iter - ' + str(n) + ':: Train loss - ' + "{:.6f}".format(train_loss) + ', Test loss - ' + "{:.6f}".format(test_loss))


        summary_writer.close()

