import tensorflow as tf
import numpy as np
from arch import *
import os
import shutil
from tensorflow import keras
import random
import cv2

def preprocess_img(img, img_size):

    scaled_img = np.zeros_like(img, dtype= 'float')
    for c in range(img.shape[3]):
        curr_c = img[:, :, :, c]
        curr_mean = np.mean(curr_c)
        curr_std = np.sqrt(np.var(curr_c))
        scaled_img[:, :, :, c] = (curr_c - curr_mean)/curr_std

    # scaled_img = img/127.5 - 1
    # scaled_img = img

    resized_img = np.zeros((scaled_img.shape[0], img_size, img_size, scaled_img.shape[3]))
    for n in range(scaled_img.shape[0]):
        resized_img[n] = cv2.resize(scaled_img[n], dsize=(img_size, img_size))

    return resized_img

def to_one_hot(x, num_classes):
    one_hot = np.zeros([x.shape[0], num_classes])
    idx = list(range(x.shape[0]))
    one_hot[idx, x[:, 0]] = 1

    return one_hot

def calculate_accuracy(pred, truth):
    pred_class = []
    true_class = []
    for n in range(pred.shape[0]):
        curr_pred_class = [i for i in range(pred.shape[1]) if pred[n, i] == np.max(pred[n, :])]
        pred_class.append(curr_pred_class[0])

        curr_true_class = [i for i in range(truth.shape[1]) if truth[n, i] == np.max(truth[n, :])]
        true_class.append(curr_true_class[0])

    correct = [1 if pred_class[n] == true_class[n] else 0 for n in range(len(pred_class))]
    accuracy = np.sum(correct)/len(correct)

    return accuracy



if __name__ == '__main__':

    #load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    x_train = preprocess_img(x_train, 72)
    x_test = preprocess_img(x_test, 72)
    num_classes = np.max(y_train[:, 0]) + 1
    y_train = to_one_hot(y_train, num_classes)
    y_test = to_one_hot(y_test, num_classes)

    patch_size = 6
    num_heads = 4
    d_model = 64
    d_proj = int(d_model/num_heads)
    d_inner = d_model*2
    num_blocks = 8

    lr = 0.001
    batch_size = 256
    iters = 500


    x = tf.placeholder('float', [None, x_train.shape[1], x_train.shape[2], x_train.shape[3]])
    y = tf.placeholder('float', [None, num_classes])

    x_patch = tf.extract_image_patches(x, [1, patch_size, patch_size, 1], [1, patch_size, patch_size, 1], [1, 1, 1, 1], 'SAME')
    x_patch = tf.reshape(x_patch, [-1, x_patch.shape[1]*x_patch.shape[2], x_patch.shape[3]])

    logits = model_v3.net(x_patch, num_blocks, d_model, num_heads, d_proj, d_inner, num_classes)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= logits))
    opt = tf.train()

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

        idx = random.sample(range(x_train.shape[0]), 2000)
        curr_batch_x = x_train[idx, :, :, :]
        curr_batch_y = y_train[idx, :]

        for n in range(iters):
            for batch in range(curr_batch_x.shape[0]//batch_size):
                batch_x = curr_batch_x[batch * batch_size:min((batch + 1) * batch_size, curr_batch_x.shape[0])]
                batch_y = curr_batch_y[batch * batch_size:min((batch + 1) * batch_size, curr_batch_y.shape[0])]
                op = sess.run(opt, feed_dict= {x: batch_x, y: batch_y})

            idx = random.sample(range(x_train.shape[0]), 50)
            sample_train_x = x_train[idx, :, :, :]
            sample_train_y = y_train[idx, :]
            train_loss, train_logits = sess.run([loss, logits], feed_dict= {x: sample_train_x, y: sample_train_y})
            train_accuracy = calculate_accuracy(train_logits, sample_train_y)

            idx = random.sample(range(x_test.shape[0]), 50)
            sample_test_x = x_test[idx, :, :, :]
            sample_test_y = y_test[idx, :]
            test_loss, test_logits = sess.run([loss, logits], feed_dict={x: sample_test_x, y: sample_test_y})
            test_accuracy = calculate_accuracy(test_logits, sample_test_y)

            print('Iter - ' + str(n) + ':: Train loss - ' + "{:.6f}".format(train_loss) + ', Train acc - ' + "{:.6f}".format(train_accuracy) + ', Test loss - ' + "{:.6f}".format(test_loss) + ', Test acc - ' + "{:.6f}".format(test_accuracy))


        summary_writer.close()

