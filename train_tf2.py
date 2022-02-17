import tensorflow as tf
import numpy as np
from arch import *
import os
import shutil
from tensorflow import keras
import random
import cv2
import tensorflow_addons as tfa

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

def loss_fn(y_true, y_pred):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels= y_true, logits= y_pred)

    return loss



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


    x = keras.Input(shape= (x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    vit = model_v3_tf2.ViT(x, num_blocks, d_model, num_heads, d_proj, d_inner, num_classes, patch_size)
    vit.compile(optimizer= tfa.optimizers.AdamW(learning_rate= lr, weight_decay= 0.0001), loss= loss_fn)

    history = vit.fit(x_train, y_train, batch_size= 256, epochs= 10)

