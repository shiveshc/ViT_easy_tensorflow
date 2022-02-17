import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_addons as tfa

def layer_norm(x):
    mean, var = tf.nn.moments(x, axes= [-1], keepdims= True)
    epsilon = 1e-6

    # beta = tf.get_variable('beta', shape= (x.shape[2]), initializer= tf.initializers.zeros)
    # gamma = tf.get_variable('gamma', shape= (x.shape[2]), initializer= tf.initializers.ones)

    # beta = tf.get_variable(tf.zeros([x.shape[3]]))
    # gamma = tf.Variable(tf.ones([x.shape[3]]))

    x = tf.divide(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))
    # temp_x = tf.transpose(x, [0, 2, 1])
    # temp_x = gamma*temp_x + beta
    # x = tf.transpose(temp_x, [0, 2, 1])

    return x

def embedding_layer(x, d_model):
    dense = keras.layers.Dense(d_model)
    x = dense(x)

    return x

def pos_embedding(x, d_model):
    embed = keras.layers.Embedding(input_dim= x.shape[1], output_dim= d_model)
    out = embed(tf.range(0, x.shape[1]))

    return out

def MSA(q, k, v, d_model, num_heads, d_proj):

    bs = tf.shape(q)[0]

    Wq = keras.layers.Dense(num_heads*d_proj)
    Wk = keras.layers.Dense(num_heads*d_proj)
    Wv = keras.layers.Dense(num_heads*d_proj)
    Q = Wq(q)
    K = Wk(k)
    V = Wv(v)

    Q = tf.transpose(tf.reshape(Q, [-1, Q.shape[1], num_heads, d_proj]), [0, 2, 1, 3])
    K = tf.transpose(tf.reshape(K, [-1, K.shape[1], num_heads, d_proj]), [0, 2, 1, 3])
    V = tf.transpose(tf.reshape(V, [-1, V.shape[1], num_heads, d_proj]), [0, 2, 1, 3])

    att_wts = tf.nn.softmax(tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2]))/np.sqrt(d_proj), axis= -1)
    att_score = tf.matmul(att_wts, V)

    # att_concat = tf.reshape(att, [tf.shape(att)[0], d_model, -1])
    att_concat = tf.concat(tf.unstack(att_score, axis= 1), axis= 2)
    Wo = keras.layers.Dense(d_model)
    out = Wo(att_concat)

    return out

def ffn(x, d_model, d_inner, drop_rate):

    dense1 = keras.layers.Dense(d_inner, activation= tfa.activations.gelu)
    dense2 = keras.layers.Dense(d_model, activation= tfa.activations.gelu)

    out = dense1(x)
    out = tf.nn.dropout(out, 1 - drop_rate)
    out = dense2(out)
    out = tf.nn.dropout(out, 1 - drop_rate)

    return out

def class_head(z, num_classes, drop_rate):
    dense1 = keras.layers.Dense(2048, activation= tfa.activations.gelu)
    dense2 = keras.layers.Dense(1024, activation= tfa.activations.gelu)
    dense3 = keras.layers.Dense(num_classes)

    out = dense1(z)
    out = tf.nn.dropout(out, 1 - drop_rate)
    out = dense2(out)
    out = tf.nn.dropout(out, 1 - drop_rate)
    out = dense3(out)

    return out


def net(x, num_blocks, d_model, num_heads, d_proj, d_inner, num_classes):

    x = embedding_layer(x, d_model)
    pos_embed = pos_embedding(x, d_model)

    x = tf.add(x, pos_embed)


    for block in range(num_blocks):
        x_norm = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        # x_norm = layer_norm(x)
        x_att = MSA(x_norm, x_norm, x_norm, d_model, num_heads, d_proj)
        x = x_att + x

        x_norm = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        # x_norm = layer_norm(x)
        x_ffn = ffn(x_norm, d_model, d_inner, 0.1)
        x = x_ffn + x

    # x = layer_norm(x)
    x = tf.reshape(x, [-1, x.shape[1]*x.shape[2]])
    x = tf.nn.dropout(x, 0.5)
    logits = class_head(x, num_classes, 0.5)

    return logits

def ViT(x, num_blocks, d_model, num_heads, d_proj, d_inner, num_classes, patch_size):
    x_patch = tf.image.extract_patches(x, [1, patch_size, patch_size, 1], [1, patch_size, patch_size, 1], [1, 1, 1, 1], 'SAME')
    x_patch = tf.reshape(x_patch, [-1, x_patch.shape[1] * x_patch.shape[2], x_patch.shape[3]])

    logits = net(x_patch, num_blocks, d_model, num_heads, d_proj, d_inner, num_classes)
    model = keras.Model(inputs= x, outputs= logits)

    return model


