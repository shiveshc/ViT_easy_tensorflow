import tensorflow as tf
import numpy as np


def layer_norm(x):
    mean, var = tf.nn.moments(x, axes= [-1], keep_dims= True)
    epsilon = 1e-8

    beta = tf.get_variable('beta', shape= (x.shape[2]), initializer= tf.initializers.zeros)
    gamma = tf.get_variable('gamma', shape= (x.shape[2]), initializer= tf.initializers.ones)

    # beta = tf.get_variable(tf.zeros([x.shape[3]]))
    # gamma = tf.Variable(tf.ones([x.shape[3]]))

    x = tf.divide(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))
    x = gamma*x + beta

    return x

def embedding_layer(x, d_model):
    We = tf.get_variable('e_w', shape=(x.shape[2], d_model), initializer=tf.contrib.layers.xavier_initializer())
    x = tf.tensordot(x, We, [[2], [0]])

    return x

def pos_embedding(x, d_model):
    bs = tf.shape(x)[0]
    Wp = tf.get_variable('pos_w', shape=(x.shape[1], d_model), initializer=tf.contrib.layers.xavier_initializer())
    out = tf.tile(tf.expand_dims(Wp, axis= 0), [bs, 1, 1])

    return out

def MSA(q, k, v, d_model, num_heads, d_proj):

    bs = tf.shape(q)[0]

    q_in = tf.tile(tf.expand_dims(q, axis=1), [1, num_heads, 1, 1])
    k_in = tf.tile(tf.expand_dims(k, axis=1), [1, num_heads, 1, 1])
    v_in = tf.tile(tf.expand_dims(v, axis=1), [1, num_heads, 1, 1])

    Wq = tf.get_variable('msa_wq', shape=(num_heads, d_model, d_proj), initializer=tf.contrib.layers.xavier_initializer())
    Wk = tf.get_variable('msa_wk', shape=(num_heads, d_model, d_proj), initializer=tf.contrib.layers.xavier_initializer())
    Wv = tf.get_variable('msa_wv', shape=(num_heads, d_model, d_proj), initializer=tf.contrib.layers.xavier_initializer())
    Q = tf.matmul(q_in, tf.tile(tf.expand_dims(Wq, axis= 0), [bs, 1, 1, 1]))
    K = tf.matmul(k_in, tf.tile(tf.expand_dims(Wk, axis= 0), [bs, 1, 1, 1]))
    V = tf.matmul(v_in, tf.tile(tf.expand_dims(Wv, axis= 0), [bs, 1, 1, 1]))

    att_wts = tf.nn.softmax(tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2]))/np.sqrt(d_proj), axis= -1)
    att_score = tf.matmul(att_wts, V)

    # att_concat = tf.reshape(att, [tf.shape(att)[0], d_model, -1])
    att_concat = tf.concat(tf.unstack(att_score, axis= 1), axis= 2)
    Wo = tf.get_variable('msa_wo', shape=(att_concat.shape[2], d_model), initializer=tf.contrib.layers.xavier_initializer())
    out = tf.tensordot(att_concat, Wo, [[2], [0]])

    return out

def ffn(x, d_model, d_inner):

    W1_ffn = tf.get_variable('ffn_w1', shape=(d_model, d_inner), initializer=tf.contrib.layers.xavier_initializer())
    B1_ffn = tf.get_variable('ffn_b1', shape=(d_inner), initializer=tf.contrib.layers.xavier_initializer())
    W2_ffn = tf.get_variable('ffn_w2', shape=(d_inner, d_model), initializer=tf.contrib.layers.xavier_initializer())
    B2_ffn = tf.get_variable('ffn_b2', shape=(d_model), initializer=tf.contrib.layers.xavier_initializer())

    out = tf.nn.bias_add(tf.tensordot(x, W1_ffn, [[2], [0]]), B1_ffn)
    out = tf.nn.relu(out)
    out = tf.nn.bias_add(tf.tensordot(out, W2_ffn, [[2], [0]]), B2_ffn)

    return out

def class_head(z, d_model, num_classes):
    W1_ch = tf.get_variable('ch_w1', shape=(d_model, 1024), initializer=tf.contrib.layers.xavier_initializer())
    B1_ch = tf.get_variable('ch_b1', shape=(1024), initializer=tf.contrib.layers.xavier_initializer())
    W2_ch = tf.get_variable('ch_w2', shape=(1024, num_classes), initializer=tf.contrib.layers.xavier_initializer())
    B2_ch = tf.get_variable('ch_b2', shape=(num_classes), initializer=tf.contrib.layers.xavier_initializer())

    out = tf.nn.bias_add(tf.matmul(z, W1_ch), B1_ch)
    out = tf.nn.relu(out)
    out = tf.nn.bias_add(tf.matmul(out, W2_ch), B2_ch)

    return out


def net(x, num_blocks, d_model, num_heads, d_proj, d_inner, num_classes):

    x = embedding_layer(x, d_model)
    cls_token = tf.zeros((tf.shape(x)[0], 1, d_model), 'float')
    x = tf.concat([cls_token, x], axis= 1)

    pos_embed = pos_embedding(x, d_model)

    x = tf.add(x, pos_embed)


    for block in range(num_blocks):
        with tf.variable_scope('block_' + str(block), reuse= tf.AUTO_REUSE):
            with tf.variable_scope('LN1', reuse= tf.AUTO_REUSE):
                x_norm = layer_norm(x)
            with tf.variable_scope('MSA'):
                x_att = MSA(x_norm, x_norm, x_norm, d_model, num_heads, d_proj)
            x = x_att + x

            with tf.variable_scope('LN2', reuse=tf.AUTO_REUSE):
                x_norm = layer_norm(x)
            with tf.variable_scope('ffn'):
                x_ffn = ffn(x_norm, d_model, d_inner)
            x = x_ffn + x

    with tf.variable_scope('class_head', reuse= tf.AUTO_REUSE):
        logits = class_head(tf.gather(x, 0, axis= 1), d_model, num_classes)

    return logits


