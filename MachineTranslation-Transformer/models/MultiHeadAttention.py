#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author 杨博

import tensorflow as tf
from tensorflow import keras

class MultiHeadAttention(keras.layers.Layer):
    """
    理论上:
    x -> Wq0 -> q0
    x -> Wk0 -> k0
    x -> Wv0 -> v0

    实战中:
    q -> Wq0 -> q0
    k -> Wk0 -> k0
    v -> Wv0 -> v0

    实战中技巧：
    q -> Wq -> Q -> split -> q0, q1, q2...
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        # d_model 必须可以正确分为各个头
        self.d_model = d_model
        assert self.d_model % self.num_heads == 0

        # 分头后的维度
        self.depth = self.d_model // self.num_heads

        self.WQ = keras.layers.Dense(self.d_model)
        self.WK = keras.layers.Dense(self.d_model)
        self.WV = keras.layers.Dense(self.d_model)

        self.dense = keras.layers.Dense(self.d_model)

    # 分头, 将头个数的维度 放到 seq_len 前面
    def split_heads(self, x, batch_size):
        # x.shape: (batch_size, seq_len, d_model)
        # d_model = num_heads * depth
        # x -> (batch_size, num_heads, seq_len, depth)

        x = tf.reshape(x,
                       (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        # 分头前的前向网络，获取q、k、v语义
        q = self.WQ(q)  # q.shape: (batch_size, seq_len_q, d_model)
        k = self.WK(k)  # k.shape: (batch_size, seq_len_k, d_model)
        v = self.WV(v)  # v.shape: (batch_size, seq_len_v, d_model)

        # 分头
        # q.shape: (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # k.shape: (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # v.shape: (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # 通过缩放点积注意力层
        # scaled_attention_outputs.shape: (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention_outputs, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # 把多头维度后移
        # scaled_attention_outputs.shape: (batch_size, seq_len_q, num_heads, depth)
        scaled_attention_outputs = tf.transpose(
            scaled_attention_outputs, perm=[0, 2, 1, 3])

        # 合并多头
        # concat_attention.shape: (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention_outputs,
                                      (batch_size, -1, self.d_model))
        # 全连接重塑
        # output.shape : (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights

def scaled_dot_product_attention(q, k, v, mask):
    """
    Args:
    - q: shape == (..., seq_len_q, depth)
    - k: shape == (..., seq_len_k, depth)
    - v: shape == (..., seq_len_v, depth_v)
    - seq_len_k == seq_len_v
    - mask: shape == (..., seq_len_q, seq_len_k)
    Returns:
    - output: weighted sum
    - attention_weights: weights of attention
    """
    # query key 相乘获取匹配关系
    # matmul_qk.shape: (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b = True)

    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # 掩码
    if mask is not None:
        # 使得在softmax后值趋近于0
        scaled_attention_logits += (mask * -1e9)

    # 通过softmax获取attention权重
    # attention_weights.shape: (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis = -1)

    # attention 乘上value
    # output.shape: (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights