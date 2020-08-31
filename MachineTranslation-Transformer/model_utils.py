#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author zhang
import tensorflow as tf
import numpy as np
from tensorflow import keras


# 将位置编码矢量添加得到词嵌入，相同位置的词嵌入将会更接近，但并不能直接编码相对位置
# 基于角度的位置编码方法如下：
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
# pos.shape: [sentence_length, 1]
# i.shape  : [1, d_model]
# result.shape: [sentence_length, d_model]
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000,
                               (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def get_position_embedding(sentence_length, d_model):
    angle_rads = get_angles(np.arange(sentence_length)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # sines.shape: [sentence_length, d_model / 2]
    # cosines.shape: [sentence_length, d_model / 2]
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])

    # position_embedding.shape: [sentence_length, d_model]
    position_embedding = np.concatenate([sines, cosines], axis = -1)
    # position_embedding.shape: [1, sentence_length, d_model]
    position_embedding = position_embedding[np.newaxis, ...]

    return tf.cast(position_embedding, dtype=tf.float32)


#为了避免输入中padding的token对句子语义的影响，需要将padding位mark掉，原来为0的padding项的mark输出为1
# 1. padding mask, 2. look ahead
# batch_data.shape: [batch_size, seq_len]
def create_padding_mask(batch_data):
    padding_mask = tf.cast(tf.math.equal(batch_data, 0), tf.float32)
    # [batch_size, 1, 1, seq_len]
    return padding_mask[:, tf.newaxis, tf.newaxis, :]

# look-ahead mask 用于对未预测的token进行掩码 这意味着要预测第三个单词，
# 只会使用第一个和第二个单词。 要预测第四个单词，仅使用第一个，第二个和第三个单词，依此类推。
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask # (seq_len, seq_len)

# 前向网络
def feed_forward_network(d_model, dff):
    # dff: dim of feed forward network.
    return keras.Sequential([
        keras.layers.Dense(dff, activation='relu'),
        keras.layers.Dense(d_model)
    ])

# 由于目标序列是填充的，因此在计算损耗时应用填充掩码很重要。 padding的掩码为0，没padding的掩码为1
loss_object = keras.losses.SparseCategoricalCrossentropy(
    from_logits = True, reduction = 'none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# 构建掩码
def create_masks(inp, tar):
    """
    Encoder:
      - encoder_padding_mask (self attention of EncoderLayer)
    Decoder:
      - look_ahead_mask (self attention of DecoderLayer)
      - encoder_decoder_padding_mask (encoder-decoder attention of DecoderLayer)
      - decoder_padding_mask (self attention of DecoderLayer)
    """
    encoder_padding_mask = create_padding_mask(inp)
    # 这个掩码用于掩输入解码层第二层的编码层输出
    encoder_decoder_padding_mask = create_padding_mask(inp)
    # look_ahead 掩码， 掩掉未预测的词
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    # 解码层第一层得到padding掩码
    decoder_padding_mask = create_padding_mask(tar)
    # 合并解码层第一层掩码
    decoder_mask = tf.maximum(decoder_padding_mask,
                              look_ahead_mask)

    return encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask