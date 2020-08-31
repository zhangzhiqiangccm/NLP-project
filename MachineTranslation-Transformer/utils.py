#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author zhang
import tensorflow as tf

import unicodedata
import re
import os
import io
import time
import numpy as np
from tqdm import tqdm
import time
from datetime import timedelta
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds



def preprocess_sentence(w):
    w = w.lower().strip()
    w = w.rstrip().strip()
    return w

# 1. 去除重音符号
# 2. 清理句子
# 3. 返回这样格式的单词对：[source, target]
def create_dataset(path, num_examples):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

    return zip(*word_pairs)


def load_dataset(path, config, num_examples=None):
    # 创建清理过的输入输出对
    input_text, targ_text = create_dataset(path, num_examples)

    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_text, targ_text,
                                                                                                    test_size=0.2)

    train_examples = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
    val_examples = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))

    #构建词汇表
    en_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for en, zh in train_examples),
        target_vocab_size=2 ** 13)
    zh_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (zh.numpy().decode('utf-8') for en, zh in train_examples),
        target_vocab_size=2 ** 13)

    # 添加start 和 end的token表示
    def encode_to_subword(en_sentence, zh_sentence):
        en_sequence = [en_tokenizer.vocab_size] + en_tokenizer.encode(en_sentence.numpy()) + [
            en_tokenizer.vocab_size + 1]
        zh_sequence = [zh_tokenizer.vocab_size] + zh_tokenizer.encode(zh_sentence.numpy().decode('utf-8')) + [
            zh_tokenizer.vocab_size + 1]
        return en_sequence, zh_sequence

    # 过滤长度大于 max_length的句子
    def filter_by_max_length(en, zh):
        return tf.logical_and(tf.size(en) <= config.max_length,
                              tf.size(zh) <= config.max_length)

    # 将python运算，转换为tensorflow运算节点
    def tf_encode_to_subword(en_sentence, zh_sentence):
        return tf.py_function(encode_to_subword,
                              [en_sentence, zh_sentence],
                              [tf.int64, tf.int64])

    #使用.map()运行相关图操作
    train_dataset = train_examples.map(tf_encode_to_subword)
    # 过滤过长的数据
    train_dataset = train_dataset.filter(filter_by_max_length)
    # 使用缓存数据加速读入
    train_dataset = train_dataset.cache()
    # 打乱并获取批数据
    train_dataset = train_dataset.shuffle(
        config.buffer_size).padded_batch(
        config.batch_size, padded_shapes=([-1], [-1]))
    # 验证集数据
    valid_dataset = val_examples.map(tf_encode_to_subword)
    valid_dataset = valid_dataset.filter(
        filter_by_max_length).padded_batch(
        config.batch_size, padded_shapes=([-1], [-1]))

    return train_dataset, valid_dataset, en_tokenizer, zh_tokenizer

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

