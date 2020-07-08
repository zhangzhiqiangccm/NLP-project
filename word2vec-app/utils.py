#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author zhang

import pandas as pd
import re
import codecs
import numpy as np

from gensim.models import Word2Vec

"""
 清除标点符合及非法字符
"""
def split_sentence(sentence):
    stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    sentence = re.sub(stop, '', sentence)
    return sentence.split()

"""
 读取数据集
"""
def read_corpus(path):
    data = pd.read_csv(path, sep='\t')
    sentences  = data.review.apply(split_sentence)
    return sentences, data

"""
 加载词向量
"""
def load_w2v(w2v_path):
    return Word2Vec.load(w2v_path)

"""
 根据词向量构建word2id
"""
def word2id(w2v_model):
    # 取得所有单词
    vocab_list = list(w2v_model.wv.vocab.keys())
    # 每个词语对应的索引
    word2id = {word: index for index, word in enumerate(vocab_list)}
    return word2id

"""
 获得编码后的序列
"""
def get_sequences(word2id, sentences):
    sequences = []
    for sentence in sentences:
        sequence = []
        for word in sentence:
            try:
                sequence.append(word2id[word])
            except KeyError:
                pass
        sequences.append(sequence)
    return sequences