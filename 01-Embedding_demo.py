#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author zhang

import tensorflow as tf

docs =[ "i like dog", "i like cat", "i like animal",
              "dog is animal", "cat is animal","dog like apple", "cat like fish",
              "dog like milk", "i like apple", "i hate apple",
              "i like movie", "i like book","i like music","cat hate dog", "cat like dog"]

# 只考虑最常见的15个单词
max_words = 15

# 统一的序列化长度
# 截长补短 0填充
max_len = 3

# 词嵌入维度
embedding_dim = 3

# 分词
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)

# fit_on_texts 获取训练文本的词表
tokenizer.fit_on_texts(docs)

# 字典索引
word_index = tokenizer.word_index

# 序列化
sequences = tokenizer.texts_to_sequences(docs)

# 统一序列长度
data = tf.keras.preprocessing.sequence.pad_sequences(sequences = sequences, maxlen= max_len)

# Embedding层
model = tf.keras.models.Sequential()

embedding_layer = tf.keras.layers.Embedding(input_dim=max_words, output_dim= embedding_dim, input_length=max_len)

model.add(embedding_layer)

model.compile('rmsprop', 'mse')

out = model.predict(data)

print(out)
print(out.shape)

# 查看权重
layer = model.get_layer('embedding')
print(layer.get_weights())