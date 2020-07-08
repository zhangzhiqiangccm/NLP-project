#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author zhang

from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding,GlobalAveragePooling1D

# 特征单词数
max_words = 10000
# 在50单词后截断文本
# 这些单词都属于max_words中的单词
maxlen = 20
# 嵌入维度
embedding_dim = 8

# 加载数据集
# 加载的数据已经序列化过了，每个样本都是一个sequence列表
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

# 统计序列长度，将数据集转换成形状为（samples，maxlen）的二维整数张量
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# 构建模型
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
# 将3维的嵌入张量展平成形状为（samples，maxlen * embedding_dim）的二维张量
model.add(Flatten())
#model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

print(model)

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)