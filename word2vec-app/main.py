#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author zhang

import utils

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, GlobalAveragePooling1D



sequences, data = utils.read_corpus("data\data.tsv")
w2v_model = utils.load_w2v("data\word2vec.model")
word2id = utils.word2id(w2v_model)
X_data = utils.get_sequences(word2id, sequences)

# 截长补短
maxlen = 20
X_pad = pad_sequences(X_data, maxlen=maxlen)
# 取得标签
Y = data.sentiment.values
# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(
    X_pad,
    Y,
    test_size=0.2,
    random_state=42)

"""
 构建分类模型
"""
# 让 tf.keras 的 Embedding 层使用训练好的Word2Vec权重
embedding_matrix = w2v_model.wv.vectors

model = Sequential()
model.add(Embedding(
    input_dim=embedding_matrix.shape[0],
    output_dim=embedding_matrix.shape[1],
    input_length=maxlen,
    weights=[embedding_matrix],
    trainable=False))
#model.add(Flatten())
model.add(GlobalAveragePooling1D())
model.add(Dense(5))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss="binary_crossentropy",
    optimizer='adam',
    metrics=['accuracy'])

model.summary()
history = model.fit(
    x=X_train,
    y=Y_train,
    validation_data=(X_test, Y_test),
    batch_size=128,
    epochs=10)