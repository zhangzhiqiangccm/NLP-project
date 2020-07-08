#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author 杨博

from gensim.models import Word2Vec
import re

docs = [ "i like dog", "i like cat", "i like animal",
              "dog is animal", "cat is animal","dog like apple", "cat like fish",
              "dog like milk", "i like apple", "i hate apple",
              "i like movie", "i like book","i like music","cat hate dog", "cat like dog"]
sentences = []
# 去标点符号
stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'

for doc in docs:
    doc = re.sub(stop, '', doc)
    sentences.append(doc.split())

# size嵌入的维度，window窗口大小，workers训练线程数
# 忽略单词出现频率小于min_count的单词
# sg=1使用Skip-Gram，否则使用CBOW
model = Word2Vec(sentences, size=5, window=1, min_count=1, workers=4, sg=1)
print(model.wv['cat'])