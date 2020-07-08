#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author 张

from gensim.models import Word2Vec
import utils

"""
 训练Word2Vec
"""

def train_w2v_model(sentences, embedding_vector_size):
    w2v_model = Word2Vec(
        sentences=sentences,
        size=embedding_vector_size,
        min_count=3, window=5, workers=4)
    w2v_model.save("data\word2vec.model")


if __name__ =="__main__":
    corpus_path = "data\data.tsv"
    sentences, _ = utils.read_corpus(corpus_path)
    train_w2v_model(sentences, 100)