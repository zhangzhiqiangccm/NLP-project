#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author zhang

import os
import tensorflow as tf
from transformers import BertTokenizer, BertConfig, TFBertModel
from tensorflow.keras.layers import Dense

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        # 模型名称
        self.model_name="BruceBertRNN"
        # 训练集
        self.train_path = dataset + '/data/train.txt'
        # 测试集
        self.test_path = dataset + '/data/test.txt'
        # 校验集
        self.dev_path = dataset + '/data/dev.txt'
        # dataset
        self.datasetpkl = dataset + '/data/dataset.pkl'
        # 类别
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        # 模型训练结果
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.h5'

        # 类别数
        self.num_classes = len(self.class_list)
        # epoch数
        self.num_epochs = 3
        # batch_size
        self.batch_size = 128
        # 每句话处理的长度(短填，长切）
        self.max_len = 32
        # 学习率
        self.learning_rate = 1e-5
        # bert预训练模型位置
        self.bert_path = 'bert_pretrain'
        self.bert_model_config_path = os.path.join(self.bert_path, 'bert-base-chinese-config.json')
        self.bert_model_weights_path = os.path.join(self.bert_path, 'bert-base-chinese-tf_model.h5')
        self.bert_model_vocab_path = os.path.join(self.bert_path, 'bert-base-chinese-vocab.txt')
        # bert切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_vocab_path)
        # bert隐层层个数
        self.hidden_size = 768
        # 随机失活
        self.dropout = 0.5
        # lstm隐藏层
        self.hidden_size = 128

class MyModel(tf.keras.Model):

    def __init__(self, config):
        super(MyModel, self).__init__()
        self.bert_model_config = BertConfig.from_pretrained(config.bert_model_config_path)
        self.bert_model = TFBertModel.from_pretrained(config.bert_model_weights_path,
                                                      config=self.bert_model_config)
        self.biRNN = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=config.hidden_size,
                                        return_sequences=False,
                                        activation='relu',
                                        ))
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.fc = Dense(config.num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        cls_token, _= self.bert_model(inputs)
        x = self.biRNN(cls_token)
        x = self.dropout(x)
        output = self.fc(x)
        return output
