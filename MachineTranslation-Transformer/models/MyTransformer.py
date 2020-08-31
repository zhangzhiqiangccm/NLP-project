#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author zhang
import tensorflow as tf
from models.BruceTransformer import Transformer

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'Transformer'
        self.dataset_path = dataset + '/data/en-zh.csv'                                   # 数据集
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.num_samples = 1000                                                     # 最多10000个样本参与训练

        self.num_epochs = 200                                           # epoch数
        self.batch_size = 64                                            # mini-batch大小
        self.learning_rate = 1e-3                                       # 学习率
        self.num_layers = 4                                             #迭代多少次打印输出
        self.d_model = 128
        self.dff = 512                                        # 卷积核数量(channels数)
        self.num_heads = 8                                  # encoder中token数量，在运行时赋值
        self.buffer_size = 20000
        self.max_length = 50                                      # decoder中token数量，在运行时赋值
        self.input_vocab_size = 0                                 # encoder中最大序列长度，在运行时赋值
        self.target_vocab_size  = 0                                 # dncoder中最大序列长度，在运行时赋值
        self.steps_per_epoch = 0


class MyModel(tf.keras.Model):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config


    def createModel(self):

        model = Transformer(self.config)

        return model
