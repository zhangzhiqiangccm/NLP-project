#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author zhang
import time
import tensorflow as tf
from importlib import import_module
from tensorflow import keras
from utils import load_dataset
from model_utils import loss_function, create_masks
from models.BruceSchedule import CustomizedSchedule
import argparse



parser = argparse.ArgumentParser(description='MatchineTranslation')
parser.add_argument('--model',default="MyTransformer", type=str, help='choose a model: Transformer')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'Data'  # 数据集
    model_name = args.model  # MyTransformer

    x = import_module('models.' + model_name) #一个函数运行需要根据不同项目的配置，动态导入对应的配置文件运行。
    config = x.Config(dataset) #进入到对应模型的__init__方法进行参数初始化
    start_time = time.time()
    print("Loading data...")
    train_dataset, valid_dataset, en_tokenizer, zh_tokenizer = load_dataset(config.dataset_path, config, config.num_samples)

    config.input_vocab_size = en_tokenizer.vocab_size + 2
    config.target_vocab_size = zh_tokenizer.vocab_size + 2

    model = x.MyModel(config)
    transformer = model.createModel()

    learning_rate = CustomizedSchedule(config.d_model)
    optimizer = keras.optimizers.Adam(learning_rate,
                                      beta_1=0.9,
                                      beta_2=0.98,
                                      epsilon=1e-9)
    #创建checkpoint
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=transformer)


    train_loss = keras.metrics.Mean(name = 'train_loss')
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(
        name = 'train_accuracy')

    #@tf.function(experimental_relax_shapes=True)
    def train_step(inp, tar):
        #target分为tar_inp和tar_real.
        # target_input是传给解码器的输入，target_real是其左移一个位置的结果，每个target_input位置对应下一个预测的标签
        tar_inp  = tar[:, :-1]
        tar_real = tar[:, 1:]

        encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask     = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp, True,
                                         encoder_padding_mask,
                                         decoder_mask,
                                         encoder_decoder_padding_mask)
            loss = loss_function(tar_real, predictions)
        # 求梯度
        gradients = tape.gradient(loss, transformer.trainable_variables)
        # 反向传播
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables))
        # 记录loss和准确率
        train_loss(loss)
        train_accuracy(tar_real, predictions)

    for epoch in range(config.num_epochs):
        start = time.time()
        # 重置记录项
        train_loss.reset_states()
        train_accuracy.reset_states()
        # inputs 英语， targets中文
        for (batch, (inp, tar)) in enumerate(train_dataset):
            # 训练
            train_step(inp, tar)
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(),
                    train_accuracy.result()))
                checkpoint.save(file_prefix = config.save_path)

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
            epoch + 1, train_loss.result(), train_accuracy.result()))
        print('Time take for 1 epoch: {} secs\n'.format(
            time.time() - start))