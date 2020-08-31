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
import matplotlib.pyplot as plt
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

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=transformer)

    if tf.train.latest_checkpoint(config.save_path):
        checkpoint.restore(tf.train.latest_checkpoint(config.save_path))


    train_loss = keras.metrics.Mean(name = 'train_loss')
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(
        name = 'train_accuracy')


    def evaluate(inp_sentence):
        input_id_sentence = [en_tokenizer.vocab_size] + en_tokenizer.encode(inp_sentence) + [en_tokenizer.vocab_size + 1]
        # encoder_input.shape: (1, input_sentence_length)
        encoder_input = tf.expand_dims(input_id_sentence, 0)

        # decoder_input.shape: (1, 1)
        decoder_input = tf.expand_dims([zh_tokenizer.vocab_size], 0)

        for i in range(config.max_length):
            encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask = create_masks(encoder_input, decoder_input)
            # predictions.shape: (batch_size, output_target_len, target_vocab_size)
            predictions, attention_weights = transformer(
                encoder_input,
                decoder_input,
                False,
                encoder_padding_mask,
                decoder_mask,
                encoder_decoder_padding_mask)
            # predictions.shape: (batch_size, target_vocab_size)
            predictions = predictions[:, -1, :]

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1),
                                   tf.int32)

            if tf.equal(predicted_id, zh_tokenizer.vocab_size + 1):
                return tf.squeeze(decoder_input, axis=0), attention_weights

            decoder_input = tf.concat([decoder_input, [predicted_id]],
                                      axis=-1)
        return tf.squeeze(decoder_input, axis=0), attention_weights


    def plot_encoder_decoder_attention(attention, input_sentence,
                                       result, layer_name):
        fig = plt.figure(figsize=(16, 8))

        input_id_sentence = en_tokenizer.encode(input_sentence)

        # attention.shape: (num_heads, tar_len, input_len)
        attention = tf.squeeze(attention[layer_name], axis=0)

        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head + 1)

            ax.matshow(attention[head][:-1, :])

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(input_id_sentence) + 2))
            ax.set_yticks(range(len(result)))

            ax.set_ylim(len(result) - 1.5, -0.5)

            ax.set_xticklabels(
                ['<start>'] + [zh_tokenizer.decode([i]) for i in input_id_sentence] + ['<end>'],
                fontdict=fontdict, rotation=90)
            ax.set_yticklabels(
                [zh_tokenizer.decode([i]) for i in result if i < zh_tokenizer.vocab_size],
                fontdict=fontdict)
            ax.set_xlabel('Head {}'.format(head + 1))
        plt.tight_layout()
        plt.show()


    def translate(input_sentence, layer_name=''):
        result, attention_weights = evaluate(input_sentence)

        predicted_sentence = zh_tokenizer.decode(
            [i for i in result if i < zh_tokenizer.vocab_size])

        print("Input: {}".format(input_sentence))
        print("Predicted translation: {}".format(predicted_sentence))

        if layer_name:
            plot_encoder_decoder_attention(attention_weights, input_sentence,
                                           result, layer_name)


    translate('Hello world')