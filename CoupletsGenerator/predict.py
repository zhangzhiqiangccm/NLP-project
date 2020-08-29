#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author zhang

import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, Input
from importlib import import_module
from utils import process_text, bulid_token_index, build_dataset
import argparse


parser = argparse.ArgumentParser(description='PoemsGenerator')
parser.add_argument('--model',default="Seq2Seq", type=str, help='choose a model: Seq2Se2')
args = parser.parse_args()



if __name__ == '__main__':
    dataset = 'Couplets'  # 数据集

    model_name = args.model  # Seq2Seq

    x = import_module('models.' + model_name) #一个函数运行需要根据不同项目的配置，动态导入对应的配置文件运行。
    config = x.Config(dataset) #进入到对应模型的__init__方法进行参数初始化
    start_time = time.time()

    input_texts, target_texts, input_characters, target_characters = process_text(config)

    num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, input_token_index, target_token_index = bulid_token_index(
        input_texts, target_texts, input_characters, target_characters)

    encoder_input_data, decoder_input_data, decoder_target_data = build_dataset(input_texts, target_texts, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, input_token_index, target_token_index)

    # 加载模型
    model = load_model(config.save_path)

    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = tf.keras.Model(encoder_inputs, encoder_states) #获取编码器

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = Input(shape=(config.hidden_size,))
    decoder_state_input_c = Input(shape=(config.hidden_size,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states) # 获取解码器

    # 构建索引到token
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())


    # 解码序列
    def decode_sequence(input_seq):
        # 编码出入
        states_value = encoder_model.predict(input_seq)

        # 生成长度为1的空目前序列
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # 用#号填充目标序列的第一个字符。
        target_seq[0, 0, target_token_index['#']] = 1.

        # 训练采样，假设batch_size为1.
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # 采样token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # 结束条件: 超过max_decoder_seq_length或者遇到结束字符'\n'
            if (sampled_char == '\n' or
                    len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # 更新序列
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # 更新状态
            states_value = [h, c]

        return decoded_sentence


    for seq_index in range(100):
        # 进行检测
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)