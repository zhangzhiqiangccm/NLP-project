#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author zhang
import time
from importlib import import_module
from utils import build_dataset, get_time_dif,process_text, bulid_token_index, build_dataset
import argparse


parser = argparse.ArgumentParser(description='PoemsGenerator')
parser.add_argument('--model',default="Seq2Seq", type=str, help='choose a model: Seq2Seq')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'Couplets'  # 数据集
    model_name = args.model  # Seq2Seq

    x = import_module('models.' + model_name) #一个函数运行需要根据不同项目的配置，动态导入对应的配置文件运行。
    config = x.Config(dataset) #进入到对应模型的__init__方法进行参数初始化
    start_time = time.time()
    print("Loading data...")
    input_texts, target_texts, input_characters, target_characters = process_text(config)

    num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, input_token_index, target_token_index = bulid_token_index(
        input_texts, target_texts, input_characters, target_characters)

    encoder_input_data, decoder_input_data, decoder_target_data = build_dataset(input_texts, target_texts, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length, input_token_index, target_token_index)

    config.num_encoder_tokens = num_encoder_tokens
    config.num_decoder_tokens = num_decoder_tokens
    config.max_encoder_seq_length = max_encoder_seq_length
    config.max_decoder_seq_length = max_decoder_seq_length
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.MyModel(config)
    model = model.createModel()
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit([encoder_input_data, decoder_input_data],
              decoder_target_data,
              batch_size=config.batch_size,
              epochs=config.num_epochs,
              validation_split=0.2)
    # Save model
    model.save(config.save_path)

