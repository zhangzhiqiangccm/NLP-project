#!/usr/bin/python
# -*- coding: UTF-8 -*-
#Author zhang
import os
import tensorflow as tf
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta



def build_dataset(config):
    def load_dataset(path):
        lables = []
        input_ids, token_type_ids, attention_masks = [], [], []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                input_id, token_type_id, attention_mask = convert_content_to_inputs(content, config)
                input_ids.append(input_id)
                token_type_ids.append(token_type_id)
                attention_masks.append(attention_mask)
                lables.append(int(label))
        return (list(map(lambda x: np.asarray(x, dtype=np.int32), [input_ids, attention_masks, token_type_ids])), lables)

    if os.path.exists(config.datasetpkl):
        dataset = pkl.load(open(config.datasetpkl, 'rb'))
        train = dataset['train']
        dev = dataset['dev']
        test = dataset['test']
    else:
        train = load_dataset(config.train_path)
        dev = load_dataset(config.dev_path)
        test = load_dataset(config.test_path)
        dataset = {}
        dataset['train'] = train
        dataset['dev'] = dev
        dataset['test'] = test
        pkl.dump(dataset, open(config.datasetpkl, 'wb'))
    return train, dev, test

def build_net_data(dataset, config):
    data_x = dataset[0]
    label_y = dataset[1]
    label_y = tf.keras.utils.to_categorical(label_y, num_classes=config.num_classes)
    return data_x, label_y


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def convert_content_to_inputs(content, config):
    tokenized_outputs = config.tokenizer.encode_plus(content,add_special_tokens=True,
                                                     max_length=config.max_len,
                                                     truncation_strategy='longest_first',
                                                     pad_to_max_length=True)
    input_id = tokenized_outputs['input_ids']
    token_type_id = tokenized_outputs['token_type_ids']
    attention_mask = tokenized_outputs['attention_mask']

    return input_id, token_type_id, attention_mask