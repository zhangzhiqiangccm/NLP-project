
import pandas as pd
import numpy as np
import collections
import utils

#TODO 查看数据标签分布状况
utils.get_label_distribution(relation_file_path="./relation2id.txt",data_file_path="./train.txt")

'''
标签的分布状况
relation_id    numbers
        0     78642
        2     36889
        1     28864
        5     14030
        4     12223
        3      8385
        6      8221
        8      3259
        10     2740
        11     2709
        7      2036
        9      2002
        dtype: int64
'''

relation2id = {}

with open('relation2id.txt',"r",encoding="utf-8") as fr:
    for line in fr.readlines():
        line = line.strip().split(" ")
        relation2id[line[0]] = int(line[1])

datas = []
labels = []
positionE1 = []
positionE2 = []
#TODO count为了使抽取出的样本类别均匀
count = [0,0,0,0,0,0,0,0,0,0,0,0]
total_data=0
with open('train.txt','r',encoding='utf-8') as fr:
    for lines in fr:
        line = lines.split("\t")
        # line[2] : relation , line[3] : text
        if count[relation2id[line[2]]] < 1400:
            sentence = []
            index1 = line[3].index(line[0]) # 在 text中找出 实体1所在位置
            position1 = []
            index2 = line[3].index(line[1]) # 在 text 中找出 实体2所在位置
            position2 = []

            for i,word in enumerate(line[3]):
                sentence.append(word) # 以word的形式记录每个字,作为一个sentence
                position1.append(i-index1) # 从0位置开始记录每个word和 实体1的距离
                position2.append(i-index2) # 从0位置开始记录每个word和 实体2的距离
            datas.append(sentence)
            labels.append(relation2id[line[2]])
            positionE1.append(position1)
            positionE2.append(position2)
        count[relation2id[line[2]]] += 1
        total_data+=1
print(len(datas),len(positionE1),len(positionE2))


all_words = list(utils.flat_gen(datas)) # 转换为一行的列表,word
# print(all_words)
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts() # 统计不同word个数

set_words = sr_allwords.index # 不同word的词频排序后的词列表（默认降序）
set_ids = range(1, len(set_words)+1)
word2id = pd.Series(set_ids, index=set_words) # 构建word到索引的映射
id2word = pd.Series(set_words, index=set_ids) # 构建索引到word的映射

word2id["BLANK"] = len(word2id)+1 # 加入补全符号和未知符号，用于表示未知word
word2id["UNKNOW"] = len(word2id)+1
id2word[len(id2word)+1]="BLANK"
id2word[len(id2word)+1]="UNKNOW"

max_len = 50
def X_padding(sentence):
    """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
    ids = []
    for word in sentence:
        if word in word2id:
            ids.append(word2id[word])
        else:
            ids.append(word2id["UNKNOW"]) # 如果是未知word 则填充 unknow
    if len(ids) >= max_len: #如果该sentence过长，则截取0 - max_len-1
        return ids[:max_len]
    ids.extend([word2id["BLANK"]]*(max_len-len(ids))) # 如果sentence小于max_len，则填充BLANK
    return ids # 返回一个用id 表示每个word的sentence的表示

# 把各word距离实体词的距离标准化到0 到 80之间
def pos(num):
    if num < -40:
        return 0
    if num >= -40 and num <= 40:
        return num + 40
    if num > 40:
        return 80

def position_padding(words):
    words = [pos(i) for i in words]
    if len(words) >= max_len:  
        return words[:max_len]
    words.extend([81]*(max_len-len(words))) 
    return words


df_data = pd.DataFrame({'words': datas, 'tags': labels,'positionE1':positionE1,'positionE2':positionE2}, index=range(len(datas)))
df_data['words'] = df_data['words'].apply(X_padding)
df_data['tags'] = df_data['tags']
df_data['positionE1'] = df_data['positionE1'].apply(position_padding)
df_data['positionE2'] = df_data['positionE2'].apply(position_padding)

# 注意需要在外面加上list才能转换为ndarray
datas = np.asarray(list(df_data['words'].values))
labels = np.asarray(df_data['tags'].values)
positionE1 = np.asarray(list(df_data['positionE1'].values))
positionE2 = np.asarray(list(df_data['positionE2'].values))

# 把模型序列号
import pickle
with open('../people_relation_train.pkl', 'wb') as fw:
    pickle.dump(word2id, fw)
    pickle.dump(id2word, fw)
    pickle.dump(relation2id, fw)
    pickle.dump(datas, fw)
    pickle.dump(labels, fw)
    pickle.dump(positionE1, fw)
    pickle.dump(positionE2, fw)
print ('** Finished saving the train data.')

datas = []
labels = []
positionE1 = []
positionE2 = []
count = [0,0,0,0,0,0,0,0,0,0,0,0]
with open('train.txt','r',encoding='utf-8') as fr:
    for lines in fr:
        line = lines.split()
        if count[relation2id[line[2]]] > 1400 and count[relation2id[line[2]]] <= 1700:
            sentence = []
            index1 = line[3].index(line[0])
            position1 = []
            index2 = line[3].index(line[1])
            position2 = []

            for i,word in enumerate(line[3]):
                sentence.append(word)
                position1.append(i-3-index1)
                position2.append(i-3-index2)
            datas.append(sentence)
            labels.append(relation2id[line[2]])
            positionE1.append(position1)
            positionE2.append(position2)
        count[relation2id[line[2]]] += 1
        

df_data = pd.DataFrame({'words': datas, 'tags': labels,'positionE1':positionE1,'positionE2':positionE2}, index=range(len(datas)))
df_data['words'] = df_data['words'].apply(X_padding)
df_data['tags'] = df_data['tags']
df_data['positionE1'] = df_data['positionE1'].apply(position_padding)
df_data['positionE2'] = df_data['positionE2'].apply(position_padding)

datas = np.asarray(list(df_data['words'].values))
labels = np.asarray(list(df_data['tags'].values))
positionE1 = np.asarray(list(df_data['positionE1'].values))
positionE2 = np.asarray(list(df_data['positionE2'].values))

print(len(datas),len(positionE1),len(positionE2))

import pickle
with open('../people_relation_validate.pkl', 'wb') as fw:
    pickle.dump(datas, fw)
    pickle.dump(labels, fw)
    pickle.dump(positionE1, fw)
    pickle.dump(positionE2, fw)
print('** Finished saving the validate data.')


# TODO 测试数据集
datas = []
labels = []
positionE1 = []
positionE2 = []
count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
with open('train.txt', 'r', encoding='utf-8') as fr:
    for lines in fr:
        line = lines.split()
        if count[relation2id[line[2]]] > 1700 and count[relation2id[line[2]]] <= 2000:
            sentence = []
            index1 = line[3].index(line[0])
            position1 = []
            index2 = line[3].index(line[1])
            position2 = []

            for i, word in enumerate(line[3]):
                sentence.append(word)
                position1.append(i - 3 - index1)
                position2.append(i - 3 - index2)
            datas.append(sentence)
            labels.append(relation2id[line[2]])
            positionE1.append(position1)
            positionE2.append(position2)
        count[relation2id[line[2]]] += 1

df_data = pd.DataFrame({'words': datas, 'tags': labels, 'positionE1': positionE1, 'positionE2': positionE2},
                       index=range(len(datas )))
df_data['words'] = df_data['words'].apply(X_padding)
df_data['tags'] = df_data['tags']
df_data['positionE1'] = df_data['positionE1'].apply(position_padding)
df_data['positionE2'] = df_data['positionE2'].apply(position_padding)

datas = np.asarray(list(df_data['words'].values))
labels = np.asarray(list(df_data['tags'].values))
positionE1 = np.asarray(list(df_data['positionE1'].values))
positionE2 = np.asarray(list(df_data['positionE2'].values))

print(len(datas), len(positionE1), len(positionE2))

import pickle

with open('../people_relation_test.pkl', 'wb') as fw:
    pickle.dump(datas, fw)
    pickle.dump(labels, fw)
    pickle.dump(positionE1, fw)
    pickle.dump(positionE2, fw)
print('** Finished saving the test data.')

