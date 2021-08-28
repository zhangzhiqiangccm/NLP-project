# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 07:43:42 2020

@author: ljh
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

import collections
from collections import Counter
import numpy as np
import random
import jieba

from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.sans-serif']=['SimHei']#用来正常显示中文标签  
mpl.rcParams['font.family'] = 'STSong'
mpl.rcParams['font.size'] = 20


 #指定设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


training_file = '人体阴阳与电能.txt '

#中文字
def get_ch_lable(txt_file):  
    labels= ""
    with open(txt_file, 'rb') as f:
        for label in f: 
            labels =labels+label.decode('gb2312')
    return  labels
   
#分词
def fenci(training_data):
    seg_list = jieba.cut(training_data)  # 默认是精确模式  
    training_ci = " ".join(seg_list)
    training_ci = training_ci.split()
    #以空格将字符串分开
    training_ci = np.array(training_ci)
    training_ci = np.reshape(training_ci, [-1, ])
    return training_ci

def build_dataset(words, n_words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  
  return data, count, dictionary, reversed_dictionary

training_data =get_ch_lable(training_file)
print("总字数",len(training_data))
training_ci =fenci(training_data)
print("总词数",len(training_ci))    
training_label, count, dictionary, words = build_dataset(training_ci, 350)

#计算词频
word_count = np.array([freq for _,freq in count], dtype=np.float32)
word_freq = word_count / np.sum(word_count)#计算每个词的词频
word_freq = word_freq ** (3. / 4.)#词频变换
words_size = len(dictionary)
print("字典词数",words_size) 
print('Sample data', training_label[:10], [words[i] for i in training_label[:10]])


C = 3 
num_sampled = 64  # 负采样个数   
BATCH_SIZE = 12 
EMBEDDING_SIZE = 128


class SkipGramDataset(Dataset):
    def __init__(self, training_label, word_to_idx, idx_to_word, word_freqs):
        super(SkipGramDataset, self).__init__()
        self.text_encoded = torch.Tensor(training_label).long()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        idx = min( max(idx,C),len(self.text_encoded)-2-C)#防止越界
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx-C, idx)) + list(range(idx+1, idx+1+C))
        pos_words = self.text_encoded[pos_indices] 
        #多项式分布采样，取出指定个数的高频词
        neg_words = torch.multinomial(self.word_freqs, num_sampled+2*C, False)#True)
        #去掉正向标签
        neg_words = torch.Tensor(np.setdiff1d(neg_words.numpy(),pos_words.numpy())[:num_sampled]).long()
        return center_word, pos_words, neg_words


print('制作数据集...')
train_dataset = SkipGramDataset(training_label, dictionary, words, word_freq)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,drop_last=True, shuffle=True)

#测试数据集
#one = train_dataset[0][0].numpy()
#print('Sample data', training_label[:10], [words[i] for i in training_label[:10]])
#print(one,words[np.long(one)],[words[i] for i in train_dataset[0][1].numpy()])


sample = iter(dataloader)					#将数据集转化成迭代器
center_word, pos_words, neg_words = sample.next()				#从迭代器中取出一批次样本
print(center_word[0],words[np.long(center_word[0])],[words[i] for i in pos_words[0].numpy()])




class Model(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5 / self.embed_size
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        input_embedding = self.in_embed(input_labels)
                
        pos_embedding = self.in_embed(pos_labels)
        neg_embedding = self.in_embed(neg_labels)
        
        log_pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze()
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze()

        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)
        loss = log_pos + log_neg
        return -loss



model = Model(words_size, EMBEDDING_SIZE).to(device)
model.train()

valid_size = 16        
valid_window = words_size/2  # 取样数据的分布范围.
valid_examples = np.random.choice(int(valid_window), valid_size, replace=False)#0- words_size/2,中的数取16个。不能重复。

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
NUM_EPOCHS = 200
for e in range(NUM_EPOCHS):
    for ei, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        input_labels = input_labels.to(device)
        pos_labels = pos_labels.to(device)
        neg_labels = neg_labels.to(device)

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        if ei % 20 == 0:
            print("epoch: {}, iter: {}, loss: {}".format(e, ei, loss.item()))
    if e %40 == 0:           
        norm = torch.sum(model.in_embed.weight.data.pow(2),-1).sqrt().unsqueeze(1)
        normalized_embeddings = model.in_embed.weight.data / norm
        valid_embeddings = normalized_embeddings[valid_examples]
        
        similarity = torch.mm(valid_embeddings, normalized_embeddings.T)
        for i in range(valid_size):
            valid_word = words[valid_examples[i]]
            top_k = 8  # 取最近的排名前8的词
            nearest = (-similarity[i, :]).argsort()[1:top_k + 1]  #argsort函数返回的是数组值从小到大的索引值
            log_str = 'Nearest to %s:' % valid_word  
            for k in range(top_k):
                close_word = words[nearest[k].cpu().item()]
                log_str = '%s,%s' % (log_str, close_word)
            print(log_str)
    
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,xy=(x, y),xytext=(5, 2), textcoords='offset points',
                     ha='right',va='bottom')
    plt.savefig(filename)


final_embeddings = model.in_embed.weight.data.cpu().numpy()
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 200#输出100个词
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [words[i] for i in range(plot_only)]
  
plot_with_labels(low_dim_embs, labels)


 

 

 

 
