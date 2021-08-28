# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch从深度学习到图神经网络>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Tue Jan 21 00:24:15 2020
"""

import torch
import torch.nn.functional as F
import time
import random
import numpy as np
from collections import Counter


RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


training_file = 'wordstest.txt'


#中文多文件
def readalltxt(txt_files):
    labels = []
    for txt_file in txt_files:
        
        target = get_ch_lable(txt_file)
        labels.append(target)  
    return labels
    
#中文字
def get_ch_lable(txt_file):  
    labels= ""
    with open(txt_file, 'rb') as f:
        for label in f: 
            #labels =label.decode('utf-8')
            labels =labels+label.decode('gb2312')
           
    return  labels
    


#优先转文件里的字符到向量
def get_ch_lable_v(txt_file,word_num_map,txt_label=None):
      
    words_size = len(word_num_map)   
    to_num = lambda word: word_num_map.get(word, words_size) 
    if txt_file!= None:
        txt_label = get_ch_lable(txt_file)

    labels_vector = list(map(to_num, txt_label)) 
    return labels_vector  
    
training_data =get_ch_lable(training_file)

print("Loaded training data...")

print('样本长度:',len(training_data))
counter = Counter(training_data)  
words = sorted(counter)
words_size= len(words)
word_num_map = dict(zip(words, range(words_size))) 

print('字表大小:', words_size)     
wordlabel = get_ch_lable_v(training_file,word_num_map)

class GRURNN(torch.nn.Module):
    def __init__(self, word_size, embed_dim,
                 hidden_dim, output_size, num_layers):
        super(GRURNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.embed = torch.nn.Embedding(word_size, embed_dim)
        self.gru = torch.nn.GRU(input_size=embed_dim,
                                hidden_size=hidden_dim,
                                num_layers=num_layers,bidirectional=True)
        self.fc = torch.nn.Linear(hidden_dim*2, output_size)

    
    def forward(self, features, hidden):
        embedded = self.embed(features.view(1, -1))
        output, hidden = self.gru(embedded.view(1, 1, -1), hidden)
        output = self.fc(output.view(1, -1))
        return output, hidden
      
    def init_zero_state(self):
        init_hidden = torch.zeros(self.num_layers*2, 1, self.hidden_dim).to(DEVICE)
        return init_hidden



EMBEDDING_DIM = 10
HIDDEN_DIM = 20
NUM_LAYERS = 1

model = GRURNN(words_size, EMBEDDING_DIM, HIDDEN_DIM, words_size, NUM_LAYERS)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def evaluate(model, prime_str, predict_len, temperature=0.8):

    hidden = model.init_zero_state().to(DEVICE)
    predicted = ''

    #处理输入语义
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_str[p], hidden)
        predicted +=words[prime_str[p]]
    inp = prime_str[-1]
    predicted +=words[inp]
    
    for p in range(predict_len):
        output, hidden = model(inp, hidden)
        
        #从多项式分布中采样
        output_dist = output.data.view(-1).div(temperature).exp()
        inp = torch.multinomial(output_dist, 1)[0]
        
        predicted += words[inp]

    return predicted


#定义参数训练模型
training_iters = 5000
display_step = 1000
n_input = 4
step = 0
offset = random.randint(0,n_input+1)
end_offset = n_input + 1

while step < training_iters:
    start_time = time.time()

    # 随机取一个位置偏移
    if offset > (len(training_data)-end_offset):
        offset = random.randint(0, n_input+1)
   

    inwords =wordlabel[offset:offset+n_input]
    inwords = np.reshape(np.array(inwords), [n_input, -1,  1])

    out_onehot = wordlabel[offset+1:offset+n_input+1]


    hidden = model.init_zero_state()
    optimizer.zero_grad()
    
    loss = 0.
    inputs, targets = torch.LongTensor(inwords).to(DEVICE), torch.LongTensor(out_onehot).to(DEVICE)
    for c in range(n_input):

        outputs, hidden = model(inputs[c], hidden)
        loss += F.cross_entropy(outputs, targets[c].view(1))

    loss /= n_input
    loss.backward()

    optimizer.step()

    #输出日志
    with torch.set_grad_enabled(False):
        if (step+1) % display_step == 0:
            print(f'Time elapsed: {(time.time() - start_time)/60:.4f} min')
            print(f'step {step+1} | Loss {loss.item():.2f}\n\n')
            with torch.no_grad():
                print(evaluate(model, inputs, 32), '\n')
            print(50*'=')

    step += 1
    offset += (n_input+1)#中间隔了一个，作为预测

print("Finished!")

while True:
    prompt = "请输入几个字，最好是%s个: " % n_input
    sentence = input(prompt)
    inputword = sentence.strip()
    
    try:
        inputword = get_ch_lable_v(None,word_num_map,inputword)
        keys = np.reshape(np.array(inputword), [ len(inputword),-1, 1])
        model.eval()
        with torch.no_grad():
            sentence =evaluate(model, torch.LongTensor(keys).to(DEVICE), 32)

        print(sentence)
    except:
        print("该字我还没学会")









