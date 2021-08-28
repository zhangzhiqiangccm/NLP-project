# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:21:46 2020

@author: ljh
"""
import os
import torch

from transformers import (
        get_linear_schedule_with_warmup,BertTokenizer,
        AdamW,
        AutoModelForSequenceClassification,
        AutoConfig
        )

from torch.utils.data import DataLoader,dataset
import time
import numpy as np
from sklearn import metrics
from datetime import timedelta
    
data_dir='./THUCNews/data'
def read_file(path):
    with open(path, 'r', encoding="UTF-8") as file:
        docus = file.readlines()
        newDocus = []
        for data in docus:
            newDocus.append(data)
    return newDocus


#建立数据集 
class Label_Dataset(dataset.Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):#返回数据长度
        return len(self.data)
    def __getitem__(self,ind):
        onetext = self.data[ind]
        content, label = onetext.split('\t')
        label = torch.LongTensor([int(label)])
        return content,label

trainContent = read_file(os.path.join(data_dir, "train.txt")) 
testContent = read_file(os.path.join(data_dir, "test.txt"))

traindataset =Label_Dataset( trainContent )
testdataset =Label_Dataset( testContent )

testdataloder = DataLoader(testdataset, batch_size=1, shuffle = False)
batch_size = 8
traindataloder = DataLoader(traindataset, batch_size=batch_size, shuffle = True)

class_list = [x.strip() for x in open(
        os.path.join(data_dir, "class.txt")).readlines()]


pretrained_weights = 'bert-base-chinese'#建立模型
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
config = AutoConfig.from_pretrained(pretrained_weights,num_labels=len(class_list)) 
#单独指定config，在config中指定分类个数
nlp_classif = AutoModelForSequenceClassification.from_pretrained(pretrained_weights,
                                                           config=config)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nlp_classif = nlp_classif.to(device)

time_start = time.time() #开始时间

epochs = 2
gradient_accumulation_steps = 1
max_grad_norm =0.1  #梯度剪辑的阀值

require_improvement = 1000                 # 若超过1000batch效果还没提升，则提前结束训练
savedir = './myfinetun-bert_chinese/'
os.makedirs(savedir, exist_ok=True)
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train( model, traindataloder, testdataloder):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)

    
    scheduler = get_linear_schedule_with_warmup(optimizer,
                num_warmup_steps=0, num_training_steps=len(traindataloder) * epochs)


    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    
    for epoch in range(epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, epochs))
        for i, (sku_name, labels) in enumerate(traindataloder):
            model.train()
            
            ids = tokenizer.batch_encode_plus( sku_name,
#                max_length=model.config.max_position_embeddings,  #模型的配置文件中就是512，当有超过这个长度的会报错
                pad_to_max_length=True,return_tensors='pt')#没有return_tensors会返回list！！！！
               
            labels = labels.squeeze().to(device) 
            outputs = model(ids["input_ids"].to(device), labels=labels,
                            attention_mask =ids["attention_mask"].to(device)  )
            
            loss, logits = outputs[:2]
            
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            loss.backward()
            
            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                truelabel = labels.data.cpu()
                predic = torch.argmax(logits,axis=1).data.cpu()
#                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(truelabel, predic)
                dev_acc, dev_loss = evaluate( model, testdataloder)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    model.save_pretrained(savedir)                    
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

def evaluate(model, testdataloder):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for sku_name, labels in testdataloder:
            ids = tokenizer.batch_encode_plus( sku_name,
#                max_length=model.config.max_position_embeddings,  #模型的配置文件中就是512，当有超过这个长度的会报错
                pad_to_max_length=True,return_tensors='pt')#没有return_tensors会返回list！！！！
               
            labels = labels.squeeze().to(device) 
            outputs = model(ids["input_ids"].to(device), labels=labels, 
                                   attention_mask =ids["attention_mask"].to(device) )
            
            loss, logits = outputs[:2]
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.argmax(logits,axis=1).data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, loss_total / len(testdataloder)


train( nlp_classif, traindataloder, testdataloder)    
    
    