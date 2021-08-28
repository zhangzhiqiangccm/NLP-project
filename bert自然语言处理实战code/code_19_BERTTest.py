# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:43:59 2020

@author: ljh
"""


import torch
from transformers import BertTokenizer, BertForMaskedLM

#加载预训练模型 tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#输入文本
text = "[CLS] Who is Li Jinhong ? [SEP] Li Jinhong is a programmer [SEP]"
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)

masked_index = 8 #掩码一个标记，用' BertForMaskedLM '预测回来
tokenized_text[masked_index] = '[MASK]'
print(tokenized_text)

# 将标记转换为词汇表索引
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# 将输入转换为PyTorch张量
tokens_tensor = torch.tensor([indexed_tokens])


#指定设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 加载预训练模型 (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
model.to(device)


segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
segments_tensors = torch.tensor([segments_ids]).to(device)

tokens_tensor = tokens_tensor.to(device)
# 预测所有的tokens
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    
predictions = outputs[0]  #[1, 15, 30522]

predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0] #转成单词
print('Predicted token is:',predicted_token)














