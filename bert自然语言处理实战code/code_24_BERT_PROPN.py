# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 07:10:37 2020

@author: ljh
"""

#提取代词特征

import pandas as pd 
import pickle
import torch
from tqdm import tqdm
from transformers import BertTokenizer,BertModel,BertConfig

#指定设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
    
#读取数据    
df_test = pd.read_csv("gap-development.tsv", delimiter="\t")
df_train_val = pd.concat([
    pd.read_csv("gap-test.tsv", delimiter="\t"),
    pd.read_csv("gap-validation.tsv", delimiter="\t")
], axis=0)    


def getmodel():
    #加载词表文件tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    #添加特殊词
    special_tokens_dict = {'additional_special_tokens': ["[THISISA]","[THISISB]","[THISISP]"]}
    tokenizer.add_special_tokens(special_tokens_dict)	#添加特殊词
    print(tokenizer.additional_special_tokens,tokenizer.additional_special_tokens_ids)

    
    model = BertModel.from_pretrained('bert-base-uncased')#加载模型
    return tokenizer,model


    

############################


def insert_tag(row,hasbrack=True):#按照插入的位置，从大到小排序[(383, ' THISISP '), (366, ' THISISB '), (352, ' THISISA ')]
    orgtag=[" [THISISA] "," [THISISB] "," [THISISP] "]
    if hasbrack==False:
        orgtag=[" THISISA "," THISISB "," THISISP "]
        
    to_be_inserted = sorted([
        (row["A-offset"], orgtag[0]),
        (row["B-offset"], orgtag[1]),
        (row["Pronoun-offset"], orgtag[2])], key=lambda x: x[0], reverse=True)
    
    text = row["Text"]#len 443 
    for offset, tag in to_be_inserted:#先插最后的，不会影响前面
        text = text[:offset] + tag + text[offset:]#（插到每个代词的前面）
    return text#len 470 (443+3*9)



def tokenize(sequence_ind, tokenizer,sequence_mask= None):#将标签分离，并返回标签偏移位置
    entries = {}
    final_tokens=[]
    final_mask=[]

    for i,one in enumerate(sequence_ind):
        if one in tokenizer.additional_special_tokens_ids:
            tokenstr = tokenizer.convert_ids_to_tokens(one)
            entries[tokenstr] = len(final_tokens)
            continue
        final_tokens.append(one)
        if sequence_mask is not None:
            final_mask.append(sequence_mask[i])
    return  final_tokens, (entries["[THISISA]"], entries["[THISISB]"], entries["[THISISP]"]) ,final_mask   



def savepkl(df,name):
    bert_prediction = []    
    for _, row in tqdm(df.iterrows(),total=len(df)):    
        #循环内部
        text = insert_tag(row)#插入标签
        sequence_ind = tokenizer.encode(text)#向量化
        tokens, offsets,_ = tokenize(sequence_ind, tokenizer)#获取标签偏移        
        token_tensor = torch.LongTensor([tokens]).to(device)
        bert_outputs,bert_last_outputs=  model(token_tensor)  #[1, 107, 768] , [1, 768]            
        extracted_outputs = bert_outputs[:,offsets,:]#根据偏移位置抽取特征向量
        bert_prediction.append(extracted_outputs.cpu().numpy())    
    pickle.dump(bert_prediction, open(name, "wb"))


if __name__ == '__main__':
    
    tokenizer,model = getmodel()
    model.to(device)
    torch.set_grad_enabled(False)
    
    savepkl(df_test, 'test_bert_outputs_forPROPN.pkl')    
    savepkl(df_train_val, 'bert_outputs_forPROPN.pkl')   

