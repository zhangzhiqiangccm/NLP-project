# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 07:10:37 2020

@author: ljh
"""

#没标点

import re
import pickle
import torch
from tqdm import tqdm

from code_12_BERT_PROPN import (device,df_test,df_train_val,
                                getmodel,insert_tag,tokenize)

def clean_and_replace_target_name(row):  #去掉标点符号
    text = row['TextClean']
    text = re.sub("[^a-zA-Z]"," ",text)  #只保留英文字符，去掉标点及数字
    A = re.sub("[^a-zA-Z]"," ",row['A']) #只保留英文字符  
    B = re.sub("[^a-zA-Z]"," ",row['B']) #只保留英文字符

    # replace names  # 先分词，再取第一个，Dehner--》 ['de', '##hner']--》de  确保不被分成2个词
    text = re.sub(str(A), tokenizer.tokenize(A)[0], text) #将名称之换做一个词Bob Suter--》bob
    text = re.sub(str(B), tokenizer.tokenize(B)[0], text)
    
    text = re.sub(r"THISISA", r"[THISISA]", text)
    text = re.sub(r"THISISB", r"[THISISB]", text)
    text = re.sub(r"THISISP", r"[THISISP]", text)
    
    text = re.sub(' +', ' ', text)  #去掉多个空格
    return text


def savepkl(df,prename=''):
    offsets_lst = []
    tokens_lst = []
    max_len=269 #设置处理文本的最大长度
    bert_prediction = []
    for _, row in tqdm(df.iterrows(),total=len(df)):

        row.loc['TextClean']  = insert_tag(row,hasbrack= False)#插入标签,防止去标点时，一起被去掉
        text = clean_and_replace_target_name(row)#去除标点、空格，并压缩被指带的名词

        encode_rel= tokenizer.encode_plus(text,max_length=max_len,pad_to_max_length=True)#向量化  len=90

        tokens, offsets ,masks= tokenize(encode_rel['input_ids'] , 
                                         tokenizer,encode_rel['attention_mask'])#获取标签偏移
        offsets_lst.append(offsets)
        tokens_lst.append(tokens)
    #验证代词位置    
    #    print( tokenizer.decode(tokens),len(tokens)) 
    #    print( tokenizer.decode(np.asarray(tokens)[list(offsets)]))     
        token_tensor = torch.LongTensor([tokens]).to(device)
        masks_tensor = torch.LongTensor([masks]).to(device)
        #输入BERT模型
        bert_outputs,bert_last_outputs=  model(token_tensor,attention_mask =masks_tensor)  #[1, 107, 768] , [1, 768]
        bert_prediction.append(bert_outputs.cpu().numpy())#([1, 266, 768])
        
    pickle.dump(offsets_lst, open(prename+'offsets_NoPUNC.pkl', "wb"))
    pickle.dump(tokens_lst, open(prename+'tokens_NoPUNC_padding.pkl', "wb"))
    pickle.dump(bert_prediction, open(prename+'bert_outputs_forNoPUNC.pkl', "wb"))

if __name__ == '__main__':
    
    tokenizer,model = getmodel()
    model.to(device)
    torch.set_grad_enabled(False)
       
    savepkl(df_test, 'test_')    
    savepkl(df_train_val, ) 