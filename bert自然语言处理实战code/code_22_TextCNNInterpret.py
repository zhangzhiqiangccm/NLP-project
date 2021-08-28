# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:13:29 2019

@author: ljh
"""



import spacy  #引入分词库 
import torch#引入PyTorch库
import torch.nn.functional as F
#引入解释库
from captum.attr import (IntegratedGradients,TokenReferenceBase,visualization,
                         configure_interpretable_embedding_layer, remove_interpretable_embedding_layer)

#引入本地代码库
from code_14_TextCNN import TextCNN, TEXT,LABEL

class TextCNNInterpret(TextCNN):#定义TextCNN的子类
    def __init__(self, *args,**kwargs):#透传参数
        super().__init__(*args,**kwargs)
    def forward(self, text): #重载模型处理方法        
        embedded = self.embedding(text)#从词嵌入开始处理
        #后面的代码与TextCNN一样
        embedded = embedded.unsqueeze(1) 
        conved = [self.mish(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]        
        cat = self.dropout(torch.cat(pooled, dim = 1))
        return self.fc(cat)

##########################
#定义模型参数
INPUT_DIM = len(TEXT.vocab)#25002
EMBEDDING_DIM = TEXT.vocab.vectors.size()[1] #100 
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
#实例化模型
model = TextCNNInterpret(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

#加载模型权重
model.load_state_dict(torch.load('textcnn-model.pt') )
print('Vocabulary Size: ', len(TEXT.vocab))
#对嵌入层进行封装并提取
interpretable_embedding  =  configure_interpretable_embedding_layer (model, 'embedding')

##########################



ig = IntegratedGradients(model)#创建梯度积分算法对象

#定义列表，存放可视化记录
vis_data_records_ig = []

nlp = spacy.load('en') #为分词库加载英文语言包


#定义函数对句子进行可解释性分析
def interpret_sentence(model, sentence, min_len = 7, label = 0):
    
    sentence=sentence.lower() #将句子转为小写

    model.eval()
    #分词处理
    text = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(text) < min_len: #对小于指定长度的句子进行 填充
        text += [TEXT.pad_token] * (min_len - len(text))
    #将句子中的单词转为索引
    indexed = [TEXT.vocab.stoi[t] for t in text]
    
    model.zero_grad() #将模型中的梯度清0
    
    input_indices = torch.LongTensor(indexed) #转为张量
    input_indices = input_indices.unsqueeze(0) #增加维度

    #转为词嵌入
    input_embedding = interpretable_embedding.indices_to_embeddings(input_indices)

    #将词嵌入输入模型，进行预测
    pred = torch.sigmoid(model(input_embedding)).item()
    pred_ind = round(pred) #计算输出结果
    
    #创建梯度积分的初始输入值
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] #获得填充字符的索引
    token_reference = TokenReferenceBase(reference_token_idx=PAD_IDX)
    #制作初始输入索引：复制指定长度个token_reference，并扩展维度
    reference_indices = token_reference.generate_reference(len(indexed), device='cpu').unsqueeze(0)
    print("reference_indices",reference_indices)
    #将制作好的输入索引转成词嵌入
    reference_embedding = interpretable_embedding.indices_to_embeddings(reference_indices)


    #用梯度积分的方法计算可解释性
    attributions_ig, delta = ig.attribute(input_embedding, reference_embedding, n_steps=500, return_convergence_delta=True)
    #输出可解释性结果
    print('attributions_ig, delta',attributions_ig.size(), delta.size())
    print('pred: ', LABEL.vocab.itos[pred_ind], '(', '%.2f'%pred, ')', ', delta: ', abs(delta))
    #加入可视化记录中
    add_attributions_to_visualizer(attributions_ig, text, pred, pred_ind, label, delta, vis_data_records_ig)

#定义函数，将解释性结果放入可视化记录中    
def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, delta, vis_data_records):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.detach().numpy()

    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                            attributions,
                            pred,
                            LABEL.vocab.itos[pred_ind],
                            LABEL.vocab.itos[label],
                            LABEL.vocab.itos[1],
                            attributions.sum(),       
                            text[:len(attributions)],
                            delta))

interpret_sentence(model, 'It was a fantastic performance !', label=1)

interpret_sentence(model, 'The film is very good！', label=1)

interpret_sentence(model, 'I think this film is not very bad！', label=1)


#根据可视化记录生成网页
visualization.visualize_text(vis_data_records_ig)

#还原模型的词嵌入层
remove_interpretable_embedding_layer(model, interpretable_embedding)





 










