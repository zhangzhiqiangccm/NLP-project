"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <深度学习之TensorFlow机器视觉算法与应用>配套代码 （700+页）
@配套代码技术支持：bbs.aianaconda.com      (有问必答)
Created on Tue Mar 19 22:24:58 2019
"""

import random #引入基础库
import time

import torch#引入PyTorch库
import torch.nn as nn
import torch.nn.functional as F

from torchtext import data ,datasets,vocab #引入文本处理库
import spacy 

torch.manual_seed(1234)                    #固定随机种子
torch.backends.cudnn.deterministic = True  #固定GPU运算方式

torch.backends.cudnn.benchmark = False


############数据集的制作
in_times = 64
#定义字段，并按照指定标记化函数进行分词， 
TEXT = data.Field(tokenize = 'spacy',lower=True,fix_length = in_times) 
LABEL = data.LabelField(dtype = torch.float)

#加载数据集，并根据IMDB两个文件夹，返回两个数据集
train_data, test_data = datasets.IMDB.splits(text_field=TEXT, label_field=LABEL) 
print('---------输出一条数据------')
print(vars(train_data.examples[0]),len(train_data.examples))
print('---------------')

#将训练数据集再次拆分
train_data, valid_data = train_data.split(random_state = random.seed(1234))  
print("训练数据集: ", len(train_data),"条")
print("验证数据集: ", len(valid_data),"条")
print("测试数据集: ", len(test_data),"条")
###########
#建立词表
TEXT.build_vocab(train_data, 
                 max_size = 25000, #词的最大数量
                 vectors = "glove.6B.100d", 
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)

#创建批次数据
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device)

############

#for i in test_iterator:
#    print(i)
#    break
#
#i.label
#i.text.shape

#########################################################################################################################


class Mish(nn.Module):#Mish激活函数
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x
    
#定义AttLSTMModel模型类，该模型包括 2个RNN层和1个全连接层
class AttLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, in_times,
                 dropout, pad_idx,hidden_dim, n_layer, n_class):
        super(AttLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        #定义循环神经网络层
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, n_layer,batch_first=True)
        self.Linear = torch.nn.Linear(hidden_dim*in_times, n_class)#定义全连接层
        self.attention = AttentionSeq(hidden_dim,hard=0.03)
        self.last_norm = torch.nn.LayerNorm(hidden_dim)
        self.drop_p = dropout
        

        self.mish = Mish()
        
    def forward(self, text): #输入形状为[sent len, batch size]

        text = text.permute(1, 0)#将形状变为[batch size, sent len]
        embedded = self.embedding(text)#形状为[batch size, sent len, emb dim]
        t, _ = self.lstm(embedded)  #进行RNN处理
        t = self.last_norm(self.mish(t) ) 

        t = self.attention(t)
        t=t.reshape(t.shape[0],-1)
#        t = t[:, -1, :]      #获取RNN网络的最后一个序列数据
        t = nn.functional.dropout(t, p=self.drop_p, training=self.training)
        out = self.Linear(t)#进行全连接处理
        return out

        

class AttentionSeq(torch.nn.Module):

    def __init__(self, hidden_dim,hard= 0):
        super(AttentionSeq, self).__init__()
        self.hidden_dim = hidden_dim
        self.dense = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hard = hard

    def forward(self, features, mean=False):
        #[batch,seq,dim]
        batch_size, time_step, hidden_dim = features.size()
        weight = torch.nn.Tanh()(self.dense(features))

        # mask给负无穷使得权重为0
        mask_idx = torch.sign(torch.abs(features).sum(dim=-1))
#        mask_idx = mask_idx.unsqueeze(-1).expand(batch_size, time_step, hidden_dim)
        mask_idx = mask_idx.unsqueeze(-1).repeat(1, 1, hidden_dim)


        weight = torch.where(mask_idx== 1, weight,
                             torch.full_like(mask_idx,(-2 ** 32 + 1)))
        weight = weight.transpose(2, 1)
        weight = torch.nn.Softmax(dim=2)(weight)
        if self.hard!=0: #hard mode
            weight = torch.where(weight>self.hard, weight, torch.full_like(weight,0))
        
        if mean:
            weight = weight.mean(dim=1)
            weight = weight.unsqueeze(1)
            weight = weight.repeat(1, hidden_dim, 1)
        weight = weight.transpose(2, 1)
        features_attention = weight * features

        return features_attention
#################
        
if __name__ == '__main__':   

    INPUT_DIM = len(TEXT.vocab)#25002
    EMBEDDING_DIM = TEXT.vocab.vectors.size()[1] #100 

    OUTPUT_DIM = 1
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    
    model = AttLSTMModel(INPUT_DIM, EMBEDDING_DIM,in_times, DROPOUT, PAD_IDX,256,2,OUTPUT_DIM)

    ####################################       
    #复制词向量
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    
    #将填充的词向量清0
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    
    ####################################
    import torch.optim as optim
    from functools import partial
    from ranger import *
    opt_func = partial(Ranger,  betas=(.9,0.99), eps=1e-6)#betas=(Momentum,alpha)
    optimizer = opt_func(model.parameters(),lr=0.004)
    
    
    criterion = nn.BCEWithLogitsLoss()  # 带有sigmoid 2分类的cross entropy
    
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    
    def binary_accuracy(preds, y):#计算准确率
        rounded_preds = torch.round(torch.sigmoid(preds))#把概率的结果 四舍五入
        correct = (rounded_preds == y).float() # True False -> 转为 1， 0
        acc = correct.sum() / len(correct)
        return acc
    
    
    def train(model, iterator, optimizer, criterion):
        
        epoch_loss = 0
        epoch_acc = 0
        
        model.train()  #设置模型标志  ，保证dropout在训练模式下
        
        for batch in iterator:
            optimizer.zero_grad()
            predictions = model(batch.text).squeeze(1)# 在第1个维度上 去除维度=1
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    
    
    def evaluate(model, iterator, criterion):
        
        epoch_loss = 0
        epoch_acc = 0
        
        model.eval()
        
        with torch.no_grad():
            for batch in iterator:
                predictions = model(batch.text).squeeze(1)
                loss = criterion(predictions, batch.label)
                acc = binary_accuracy(predictions, batch.label)
                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    
    
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    
    N_EPOCHS = 5
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
    
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
    
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:#保存最优模型
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'myLSTMNet-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    
    
    #测试模型效果
    model.load_state_dict(torch.load('myLSTMNet-model.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    
    ##################################################
    
    #使用接口
    
    nlp = spacy.load('en')
    
    def predict_sentiment(model, sentence, min_len = in_times):
        model.eval()
    #    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
        tokenized = nlp.tokenizer(sentence).text.split() #
        
        if len(tokenized) < min_len: #长度不足，在后面填充
            tokenized += ['<pad>'] * (min_len - len(tokenized))
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        prediction = torch.sigmoid(model(tensor))
        return prediction.item()
    
    sen = "This film is terrible"
    print('\n预测 sen = ', sen)
    print('预测 结果:', predict_sentiment(model,sen))
     
    sen = "This film is great"
    print('\n预测 sen = ', sen)
    print('预测 结果:', predict_sentiment(model,sen))
     
    sen = "I like this film very much！"
    print('\n预测 sen = ', sen)
    print('预测 结果:', predict_sentiment(model,sen))


