

import sys
import numpy as np
import tqdm
import time
import torch
import torch.nn as nn
import torch.utils.data as D
from BiLSTM_ATT import BiLSTM_ATT
import pickle
import utils
from params_config import config
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data():
    with open('./data/people_relation_train.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        relation2id = pickle.load(inp)
        train_data = pickle.load(inp)
        train_labels = pickle.load(inp)
        train_position1 = pickle.load(inp)
        train_position2 = pickle.load(inp)

    with open('./data/people_relation_validate.pkl', 'rb') as inp:
        valid_data = pickle.load(inp)
        valid_labels = pickle.load(inp)
        valid_position1 = pickle.load(inp)
        valid_position2 = pickle.load(inp)

    with open('./data/people_relation_test.pkl', 'rb') as inp:
        test_data = pickle.load(inp)
        test_labels = pickle.load(inp)
        test_position1 = pickle.load(inp)
        test_position2 = pickle.load(inp)

    def show_label_distribution(labels):
        df = pd.Series(labels)
        print(df.value_counts())

    def random_partition(range,feats,pos1,pos2,labels):
        indices_range = range
        random_indices = np.random.permutation(indices_range)
        feats = [feats[i] for i in random_indices]
        labels = [labels[i] for i in random_indices]
        pos1 = [pos1[i] for i in random_indices]
        pos2 = [pos2[i] for i in random_indices]
        return feats,pos1,pos2,labels
    batch_size = config['BATCH']
    #TODO 对训练集进行随机打乱，防止所取的小批量样本分布不均匀
    #TODO 注意这个是在所有的训练集上面打乱才有意义，而不是先取少量再打乱

    train_data,train_position1,train_position2,train_labels =\
        random_partition(len(train_data)-len(train_data)%batch_size,train_data,train_position1,train_position2,train_labels)

    #TODO 可以选一定批次的数据进行训练
    # num_of_batch = 5
    #TODO 训练集
    # num_train = num_of_batch * batch_size
    num_train = len(train_data)-len(train_data)%batch_size
    train_data = torch.LongTensor(train_data[:num_train])  # 凑成整的batch
    train_position1 = torch.LongTensor(train_position1[:num_train])
    train_position2 = torch.LongTensor(train_position2[:num_train])
    train_labels = torch.LongTensor(train_labels[:num_train])
    # df = pd.Series(labels[:num_train])
    # print(df.value_counts())
    train_datasets = D.TensorDataset(train_data, train_position1, train_position2, train_labels)
    train_iter = D.DataLoader(train_datasets, batch_size, shuffle=False, num_workers=1)
    #TODO 查看训练数据标签分布
    show_label_distribution(train_labels)

    valid_data, valid_position1, valid_position2, valid_labels = \
        random_partition(len(valid_data) - len(valid_data) % batch_size, valid_data, valid_position1, valid_position2,
                         valid_labels)
    #TODO 验证集
    num_validate = len(valid_data) - len(valid_data)%batch_size
    # num_validate = num_of_batch * batch_size
    valid_data = torch.LongTensor(valid_data[:num_validate])
    valid_labels = torch.LongTensor(valid_labels[:num_validate])
    valid_position1 = torch.LongTensor(valid_position1[:num_validate])
    valid_position2 = torch.LongTensor(valid_position2[:num_validate])
    valid_datasets = D.TensorDataset(valid_data, valid_position1, valid_position2, valid_labels)
    valid_iter = D.DataLoader(valid_datasets, batch_size, shuffle=False, num_workers=1)
    # TODO 查看验证集数据标签分布
    show_label_distribution(valid_labels)

    test_data, test_position1, test_position2, test_labels = \
        random_partition(len(test_data) - len(test_data) % batch_size, test_data, test_position1, test_position2,
                         test_labels)
    #TODO 测试集
    num_test = len(test_data)-len(test_data)%batch_size
    # num_test = num_of_batch * batch_size
    test_data = torch.LongTensor(test_data[:num_test])
    test_position1 = torch.LongTensor(test_position1[:num_test])
    test_position2 = torch.LongTensor(test_position2[:num_test])
    test_labels = torch.LongTensor(test_labels[:num_test])
    test_datasets = D.TensorDataset(test_data, test_position1, test_position2, test_labels)
    test_iter = D.DataLoader(test_datasets, batch_size, shuffle=False, num_workers=1)
    # TODO 查看测试数据标签分布
    show_label_distribution(valid_labels)
    return train_iter,test_iter,valid_iter,word2id,relation2id

def load_pre_embedding():
    pre_embedding = []
    word2vec = {}
    with open('vec.txt', 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            word2vec[line.split()[0]] = line.split()[1:]
    # TODO 前面word2id是从 1 开始编码的，留出的位置 0 用于作为 UNKNOW 符号
    # TODO 加载预训练词向量需要按照word2id的word顺序进行，这是模型里面 Embedding.from_pretrained 的要求
    unknow_pre = [1] * config['EMBEDDING_DIM']
    pre_embedding.append(unknow_pre)
    for word in word2id.keys():
        if word in word2vec.keys():
            pre_embedding.append(word2vec[word])
        else:
            pre_embedding.append(unknow_pre)
    pre_embedding = np.asarray(pre_embedding, dtype='float32')
    pre_embedding = torch.FloatTensor(pre_embedding)
    return pre_embedding

def train(model,data_iter,valid_iter,loss,lr,num_epochs,device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-5)
    tic = time.time()
    best_loss = torch.Tensor([100.])
    for epoch in range(1,num_epochs+1):
        l_sum,n = 0.0,0
        for sentence, pos1, pos2, y in data_iter:
            optimizer.zero_grad()
            y_hat = model(sentence, pos1, pos2)
            l = loss(y_hat, y.long())
            l.backward()
            n += y.numel()
            l_sum += l.item()
            optimizer.step()
        if epoch % config['PER_PRINT'] == 0:
            print("train loss : epoch {0:4d},loss {1:.3f}, time {2:.1f} sec".format(
                epoch, (l_sum / n), time.time() - tic))

        # TODO 每轮结束测试在验证集上的性能，保存最好的一个
        val_loss = validate(model,loss,valid_iter)
        print("Epoch {}, val Loss:{:.4f}".format(epoch, val_loss))
        if val_loss < best_loss:
            best_loss = val_loss
            file_name = "./model/model_best.pkl"
            torch.save(model, file_name)
    file_name = "./model/model_final.pkl"
    torch.save(model, file_name)

# 验证
def validate(model,loss,data_iter):
    model.eval()
    with torch.no_grad():
        l_sum, n = 0.0, 0
        for sentence,pos1,pos2,y in data_iter:
            y_hat = model(sentence,pos1,pos2)
            l = loss(y_hat,y.long())
            l_sum += l.item()
            n += y.numel()
    return l_sum/n


# 测试
def predict(model,data_iter):
    model.eval()
    with torch.no_grad():
        y_pred,y_test = [],[]
        for sentence,pos1,pos2,y_ in data_iter:
            y_hat = model(sentence,pos1,pos2)
            y_pred.append(y_hat.argmax(dim=1).numpy().tolist())
            y_test.append(y_.numpy().tolist())
    return y_test,y_pred


# 计算 F1,召回率，准确率，混淆矩阵
def evaluate(test_tag_lists,pred_tag_lists,remove_O=False):
    metrics = utils.Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

if __name__ == '__main__':
    pre_embedding = []
    train_iter, test_iter,valid_iter,word2id, relation2id = load_data()
    is_train = True # 默认进行训练模型
    #TODO  根据命令行参数决定是否训练模型，是否加载预训练词向量
    if len(sys.argv) == 3:
        is_train = True
        if sys.argv[1] == "pretrained":
            pre_embedding = load_pre_embedding()
            print("use pretrained embedding")
            config["pretrained"] = True
        else:
            pre_embedding = []
    elif len(sys.argv) == 2:
        if sys.argv[1] == "train":
            is_train = True
        elif sys.argv[1] == 'test':
            is_train = False

    if is_train:
        model = BiLSTM_ATT(input_size=len(word2id) + 1, output_size=len(relation2id), config=config,
                           pre_embedding=pre_embedding)
        criterion = nn.CrossEntropyLoss()
        train(model=model, data_iter=train_iter,valid_iter=valid_iter, loss=criterion, lr=config['LEARNING_RATE'],
              num_epochs=config['EPOCHS'], device='cpu')

        y_test, y_pred = predict(model, test_iter)
        evaluate(y_test, y_pred)
    else:
        model = utils.load_model("./model/model_final.pkl")
        y_test, y_pred = predict(model,test_iter)
        evaluate(y_test, y_pred)





