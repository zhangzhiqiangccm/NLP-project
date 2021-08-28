# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch从深度学习到图神经网络>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Fri Feb  1 00:07:25 2019
"""

import sklearn.datasets     #引入数据集
import torch
import numpy as np
import matplotlib.pyplot as plt
from code_02_moons_fun import LogicNet,moving_average,predict,plot_decision_boundary
import torch.nn as nn




np.random.seed(0)           #设置随机数种子
X, Y = sklearn.datasets.make_moons(40,noise=0.2) #生成2组半圆形数据

arg = np.squeeze(np.argwhere(Y==0),axis = 1)     #获取第1组数据索引
arg2 = np.squeeze(np.argwhere(Y==1),axis = 1)#获取第2组数据索引

plt.title("train moons data")
plt.scatter(X[arg,0], X[arg,1], s=100,c='b',marker='+',label='data1')
plt.scatter(X[arg2,0], X[arg2,1],s=40, c='r',marker='o',label='data2')
plt.legend()
plt.show()

#继承LogicNet类，构建网络模型
class Logic_BN_Net(LogicNet):
    def __init__(self,inputdim,hiddendim,outputdim):#初始化网络结构
        super(Logic_BN_Net,self).__init__(inputdim,hiddendim,outputdim)
        self.BN = nn.BatchNorm1d(hiddendim) #定义BN层
    def forward(self,x): #搭建用两层全连接组成的网络模型
        x = self.Linear1(x)#将输入数据传入第1层
        x = torch.tanh(x)#对第一层的结果进行非线性变换
        x = self.BN(x)#将第一层的数据做BN处理
        x = self.Linear2(x)#再将数据传入第2层
        return x




model = Logic_BN_Net(inputdim=2,hiddendim=500,outputdim=2)#初始化模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)#定义优化器


xt = torch.from_numpy(X).type(torch.FloatTensor)#将Numpy数据转化为张量
yt = torch.from_numpy(Y).type(torch.LongTensor)
epochs = 200#定义迭代次数
losses = []#定义列表，用于接收每一步的损失值
for i in range(epochs):
    loss = model.getloss(xt,yt)
    losses.append(loss.item())
    optimizer.zero_grad()#清空之前的梯度
    loss.backward()#反向传播损失值
    optimizer.step()#更新参数


avgloss= moving_average(losses) #获得损失值的移动平均值
plt.figure(1)
plt.subplot(211)
plt.plot(range(len(avgloss)), avgloss, 'b--')
plt.xlabel('step number')
plt.ylabel('Training loss')
plt.title('step number vs. Training loss')
plt.show()


plot_decision_boundary(lambda x : predict(model,x) ,X, Y)
from sklearn.metrics import accuracy_score
print("训练时的准确率：",accuracy_score(model.predict(xt),yt))

Xtest, Ytest = sklearn.datasets.make_moons(80,noise=0.2) #生成2组半圆形数据
plot_decision_boundary(lambda x : predict(model,x) ,Xtest, Ytest)
Xtest_t = torch.from_numpy(Xtest).type(torch.FloatTensor)#将Numpy数据转化为张量
Ytest_t = torch.from_numpy(Ytest).type(torch.LongTensor)
print("测试时的准确率：",accuracy_score(model.predict(Xtest_t),Ytest_t))











