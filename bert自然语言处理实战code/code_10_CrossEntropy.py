# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch从深度学习到图神经网络>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Mon Apr  8 22:19:48 2019
"""

import torch
logits = torch.autograd.Variable(torch.tensor([[2,  0.5,6], [0.1,0,  3]]))
labels = torch.autograd.Variable(torch.LongTensor([2,1]))
print(logits)
print(labels)
print('Softmax:',torch.nn.Softmax(dim=1)(logits))
logsoftmax = torch.nn.LogSoftmax(dim=1)(logits)
print('logsoftmax:',logsoftmax)
output = torch.nn.NLLLoss()(logsoftmax, labels)
print('NLLLoss:',output)
print ( 'CrossEntropyLoss:', torch.nn.CrossEntropyLoss()(logits, labels) )