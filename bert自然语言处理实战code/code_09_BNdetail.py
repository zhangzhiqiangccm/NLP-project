# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch从深度学习到图神经网络>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Tue Jan 21 19:34:35 2020
"""


import torch
import torch.nn as nn
data=torch.randn(2,2,2,1)
print(data)
obn=nn.BatchNorm2d(2,affine=True) #实例化自适应BN对象
output=obn(data)
 

print(obn.weight)
print(obn.bias)
print(obn.eps)
print(output,output.size())


print("第1通道的数据:",data[:,0])
 
#计算第1通道数据的均值和方差
Mean=torch.Tensor.mean(data[:,0])
Var=torch.Tensor.var(data[:,0],False)   #false表示贝塞尔校正不会被使用
print(Mean)
print(Var)

#计算第1通道中第一个数据的BN
batchnorm=((data[0][0][0][0]-Mean)/(torch.pow(Var,0.5)+obn.eps))\
    *obn.weight[0]+obn.bias[0]
print(batchnorm)





import torch
data=torch.randn(1,1,1)#tensor([[[1.3868]]])
data.expand(1, 1, 2)#tensor([[[1.3868, 1.3868]]])
data.repeat(1,1,2)
import torch
data=torch.rand(2,4)#tensor([[0.2316, 0.3987, 0.6225, 0.5304],
                    #        [0.7686, 0.3504, 0.8837, 0.7697]])
torch.multinomial(data, 1)#tensor([[1], [2]])
torch.multinomial(data, 1)#tensor([[1], [0]])