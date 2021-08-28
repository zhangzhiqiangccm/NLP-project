# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch从深度学习到图神经网络>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Thu Apr 25 15:18:57 2019
"""

import torch
# [batch, in_channels, in_height, in_width] [训练时一个batch的图片数量, 图像通道数, 图片高度, 图片宽度]
input1 = torch.ones([1, 1, 5, 5])
input2 = torch.ones([1, 2, 5, 5])
input3 = torch.ones([1, 1, 4, 4])
# [ out_channels, in_channels，filter_height, filter_width] [卷积核个数，图像通道数，卷积核的高度，卷积核的宽度]
filter1 =  torch.tensor([-1.0,0,0,-1]).reshape([2, 2, 1, 1])
filter2 =  torch.tensor([-1.0,0,0,-1,-1.0,0,0,-1]).reshape([2,1,2, 2])
filter3 =  torch.tensor([-1.0,0,0,-1,-1.0,0,0,-1,-1.0,0,0,-1]).reshape([3,1,2, 2])
filter4 =  torch.tensor([-1.0,0,0,-1,-1.0,0,0,-1,
                                   -1.0,0,0,-1,
                                   -1.0,0,0,-1]).reshape([2, 2, 2, 2])
filter5 =  torch.tensor([-1.0,0,0,-1,-1.0,0,0,-1]).reshape([1,2, 2, 2])

#class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
#condv = torch.nn.Conv2d(1,1,kernel_size=1,padding=1, bias=False)
#condv.weight = torch.nn.Parameter(torch.ones([1,1,1,1]))
#padding1 = condv(input1)
#print(padding1)

#验证padding补0的规则 ——上下左右都补0
padding1 = torch.nn.functional.conv2d(input1, torch.ones([1,1,1,1]), stride=1, padding=1)
print(padding1)


padding2 = torch.nn.functional.conv2d(input1, torch.ones([1,1,1,1]), stride=1, padding=(1,2))
print(padding2)

##1个通道输入，生成1个feature map
#filter1 =  torch.tensor([-1.0,0,0,-1]).reshape([1, 1, 2, 2])
#op1 = torch.nn.functional.conv2d(input1, filter1, stride=2, padding=1)
#print('\n')
#print(padding1)
#print(filter1)
#print(op1)

#torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
#torch.nn.functional.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
#torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)





op1 = torch.nn.functional.conv2d(input1, filter1, stride=2, padding=1) #1个通道输入，生成1个feature map
op2 = torch.nn.functional.conv2d(input1, filter2, stride=2, padding=1) #1个通道输入，生成2个feature map
op3 = torch.nn.functional.conv2d(input1, filter3, stride=2, padding=1) #1个通道输入，生成3个feature map

op4 = torch.nn.functional.conv2d(input2, filter4, stride=2, padding=1) # 2个通道输入，生成2个feature
op5 = torch.nn.functional.conv2d(input2, filter5, stride=2, padding=1) # 2个通道输入，生成一个feature map

op6 = torch.nn.functional.conv2d(input1, filter1, stride=2, padding=0) # 5*5 对于pading不同而不同


print("op1:\n",op1,filter1)#1-1  后面补0
print("------------------")

print("op2:\n",op2,filter2) #1-2多卷积核 按列取
print("op3:\n",op3,filter3) #1-3
print("------------------")

print("op4:\n",op4,filter4)#2-2    通道叠加
print("op5:\n",op5,filter5)#2-1
print("------------------")

print("op1:\n",op1,filter1)#1-1
print("op6:\n",op6,filter1)

