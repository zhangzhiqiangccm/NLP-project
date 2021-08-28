# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch从深度学习到图神经网络>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Sat Apr 27 07:04:02 2019
"""

import torch

img=torch.tensor([ [ [0.,0.,0.,0.],[1.,1.,1.,1.],[2.,2.,2.,2.],[3.,3.,3.,3.] ],
                   [ [4.,4.,4.,4.],[5.,5.,5.,5.],[6.,6.,6.,6.],[7.,7.,7.,7.] ]
                 ]).reshape([1,2,4,4])
print(img)
print(img[0][0])
print(img[0][1])

#torch.nn.functional.avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
pooling=torch.nn.functional.max_pool2d(img,kernel_size =2)
print("pooling:\n",pooling)
pooling1=torch.nn.functional.max_pool2d(img,kernel_size =2,stride=1)
print("pooling1:\n",pooling1)
pooling2=torch.nn.functional.avg_pool2d(img,kernel_size =4,stride=1,padding=1)
print("pooling2:\n",pooling2)
pooling3=torch.nn.functional.avg_pool2d(img,kernel_size =4)
print("pooling3:\n",pooling3)

m1 = img.mean(3)
print("第1次平均值结果:\n",m1)
print("第2次平均值结果:\n",m1.mean(2))
