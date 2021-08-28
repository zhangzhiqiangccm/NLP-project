# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <PyTorch从深度学习到图神经网络>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Thu Mar 30 09:43:58 2017
"""

import copy, numpy as np
np.random.seed(0) #随机数生成器的种子，可以每次得到一样的值
# compute sigmoid nonlinearity
def sigmoid(x): #激活函数
    output = 1/(1+np.exp(-x))
    return output
# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):#激活函数的导数
    return output*(1-output)


int2binary = {} #整数到其二进制表示的映射
binary_dim = 8 #暂时制作256以内的减法
## 计算0-256的二进制表示
largest_number = pow(2,binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]
    
# input variables
alpha = 0.9 #学习速率
input_dim = 2 #输入的维度是2
hidden_dim = 16 
output_dim = 1 #输出维度为1

# initialize neural network weights
synapse_0 = (2*np.random.random((input_dim,hidden_dim)) - 1)*0.05 #维度为2*16， 2是输入维度，16是隐藏层维度
synapse_1 = (2*np.random.random((hidden_dim,output_dim)) - 1)*0.05
synapse_h = (2*np.random.random((hidden_dim,hidden_dim)) - 1)*0.05
# => [-0.05, 0.05)，

# 用于存放反向传播的权重更新值
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training 
for j in range(10000):
    
    #生成一个数字a
    a_int = np.random.randint(largest_number) 
    #生成一个数字b,b的最大值取的是largest_number/2,作为被减数，让它小一点。
    b_int = np.random.randint(largest_number/2) 
    #如果生成的b大了，那么交换一下
    if a_int<b_int:
        tt = a_int
        b_int = a_int
        a_int=tt
    
    a = int2binary[a_int] # binary encoding
    b = int2binary[b_int] # binary encoding    
    # true answer
    c_int = a_int - b_int
    c = int2binary[c_int]
    
    # 存储神经网络的预测值
    d = np.zeros_like(c)
    overallError = 0 #每次把总误差清零
    
    layer_2_deltas = list() #存储每个时间点输出层的误差
    layer_1_values = list() #存储每个时间点隐藏层的值
    
    layer_1_values.append(np.ones(hidden_dim)*0.1) # 一开始没有隐藏层，所以初始化一下原始值为0.1
    
    # moving along the positions in the binary encoding
    for position in range(binary_dim):#循环遍历每一个二进制位
        
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])#从右到左，每次去两个输入数字的一个bit位
        y = np.array([[c[binary_dim - position - 1]]]).T#正确答案
        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))#（输入层 + 之前的隐藏层） -> 新的隐藏层，这是体现循环神经网络的最核心的地方！！！
        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1)) #隐藏层 * 隐藏层到输出层的转化矩阵synapse_1 -> 输出层
        
        layer_2_error = y - layer_2 #预测误差
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2)) #把每一个时间点的误差导数都记录下来
        overallError += np.abs(layer_2_error[0])#总误差
    
        d[binary_dim - position - 1] = np.round(layer_2[0][0]) #记录下每一个预测bit位
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))#记录下隐藏层的值，在下一个时间点用
    
    future_layer_1_delta = np.zeros(hidden_dim)
    
    #反向传播，从最后一个时间点到第一个时间点
    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]]) #最后一次的两个输入
        layer_1 = layer_1_values[-position-1] #当前时间点的隐藏层
        prev_layer_1 = layer_1_values[-position-2] #前一个时间点的隐藏层
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1] #当前时间点输出层导数
        # error at hidden layer
        # 通过后一个时间点（因为是反向传播）的隐藏层误差和当前时间点的输出层误差，计算当前时间点的隐藏层误差
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
        
        
       # 等到完成了所有反向传播误差计算， 才会更新权重矩阵，先暂时把更新矩阵存起来。
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    
    # 完成所有反向传播之后，更新权重矩阵。并把矩阵变量清零
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
   
    # print out progress
    if(j % 800 == 0):
        #print(synapse_0,synapse_h,synapse_1)
        print("总误差:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print(str(a_int) + " - " + str(b_int) + " = " + str(out))
        print("------------")