import torch
import math
import numpy as np
from transformers import BertModel

'''
通过手动矩阵运算实现transformer结构
'''

bert = BertModel.from_pretrained(r"D:\badou\pretrain_model\chinese-bert_chinese_wwm_pytorch")
state_dict = bert.state_dict()
bert.eval()
x = np.array([2450, 15486, 15167, 2110]) #通过vocab对应输入：深度学习
torch_x = torch.LongTensor([x])  #pytorch形式输入
seqence_output, pooler_output = bert(torch_x)
print(seqence_output.shape, pooler_output.shape)
print(seqence_output, pooler_output)

# print(bert.state_dict().keys())  #查看所有的权值矩阵名称

# for key in bert.state_dict():
#     print("%s = state_dict[\"%s\"].numpy()"%(key.replace(".", "_"), key))
embeddings_word_embeddings_weight = state_dict["embeddings.word_embeddings.weight"].numpy()
embeddings_position_embeddings_weight = state_dict["embeddings.position_embeddings.weight"].numpy()
embeddings_token_type_embeddings_weight = state_dict["embeddings.token_type_embeddings.weight"].numpy()
embeddings_LayerNorm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
embeddings_LayerNorm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
encoder_layer_0_attention_self_query_weight = state_dict["encoder.layer.0.attention.self.query.weight"].numpy()
encoder_layer_0_attention_self_query_bias = state_dict["encoder.layer.0.attention.self.query.bias"].numpy()
encoder_layer_0_attention_self_key_weight = state_dict["encoder.layer.0.attention.self.key.weight"].numpy()
encoder_layer_0_attention_self_key_bias = state_dict["encoder.layer.0.attention.self.key.bias"].numpy()
encoder_layer_0_attention_self_value_weight = state_dict["encoder.layer.0.attention.self.value.weight"].numpy()
encoder_layer_0_attention_self_value_bias = state_dict["encoder.layer.0.attention.self.value.bias"].numpy()
encoder_layer_0_attention_output_dense_weight = state_dict["encoder.layer.0.attention.output.dense.weight"].numpy()
encoder_layer_0_attention_output_dense_bias = state_dict["encoder.layer.0.attention.output.dense.bias"].numpy()
encoder_layer_0_attention_output_LayerNorm_weight = state_dict["encoder.layer.0.attention.output.LayerNorm.weight"].numpy()
encoder_layer_0_attention_output_LayerNorm_bias = state_dict["encoder.layer.0.attention.output.LayerNorm.bias"].numpy()
encoder_layer_0_intermediate_dense_weight = state_dict["encoder.layer.0.intermediate.dense.weight"].numpy()
encoder_layer_0_intermediate_dense_bias = state_dict["encoder.layer.0.intermediate.dense.bias"].numpy()
encoder_layer_0_output_dense_weight = state_dict["encoder.layer.0.output.dense.weight"].numpy()
encoder_layer_0_output_dense_bias = state_dict["encoder.layer.0.output.dense.bias"].numpy()
encoder_layer_0_output_LayerNorm_weight = state_dict["encoder.layer.0.output.LayerNorm.weight"].numpy()
encoder_layer_0_output_LayerNorm_bias = state_dict["encoder.layer.0.output.LayerNorm.bias"].numpy()
pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()

#bert的计算过程
#这里是伪码，没有传入模型权重，所以不能直接计算
def bert_forward_calculation(x):
    #x.shape = [max_len, 1]
    x_embedding      = embedding_layer(x)           #shpae: [max_len, hidden_size]
    x_self_attention = self_attention(x_embedding)  #shpae: [max_len, hidden_size]
    x_attention      = layer_norm(x_embedding + x_self_attention) #shpae: [max_len, hidden_size]
    x_feed_forward   = feed_forward(x_attention)    #shpae: [max_len, hidden_size]
    x_hidden         = layer_norm(x_feed_forward + x_attention) #shpae: [max_len, hidden_size]
    return x_hidden

#layer normalization
#只针对2维的输入（实际上并无此限制，只是为了实现方便）
#输入输出同维度
#x.shape = m x n, output shape = m x n
def layer_norm(w, b, x):
    x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
    x = x * w + b
    return x

#embedding层实际上相当于按index索引
def get_embedding(embedding_matrix, x):
    return np.array([embedding_matrix[index] for index in x])

#softmax归一化
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

#gelu激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

#手动实现bert的Embedding层
#word embedding + position embedding + token_type_embedding
#之后layer normalization
def embedding_layer(word_embeddings,
                    position_embeddings,
                    token_type_embeddings,
                    embeddings_layer_norm_weight,
                    embeddings_layer_norm_bias,
                    x):
    #x.shape = [max_len]
    we = get_embedding(word_embeddings, x)  #shpae: [max_len, hidden_size]
    #position embeding的输入 [0, 1, 2, 3]
    pe = get_embedding(position_embeddings, np.array(list(range(len(x))))) #shpae: [max_len, hidden_size]
    #token type embedding,单输入的情况下为[0, 0, 0, 0]
    te = get_embedding(token_type_embeddings, np.array([0] * len(x))) #shpae: [max_len, hidden_size]
    embedding = we + pe + te
    #加和后有一个归一化层
    embedding = layer_norm(embeddings_layer_norm_weight, embeddings_layer_norm_bias, embedding) #shpae: [max_len, hidden_size]
    return embedding

#torch原装，只输出embedding层结果
torch_embeddings_layer_output = bert.embeddings(torch_x)
# embedding层计算结果
diy_embedding_output = embedding_layer(embeddings_word_embeddings_weight,
                                              embeddings_position_embeddings_weight,
                                              embeddings_token_type_embeddings_weight,
                                              embeddings_LayerNorm_weight,
                                              embeddings_LayerNorm_bias,
                                              x)

# print("pytorch Bert Embedding层输出：", torch_embeddings_layer_output)
# print("手动实现 Bert Embedding层输出：", diy_embedding_output)

#多头机制，将权值矩阵拆分为num_attention_heads组
def transpose_for_scores(x, attention_head_size, num_attention_heads):
    max_len, hidden_size = x.shape
    x = x.reshape(max_len, num_attention_heads, attention_head_size)
    x = x.swapaxes(1, 0) #output shape = [num_attention_heads, max_len, attention_head_size]
    return x

#self attention的计算
def self_attention(q_w,
                   q_b,
                   k_w,
                   k_b,
                   v_w,
                   v_b,
                   attention_output_weight,
                   attention_output_bias,
                   num_attention_heads,
                   hidden_size,
                   x):
    #x.shape = max_len * hidden_size
    #q_w, k_w, v_w  shape = hidden_size * hidden_size
    #q_b, k_b, v_b  shape = hidden_size
    q = np.dot(x, q_w.T) + q_b  #shape: [max_len, hidden_size]
    k = np.dot(x, k_w.T) + k_b  #shpae: [max_len, hidden_size]
    v = np.dot(x, v_w.T) + v_b  #shpae: [max_len, hidden_size]
    attention_head_size = int(hidden_size / num_attention_heads)
    # q.shape = num_attention_heads, max_len, attention_head_size
    q = transpose_for_scores(q, attention_head_size, num_attention_heads)
    # k.shape = num_attention_heads, max_len, attention_head_size
    k = transpose_for_scores(k, attention_head_size, num_attention_heads)
    # v.shape = num_attention_heads, max_len, attention_head_size
    v = transpose_for_scores(v, attention_head_size, num_attention_heads)
    # qk.shape = num_attention_heads, max_len, max_len
    qk = np.matmul(q, k.swapaxes(1, 2))
    qk /= np.sqrt(attention_head_size)
    qk = softmax(qk)
    # qkv.shape = num_attention_heads, max_len, attention_head_size
    qkv = np.matmul(qk, v)
    # qkv.shape = max_len, hidden_size
    qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
    #attention.shape = max_len, hidden_size
    attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
    return attention

#预设参数，需要和预训练中保持一致
num_attention_heads = 12
hidden_size = 768
#计算self attention
diy_self_attention = self_attention(encoder_layer_0_attention_self_query_weight,
                                    encoder_layer_0_attention_self_query_bias,
                                    encoder_layer_0_attention_self_key_weight,
                                    encoder_layer_0_attention_self_key_bias,
                                    encoder_layer_0_attention_self_value_weight,
                                    encoder_layer_0_attention_self_value_bias,
                                    encoder_layer_0_attention_output_dense_weight,
                                    encoder_layer_0_attention_output_dense_bias,
                                    num_attention_heads,
                                    hidden_size,
                                    diy_embedding_output)

#layer norm (x + z)
#shape:max_len, hidden_size
diy_attention_output = layer_norm(encoder_layer_0_attention_output_LayerNorm_weight,
                                  encoder_layer_0_attention_output_LayerNorm_bias,
                                  diy_self_attention + diy_embedding_output)

def feed_forward(intermediate_weight,  #intermediate_size, hidden_size
                 intermediate_bias,    #intermediate_size
                 output_weight,        #hidden_size, intermediate_size
                 output_bias,          #hidden_size
                 x):
    #output shpae: [max_len, intermediate_size]
    x = np.dot(x, intermediate_weight.T) + intermediate_bias
    x = gelu(x)
    #output shpae: [max_len, hidden_size]
    x = np.dot(x, output_weight.T) + output_bias
    return x

#前馈网络
diy_feed_forward_output = feed_forward(encoder_layer_0_intermediate_dense_weight,
                                       encoder_layer_0_intermediate_dense_bias,
                                       encoder_layer_0_output_dense_weight,
                                       encoder_layer_0_output_dense_bias,
                                       diy_attention_output)

#add & normalization
diy_sequence_output = layer_norm(encoder_layer_0_output_LayerNorm_weight,
                                 encoder_layer_0_output_LayerNorm_bias,
                                 diy_feed_forward_output + diy_attention_output)
#sequence output
# print(diy_sequence_output)
# print(bert(torch_x)[0])

#pooler output
# diy_pooler_output = np.dot(diy_sequence_output[0], pooler_dense_weight.T) + pooler_dense_bias
# diy_pooler_output = np.tanh(diy_pooler_output)
# print(diy_pooler_output)
# print(bert(torch_x)[1])