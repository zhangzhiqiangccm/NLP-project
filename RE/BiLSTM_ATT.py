import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)  #为CPU设置种子用于生成随机数，以使得结果是确定的

class BiLSTM_ATT(nn.Module):
    def __init__(self,input_size,output_size,config,pre_embedding):
        super(BiLSTM_ATT,self).__init__()
        self.batch = config['BATCH']

        self.input_size = input_size
        self.embedding_dim = config['EMBEDDING_DIM'] # 词向量长度
        
        self.hidden_dim = config['HIDDEN_DIM']
        self.tag_size = output_size # 最终结果状态数，即分类数
        
        self.pos_size = config['POS_SIZE']
        self.pos_dim = config['POS_DIM'] #位置编码向量长度
        
        self.pretrained = config['pretrained']

        if self.pretrained:
            # freeze = False 表示训练过程中会更新这些词向量，默认为True 也就是不更新
            self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(pre_embedding),freeze=False)
        else:
            self.word_embeds = nn.Embedding(self.input_size,self.embedding_dim)

        self.pos1_embeds = nn.Embedding(self.pos_size,self.pos_dim) # 实体1的embedding
        self.pos2_embeds = nn.Embedding(self.pos_size,self.pos_dim) # 实体2的embedding
        self.dense = nn.Linear(self.hidden_dim,self.tag_size,bias=True)
        self.relation_embeds = nn.Embedding(self.tag_size,self.hidden_dim)

        '''
            LSTM 输入变为 pos1_dim + pos2_dim + embedding_dim
            LSTM的output 保存了最后一层，每个time step的输出h，如果是双向LSTM，每个time step的输出h = [h正向, h逆向]
            TODO 这里hidden_size=hidden_dim/2 保证了后面BiLSTM输出的维度为(seq_len,batch_size,hidden_dim)
            注意hidden_size 与 hidden_dim的区分
            hidden_size是单向的LSTM输出的维度
        '''
        self.lstm = nn.LSTM(input_size=self.embedding_dim+self.pos_dim*2,hidden_size=self.hidden_dim//2,num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim,self.tag_size)

        '''
            在嵌入层，LSTM层和倒数第二层上使用drop_out。 
        '''
        self.dropout_emb = nn.Dropout(p=0.5)
        self.dropout_lstm = nn.Dropout(p=0.5)
        self.dropout_att = nn.Dropout(p=0.5)
        
        self.hidden = self.init_hidden()

        # nn.Parameter 类型表示会算入计算图内进行求梯度
        self.att_weight = nn.Parameter(torch.randn(self.batch,1,self.hidden_dim))
        self.relation_bias = nn.Parameter(torch.randn(self.batch,self.tag_size,1))
        
    def init_hidden(self):
        return torch.randn(2, self.batch, self.hidden_dim // 2)
        
    def init_hidden_lstm(self):
        return (torch.randn(2, self.batch, self.hidden_dim // 2),
                torch.randn(2, self.batch, self.hidden_dim // 2))
    '''
        BiLSTM 最后一层的输出 (seq_len,batch_size,hidden_dim)
        attention的参数H是经过转置的结果:(batch_size,hidden_dim,seq_len)
        attention 目的就是根据不同词得到不同词的权重，然后根据权重组合得到整个句子级别的表示 
    '''
    def attention(self,H):
        M = torch.tanh(H) # 非线性变换 size:(batch_size,hidden_dim,seq_len)
        a = F.softmax(torch.bmm(self.att_weight,M),dim=2) # a.Size : (batch_size,1,seq_len)
        a = torch.transpose(a,1,2) # (batch_size,seq_len,1)
        return torch.bmm(H,a) # (batch_size,hidden_dim,1)


    def forward(self,sentence,pos1,pos2):
        '''
        batch_size : 批量大小
        seq_len：序列长度、每句单词个数、时间步数
        vector_size: 单个词向量长度
        :param sentence: (batch_size,seq_len)
        :param pos1: (batch_size,seq_len)
        :param pos2: (batch_size,seq_len)
        :return:
        '''
        self.hidden = self.init_hidden_lstm()

        #TODO 在 dim = 2（也就是词向量那一维度）上面合并，可以理解为每个向量变成了 vector_size + pos1_size + pos2_size
        embeds = torch.cat((self.word_embeds(sentence),self.pos1_embeds(pos1),self.pos2_embeds(pos2)),dim=2)
        # TODO lstm计算要求输入 (seq_len,batch_size,vector)形式，下面表示对dim = 0, 和dim = 1进行转置
        #TODO Size: (seq_len,batch_size,final_vector_size)
        embeds = torch.transpose(embeds,0,1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        
        # lstm_out = torch.transpose(lstm_out,0,1)
        # lstm_out = torch.transpose(lstm_out,1,2)
        #TODO lstm_out:(batch_size,hidden_dim,seq_len)
        lstm_out = lstm_out.permute(1,2,0)
        lstm_out = self.dropout_lstm(lstm_out)

        #TODO (batch_size,hidden_dim,1)
        att_out = torch.tanh(self.attention(lstm_out ))
        # att_out = self.dropout_att(att_out)
        relation = torch.tensor([i for i in range(self.tag_size)], dtype=torch.long).repeat(self.batch, 1)
        relation = self.relation_embeds(relation)
        out = torch.add(torch.bmm(relation, att_out), self.relation_bias)
        out = F.softmax(out,dim=1)
        # out = self.dense(att_out.view(self.batch,self.hidden_dim)) # 经过一个全连接矩阵 W*h + b

        return out.view(self.batch,-1) # size : (batch_size,tag_size)
