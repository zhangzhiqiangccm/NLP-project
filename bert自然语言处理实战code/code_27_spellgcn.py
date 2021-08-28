# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:29:59 2020

@author: ljh
"""
from dgl.nn import GraphConv
from transformers import BertTokenizer, BertModel, BertConfig,BertLMHeadModel
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_graph(graph_dir):
    nodes_vocab = {}
    with open("%s/nodes_vocab.txt"%(graph_dir),encoding="UTF-8") as f:
      for i, line in enumerate(f):
        nodes_vocab.setdefault(line.strip(), i) 

    node1s,node2s = [],[]
    with open("%s/spellGraphs.txt"%(graph_dir),encoding="UTF-8") as f:
        for i, line in enumerate(f):
          e1,e2, rel = line.strip().split("|")  
          node1s.append(nodes_vocab[e1])
          node2s.append(nodes_vocab[e2])

    g1 = dgl.graph((node1s, node2s),num_nodes=len(nodes_vocab))
          
    w2n = []
    vocab = {}
    with open("%s/vocab.txt"%(graph_dir),encoding="UTF-8") as f:
      for i, line in enumerate(f):
        word = line.strip()
        vocab.setdefault(word, i)
        if word in nodes_vocab:
          w2n.append(nodes_vocab[word])
        else:
          w2n.append(0)
    n2w = []
    with open("%s/nodes_vocab.txt"%(graph_dir),encoding="UTF-8") as f:
      for i, line in enumerate(f):
        word = line.strip()
        if word in vocab:
          n2w.append(vocab[word])
        else:
          n2w.append(0)    
    return  g1,w2n,n2w

graph_dir = r'./gcn_graph'

config = BertConfig.from_pretrained(r'./bert-base-chinese')
config.is_decoder = True

g1,w2n,n2w = load_graph(graph_dir)
w2n=torch.tensor(w2n)
n2w =torch.tensor(n2w)
g = dgl.add_self_loop(g1)

mask_nodes_ids = torch.where( w2n !=0)[0]  #找到不为0的id
maskbase = torch.zeros( (config.vocab_size,config.hidden_size) )
maskbase[mask_nodes_ids] =1.

class MGCNNet(nn.Module):
    def __init__(self):
        super(MGCNNet, self).__init__()
        self.gcn1 = GraphConv(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.gcn2 = GraphConv(config.hidden_size, config.hidden_size)

    def forward(self, g, features):
        gcn1out = self.gcn1(g, features)
        x = self.dropout(gcn1out)
        gcn2out = self.gcn2(g, x)
        return features+gcn1out+gcn2out

class spellgcnBert(nn.Module):
    def __init__(self, MLbert):
        super(spellgcnBert, self).__init__()
        self.MLbert = MLbert
        self.gnnmodel = MGCNNet()


    def getgnnemb(self):
        feat = self.MLbert.bert.embeddings.word_embeddings( n2w )#( input_ids=torch.tensor([n2w]).to(device) )   
        node_embedding = self.gnnmodel(g, feat)  #[4755, 768]
        expanded_node_embedding = node_embedding[w2n]#21128, 768
        rest_embedding = self.MLbert.bert.get_input_embeddings().weight#21128, 768]
        gcn_embedding = maskbase * expanded_node_embedding + (1 - maskbase) *  rest_embedding
        return gcn_embedding


    def forward(self, input_ids, input_mask, segment_ids):
        gcn_embedding = self.getgnnemb()
        outputs = self.MLbert.bert(input_ids, input_mask, segment_ids) #prob [batch_size, seq_len, 1]
        sequence_output = outputs[0]        
        hidden_states = self.MLbert.cls.predictions.transform(sequence_output)
        prediction_scores =F.linear(hidden_states, gcn_embedding, MLbert.cls.predictions.bias)
        return prediction_scores
    
tokenizer = BertTokenizer.from_pretrained(r'./bert-base-chinese')
MLbert = BertLMHeadModel.from_pretrained(r'./bert-base-chinese', config=config)
spellgcnBertmodel = spellgcnBert(MLbert)
    