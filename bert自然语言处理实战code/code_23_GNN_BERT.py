from torch import nn


from transformers import BertTokenizer, BertPreTrainedModel,BertConfig

from transformers.activations import  gelu,swish,gelu_new

    
import copy
import math
import numpy as np
from torch.utils.data import RandomSampler

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax


import torch

import dgl
print(dgl.__version__)         #0.5.2
import transformers
print(transformers.__version__)#3.4.0
print(torch.__version__)       #1.6.0

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}
BertLayerNorm = torch.nn.LayerNorm

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, graph):
        node_num = graph.ndata['h'].size(0)

        Q = self.query(graph.ndata['h'])
        K = self.key(graph.ndata['h'])
        V = self.value(graph.ndata['h'])

        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)

        graph.ndata['Q'] = Q
        graph.ndata['K'] = K
        graph.ndata['V'] = V

        graph.apply_edges(fn.u_mul_v('K', 'Q', 'attn_probs'))
        graph.edata['attn_probs'] = graph.edata['attn_probs'].sum(-1, keepdim=True)
        graph.edata['attn_probs'] = edge_softmax(graph, graph.edata['attn_probs'])
        graph.edata['attn_probs'] = self.dropout(graph.edata['attn_probs'])
        graph.apply_edges(fn.u_mul_e('V', 'attn_probs', 'attn_values'))

        graph.update_all(message_func = fn.copy_e('attn_values', 'm'),
                         reduce_func = fn.sum('m', 'h'))
                
        graph.ndata['h'] = graph.ndata['h'].view([node_num, -1])

        return graph


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, graph):
        input_tensor = graph.ndata['h']
        self_output_graph = self.self(graph)
        attention_output = self.output(self_output_graph.ndata['h'], input_tensor)
        graph.ndata['h'] = attention_output
        return graph


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, graph):
        graph = self.attention(graph)
        intermediate_output = self.intermediate(graph.ndata['h'])
        layer_output = self.output(intermediate_output, graph.ndata['h'])
        graph.ndata['h'] = layer_output
        return graph


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, graph):
        for layer_module in self.layer:
            graph = layer_module(graph)
        return graph


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self._init_weights)

    def forward(self, graph):
        embedding_output = self.embeddings(graph.ndata['input_ids'],
                                           graph.ndata['position_ids'],
                                           graph.ndata['segment_ids'])

        graph.ndata.pop('input_ids')
        graph.ndata.pop('position_ids')
        graph.ndata.pop('segment_ids')

        hidden_size = embedding_output.size(-1)
        embedding_output = embedding_output.view(-1, hidden_size)

        graph.ndata['h'] = embedding_output

        graph = self.encoder(graph)

        g_list = dgl.unbatch(graph)

        pooled_output = []
        for g in g_list:
            pooled_output.append(g.ndata['h'][0])
        pooled_output = torch.stack(pooled_output, 0)

        pooled_output = self.pooler(pooled_output)
        return graph, pooled_output





class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self._init_weights)

    def forward(self, graph, labels=None):
        _, pooled_output = self.bert(graph)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


if __name__ == '__main__':
    
    #指定设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    import os
    data_dir='./THUCNews/data'        
    class_list = [x.strip() for x in open(
            os.path.join(data_dir, "class.txt")).readlines()]
    tokenizer = BertTokenizer.from_pretrained( r'./bert-base-chinese/')  
    
    input_ids = tokenizer.encode("成交活跃运作规范 钢材期货上市一周运行平稳") 
    seq_length = len(input_ids)
    segment_ids = np.zeros( seq_length )
    
    g1 = dgl.DGLGraph().to(device)
    g1.add_nodes(seq_length)
    g1.ndata['input_ids'] = torch.tensor(input_ids, dtype=torch.long, device=device)
    g1.ndata['segment_ids'] = torch.tensor(segment_ids, dtype=torch.long, device=device)
    g1.ndata['position_ids'] = torch.arange(len(input_ids), dtype=torch.long, device=device)
    for i in range(seq_length):
        g1.add_edges(i, range(seq_length))
    
    Classification = BertForSequenceClassification.from_pretrained(r'./myfinetun-bert_chinese/')
    Classification.eval()
    Classification.to(device)
    with torch.no_grad():
        value = Classification(g1)
        result = torch.argmax(value,axis=1).cpu().numpy()
        print("分类结果：",class_list[result[0]]," 类索引:",result)








