import networkx as nx
# import matplotlib.pyplot as plt
import torch as t
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import dgl
import dgl.function as fn
import math


class MODEL(nn.Module):
    def __init__(self, args, userNum, itemNum, hide_dim, layerNum=1):
        super(MODEL, self).__init__()
        self.userNum = userNum
        self.itemNum = itemNum
        self.hide_dim = hide_dim
        self.layerNum = layerNum
        self.embedding_dict = self.init_weight(userNum, itemNum, hide_dim)
        self.args = args

        self.layers = nn.ModuleList()
        for i in range(self.layerNum):
            self.layers.append(GCNLayer())
    
    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(t.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(t.empty(itemNum, hide_dim))),
        })
        return embedding_dict
    

    def forward(self, graph):

        res_user_embedding = self.embedding_dict['user_emb']
        res_item_embedding = self.embedding_dict['item_emb']

        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(graph, res_user_embedding, res_item_embedding)
            else:
                embeddings = layer(graph, embeddings[: self.userNum], embeddings[self.userNum: ])
            
            res_user_embedding = res_user_embedding + embeddings[: self.userNum]*(1/(i+2))
            res_item_embedding = res_item_embedding + embeddings[self.userNum: ]*(1/(i+2))

        user_embedding = res_user_embedding# / (len(self.layers)+1)

        item_embedding = res_item_embedding# / (len(self.layers)+1)

        return user_embedding, item_embedding


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, graph, u_f, v_f):
        with graph.local_scope():
            node_f = t.cat([u_f, v_f], dim=0)
            # D^-1/2
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = t.pow(degs, -0.5).view(-1, 1)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            graph.update_all(message_func=fn.copy_src(src='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = t.pow(degs, -0.5).view(-1, 1)
            rst = rst * norm

            return rst