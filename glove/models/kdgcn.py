# -*- coding: utf-8 -*-
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from glove.layers.dynamic_rnn import DynamicLSTM

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        start = int((text.shape[1]))
        adj = torch.narrow(adj, 1, 0, start)
        adj = torch.narrow(adj, 2, 0, start)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def select(matrix, top_num):
    batch = matrix.size(0)
    len = matrix.size(1)
    matrix = matrix.reshape(batch, -1)
    maxk, _ = torch.topk(matrix, top_num, dim=1)

    for i in range(batch):
        matrix[i] = (matrix[i] >= maxk[i][-1])
    matrix = matrix.reshape(batch, len, len)
    matrix = matrix + matrix.transpose(-2, -1)

    # selfloop
    for i in range(batch):
        matrix[i].fill_diagonal_(1)

    return matrix

class MultiHeadAttention(nn.Module):
    # d_model:hidden_dim，h:head_num
    def __init__(self, head_num, hidden_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % head_num == 0

        self.d_k = int(hidden_dim // head_num)
        self.head_num = head_num
        self.linears = clones(nn.Linear(hidden_dim, hidden_dim), 2)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, score_mask, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if score_mask is not None:
            scores = scores.masked_fill(score_mask, -1e9)

        b = ~score_mask[:, :, :, 0:1]
        p_attn = F.softmax(scores, dim=-1) * b.float()
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn

    def forward(self, query, key, score_mask):
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.head_num, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        attn = self.attention(query, key, score_mask, dropout=self.dropout)

        return attn


class KDGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(KDGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.kg_gc = GraphConvolution(opt.hidden_dim*2+23, opt.hidden_dim*2+23)
        self.prase_gc = GraphConvolution(opt.hidden_dim*2, opt.hidden_dim*2)

        self.dropout = nn.Dropout(opt.dropout)
        self.weight_n = nn.Parameter(torch.Tensor(opt.hidden_dim*2, opt.hidden_dim*2))
        self.bias_n = nn.Parameter(torch.Tensor(1))
        self.attn = MultiHeadAttention(opt.num_head, opt.hidden_dim * 2+23)
        self.kg_fc = nn.Linear(2*opt.hidden_dim+23, opt.hidden_dim)
        self.fc = nn.Linear(4*opt.hidden_dim+23, opt.polarities_dim)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return weight*x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return mask*x

    def inputs_to_att_adj(self, input, score_mask):
        attn_tensor = self.attn(input, input, score_mask)  # [batch_size, head_num, seq_len, seq_len]
        attn_tensor = torch.sum(attn_tensor, dim=1)
        attn_tensor = select(attn_tensor, 2) * attn_tensor  #self.args.top_k=2
        return attn_tensor


    def forward(self, inputs):
        #预处理
        text_indices, aspect_indices, left_indices, parse_adj, kg_feature, aspect_feature, aspect_expand = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        aspect_expand_len = torch.sum(aspect_expand != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)

        # embeding
        text = self.embed(text_indices)
        aspect_expand =self.embed(aspect_expand)

        text = self.dropout(text)

        # bi-LSTM
        text_out, (text_output, _) = self.text_lstm(text, text_len)

        aspect_expand_out,_ = self.text_lstm(aspect_expand,aspect_expand_len)


        t_a_feature = torch.cat((text_out,aspect_expand_out),dim=1) #text  aspect  feature

        fuse_feature = torch.zeros(t_a_feature.shape[0], t_a_feature.shape[1], t_a_feature.shape[2]+23).cuda()
        fuse_feature[:, :text_out.shape[1], t_a_feature.shape[2]:t_a_feature.shape[2] + 23] = kg_feature[:,:text_out.shape[1],:23]         #23 dimensional emotion vector integration
        fuse_feature[:, text_out.shape[1]:, t_a_feature.shape[2]:t_a_feature.shape[2] + 23] = aspect_feature[:, :aspect_expand_out.shape[1], :23]   #23 dimensional emotion vector integration for aspect expendings

        fuse_feature[:, :text_out.shape[1], :t_a_feature.shape[2]] = text_out
        fuse_feature[:, text_out.shape[1]:, :t_a_feature.shape[2]] = aspect_expand_out


        score_mask = torch.matmul(fuse_feature, fuse_feature.transpose(1, 2))
        score_mask = (score_mask == 0)
        score_mask = score_mask.unsqueeze(1).cuda()
        att_adj = self.inputs_to_att_adj(fuse_feature, score_mask)  # [batch_size, head_num, seq_len, hidden]  #inputs)


        # gcn
        x = F.relu(self.kg_gc(fuse_feature, att_adj))
        x3 = x[:,:text_out.shape[1],:]

        x = F.relu(self.prase_gc(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), parse_adj))
        x = F.relu(self.prase_gc(self.position_weight(x, aspect_double_idx, text_len, aspect_len), parse_adj))


        # mask
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x01 = torch.matmul(alpha, text_out)
        x1 = self.dropout(x01.squeeze(1))

        x3 = self.mask(x3, aspect_double_idx)
        x3 = torch.div(torch.sum(x3, dim=1), aspect_len.float().view(aspect_len.size(0), 1))
        x3 = self.dropout(x3)

        #concat

        x_out = torch.cat((x1,x3),dim=1)

        output = self.fc(x_out)
        return output