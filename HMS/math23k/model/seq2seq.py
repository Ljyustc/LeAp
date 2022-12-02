# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .attention import get_mask
import time
import logging

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, gt_ww, common_dict, know_gt_ww, eval_know_inputs, dropout=0.5):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        self.gt_ww = gt_ww
        self.common_dict = common_dict
        self.know_gt_ww = know_gt_ww
        self.eval_know_inputs = eval_know_inputs
        hidden_size = encoder.hidden_size
        self.ww_gcn = GCN(hidden_size, hidden_size, hidden_size, dropout)
        self.norm = LayerNorm1(hidden_size)
        self.ww_trans = PositionwiseFeedForward(hidden_size, hidden_size, hidden_size, dropout)
        return
    
    def get_word_exist_matrix(self, encoder_outputs, input_lengths, span_length=None):
        span_output, word_outputs = encoder_outputs
        span_pad_length = span_output.size(1)
        word_pad_lengths = [word_output.size(1) for word_output in word_outputs]
        
        span_mask = get_mask(span_length, span_pad_length)
        word_masks = [get_mask(input_length, word_pad_length) for input_length, word_pad_length in zip(input_lengths, word_pad_lengths)]
        word_exists = [(1 - word_masks[i]) * (1 - span_mask[:, [i]]) for i in range(len(word_masks))]
        
        # for all words together
        word_exist_sequence = torch.cat(word_exists, dim=-1)
        words_num = word_exist_sequence.size(-1)
        word_exist_matrix = word_exist_sequence.repeat(1, words_num).reshape(-1, words_num, words_num)
        word_exist_mat = word_exist_matrix * torch.transpose(word_exist_matrix,1,2)
        return word_exist_mat
        
    def forward(self, input_variable, num_pos, input_lengths, span_length=None,
                target_variable=None, tree=None, max_length=None, beam_width=None,
                temp=0.5, thre=0.5, hard=False):
        generator_op_embedding = self.decoder.op_hidden(self.decoder.embed_model(self.decoder.generator_op_vocab))
        encoder_outputs, encoder_hidden, word_word, ww_prob, word_op, wo_prob= self.encoder(
            input_var=input_variable,
            input_lengths=input_lengths,
            span_length=span_length,
            ops_embed=generator_op_embedding,
            tree=tree,
            temp=temp,
            thre=thre,
            hard=hard
        )
        
        word_exist_mat = self.get_word_exist_matrix(encoder_outputs, input_lengths, span_length)
        
        max_words_length = [int(max(i)) for i in input_lengths]
        ww_adj = word_word[:,:,:,0].squeeze(-1).masked_fill(word_exist_mat!=1, 0)
        
        pade_outputs = torch.cat(encoder_outputs[1], dim=1)  # batch x len x dim
        pade_outputs_1 = self.ww_gcn(pade_outputs, ww_adj)
        pade_outputs_2 = self.norm(pade_outputs_1) + pade_outputs
        pade_outputs = self.ww_trans(pade_outputs_2) + pade_outputs_2
        
        encoder_outputs = (encoder_outputs[0], torch.split(pade_outputs, max_words_length, dim=1))
        
        output = self.decoder(
            targets=target_variable,
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            input_lengths=input_lengths,
            span_length=span_length,
            num_pos=num_pos,
            word_word=word_word,
            word_op=word_op,
            max_length=max_length,
            beam_width=beam_width
        )
        return output[0], output[1], output[2], generator_op_embedding, word_exist_mat

class LayerNorm1(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm1, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(torch.sum((x-mean)**2,dim=-1,keepdim=True))
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff,d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        
class GCN(nn.Module):
    def __init__(self, in_feat_dim, nhid, out_feat_dim, dropout):
        super(GCN, self).__init__()
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        - adjacency matrix (batch_size, K, K)
        ## Returns:
        - gcn_enhance_feature (batch_size, K, out_feat_dim)
        '''
        self.gc1 = GraphConvolution(in_feat_dim, nhid)
        self.gc2 = GraphConvolution(nhid, out_feat_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    
# Graph_Conv
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #print(input.shape)
        #print(self.weight.shape)
        support = torch.matmul(input, self.weight)
        #print(adj.shape)
        #print(support.shape)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
