# -*- coding:utf-8 -*-

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math

def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))
    
def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return F.softmax(y / tau, dim=-1)
    
def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return x

class KnowEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(KnowEncoder, self).__init__()
        self.mlp = MLP(n_in, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
                
    def know_compute(self, inputs, seq_len, temp, hard, thre=0.5):
        # inputs: (batch, seq, dim)
        x = torch.cat((inputs.unsqueeze(1).repeat(1,seq_len,1,1), inputs.unsqueeze(2).repeat(1,1,seq_len,1)), dim=-1)
        # x: (batch, seq, seq, dim*2)
        x = self.mlp(x)
        logits = self.fc_out(x)
        prob = F.softmax(logits, dim=-1)
        if hard==False:
            ww_edges = gumbel_softmax(logits, tau=temp, hard=hard) # ww_edges: (batch, seq, seq, 2)
        else:
            ww_edges = (prob > thre).float()
        return ww_edges, prob
    
    def batch_know_compute(self, inputs0, inputs1, batch, seq_len, temp, hard, thre=0.5):
        # inputs0: (1, batch, dim)
        # inputs1: (1, seq_len, dim)
        x = torch.cat((inputs0.unsqueeze(2).repeat(1,1,seq_len,1), inputs1.unsqueeze(1).repeat(1,batch,1,1)), dim=-1)
        # x: (1, batch, seq, dim*2)
        x = self.mlp(x)
        logits = self.fc_out(x)
        prob = F.softmax(logits, dim=-1)
        if hard==False:
            ww_edges = gumbel_softmax(logits, tau=temp, hard=hard) # ww_edges: (1, batch, seq, 2)
        else:
            ww_edges = (prob > thre).float()
        return ww_edges, prob
        
    def embed_know(self, inputs, temp, hard):
        # inputs: (word_num, dim)
        seq_len = inputs.size(0)
        inputs1 = inputs.unsqueeze(0)
        _, prob = self.know_compute(inputs1, seq_len, temp, hard)
        return prob[:,:,:,0].squeeze(-1).squeeze(0)
    
    def batch_embed_know(self, batch_inputs, all_inputs, temp, hard):
        # batch_inputs: (batch, dim)
        # all_inputs: (word_num, dim)
        batch = batch_inputs.size(0)
        seq_len = all_inputs.size(0)
        inputs0 = batch_inputs.unsqueeze(0)
        inputs1 = all_inputs.unsqueeze(0)
        _, prob = self.batch_know_compute(inputs0, inputs1, batch, seq_len, temp, hard)
        return prob[:,:,:,0].squeeze(-1).squeeze(0)
        
    def forward(self, inputs, temp, hard, thre):
        # inputs: (batch, seq, dim)
        _, seq_len, _ = inputs.size()
        return self.know_compute(inputs, seq_len, temp, hard, thre)

class KnowEncoder1(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(KnowEncoder1, self).__init__()
        self.mlp = MLP(n_in, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
                
    def know_compute(self, word_inputs, op_inputs, seq_len, op_num, temp, hard, thre=0.5):
        # word_inputs: (batch, seq, dim1)
        # op_inputs: (batch, op_num, dim2)
        x = torch.cat((word_inputs.unsqueeze(2).repeat(1,1,op_num,1),op_inputs.unsqueeze(1).repeat(1,seq_len,1,1)),dim=-1)
        # x: (batch, seq, op_num, dim1+dim2)
        x = self.mlp(x)
        logits = self.fc_out(x)
        prob = F.softmax(logits, dim=-1)
        if hard==False:
            ww_edges = gumbel_softmax(logits, tau=temp, hard=hard) # ww_edges: (batch, seq, op_num, 2)
        else:
            ww_edges = (prob > thre).float()
        return ww_edges, prob
        
    def embed_know(self, word_inputs, op_inputs, temp, hard):
        # inputs: (word_num, dim1)
        # op_inputs: (op_num, dim2)
        seq_len = word_inputs.size(0)
        word_inputs1 = word_inputs.unsqueeze(0)
        op_num, _ = op_inputs.size()
        op_inputs1 = op_inputs.unsqueeze(0)
        _, prob = self.know_compute(word_inputs1, op_inputs1, seq_len, op_num, temp, hard)
        return prob[:,:,:,0].squeeze(-1).squeeze(0)
        
    def forward(self, word_inputs, op_inputs, temp, hard, thre):
        # word_inputs: (batch, seq, dim1)
        # op_inputs: (op_num, dim2)
        batch_size, seq_len, _ = word_inputs.size()
        op_num, _ = op_inputs.size()
        op_inputs1 = op_inputs.unsqueeze(0).repeat(batch_size, 1, 1)
        return self.know_compute(word_inputs, op_inputs1, seq_len, op_num, temp, hard, thre)