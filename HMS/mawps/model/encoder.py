# -*- coding:utf-8 -*-

import torch
from torch import nn

from .attention import get_mask, Attention
from .knowledge_base import *

class PositionalEncoding(nn.Module):
    def __init__(self, pos_size, dim):
        super(PositionalEncoding, self).__init__()
        pe = torch.rand(pos_size, dim)
        # (0, 1) => (-1, 1)
        pe = pe * 2 - 1
        self.pe = nn.Parameter(pe)
        return
    
    def forward(self, input):
        output = input + self.pe[:input.size(1)]
        return output

class Encoder(nn.Module):
    def __init__(self, embed_model, hidden_size=512, span_size=0, dropout=0.4):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        embed_size = embed_model.embedding_dim
        
        self.embedding = embed_model
        self.ww_encoder = KnowEncoder(embed_model.embedding_dim*2, hidden_size, 2)
        self.wo_encoder = KnowEncoder1(embed_model.embedding_dim + hidden_size, hidden_size, 2)
        
        # word encoding
        self.word_rnn = nn.GRU(embed_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        # span encoding
        # span sequence
        self.span_attn = Attention(self.hidden_size, mix=True, fn=True)
        self.pos_enc = PositionalEncoding(span_size, hidden_size)
        # merge subtree/word node
        self.to_parent = Attention(self.hidden_size, mix=True, fn=True)
        return

    def bi_combine(self, output, hidden):
        # combine forward and backward LSTM
        # (num_layers * num_directions, batch, hidden_size).view(num_layers, num_directions, batch, hidden_size)
        hidden = hidden[0:hidden.size(0):2] + hidden[1:hidden.size(0):2]
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output, hidden
    
    def dependency_encode(self, word_output, tree):
        word, rel, left, right = tree
        children = left + right
        word_vector = word_output[:, word]
        if len(children) == 0:
            vector = word_vector
        else:
            children_vector = [self.dependency_encode(word_output, child).unsqueeze(1) for child in children]
            children_vector = torch.cat(children_vector, dim=1)
            query = word_vector.unsqueeze(1)
            vector = self.to_parent(query, children_vector)[0].squeeze(1)
        return vector
    
    def encode_prior(self, input_seqs, ops_embed):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        
        all_word = embedded.transpose(0, 1)
        _, ww_prob = self.ww_encoder(all_word, 0.5, True, 0.5)
        
        _, wo_prob = self.wo_encoder(all_word, ops_embed, 0.5, True, 0.5)
        return ww_prob, wo_prob
        
    def forward(self, input_var, input_lengths, span_length, ops_embed, tree=None, temp=0.5, thre=0.5, hard=False):
        use_cuda = span_length.is_cuda
        pad_hidden = torch.zeros(1, self.hidden_size)
        if use_cuda:
            pad_hidden = pad_hidden.cuda()
        
        word_outputs = []
        span_inputs = []
        word_init = []
        
        input_vars = input_var
        trees = tree
        bi_word_hidden = None
        for span_index, input_var in enumerate(input_vars):
            input_length = input_lengths[span_index]

            # word encoding
            embedded = self.embedding(input_var)
            word_init.append(embedded)
            embedded = self.dropout(embedded)
            # at least 1 word in some full padding span
            pad_input_length = input_length.clone()
            pad_input_length[pad_input_length == 0] = 1
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, pad_input_length.cpu(), batch_first=True, enforce_sorted=False)
            word_output, bi_word_hidden = self.word_rnn(embedded, bi_word_hidden)
            word_output, _ = nn.utils.rnn.pad_packed_sequence(word_output, batch_first=True)
            word_output, word_hidden = self.bi_combine(word_output, bi_word_hidden)
            
            # tree encoding
            span_span_input = []
            for data_index, data_word_output in enumerate(word_output):
                data_word_output = data_word_output.unsqueeze(0)
                tree = trees[span_index][data_index]
                if tree is not None:
                    data_span_input = self.dependency_encode(data_word_output, tree)
                else:
                    data_span_input = pad_hidden
                span_span_input.append(data_span_input)
            span_input = torch.cat(span_span_input, dim=0)
            span_inputs.append(span_input.unsqueeze(1))
            word_outputs.append(word_output)
        
        all_word = torch.cat(word_init, dim=1)
        word_word, ww_prob = self.ww_encoder(all_word, temp, hard, thre)
        word_op, wo_prob = self.wo_encoder(all_word, ops_embed, temp, hard, thre)
        
        # span encoding
        span_input = torch.cat(span_inputs, dim=1)
        span_mask = get_mask(span_length, span_input.size(1))
        span_output = self.pos_enc(span_input)
        span_output = self.dropout(span_output)
        span_output, _ = self.span_attn(span_output, span_output, span_mask)
        span_output = span_output * (span_mask == 0).unsqueeze(-1)
        dim0 = torch.arange(span_output.size(0))
        if use_cuda:
            dim0 = dim0.cuda()
        span_hidden = span_output[dim0, span_length - 1].unsqueeze(0)
        return (span_output, word_outputs), span_hidden, word_word, ww_prob, word_op, wo_prob
