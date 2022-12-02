import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from src.utils.sentence_processing import sort_by_len, restore_order
from torch.nn.parameter import Parameter
from .knowledge_base import *

class Encoder(nn.Module):
	'''
	Encoder helps in building the sentence encoding module for a batched version
	of data that is sent in [T x B] having corresponding input lengths in [1 x B]

	Args:
			hidden_size: Hidden size of the RNN cell
			embedding: Embeddings matrix [vocab_size, embedding_dim]
			cell_type: Type of RNN cell to be used : LSTM, GRU
			nlayers: Number of layers of LSTM (default = 1)
			dropout: Dropout Rate (default = 0.1)
			bidirectional: Bidirectional model to be formed (default: False)
	'''

	def __init__(self, hidden_size=512,embedding_size = 768, cell_type='lstm', nlayers=1, dropout=0.1, bidirectional=True):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size
		self.nlayers = nlayers
		self.dropout = dropout
		self.cell_type = cell_type
		self.embedding_size = embedding_size
		# self.embedding_size = self.embedding.embedding_dim
		self.bidirectional = bidirectional

		if self.cell_type == 'lstm':
			self.rnn = nn.LSTM(self.embedding_size, self.hidden_size,
							   num_layers=self.nlayers,
							   dropout=(0 if self.nlayers == 1 else dropout),
							   bidirectional=bidirectional)
		elif self.cell_type == 'gru':
			self.rnn = nn.GRU(self.embedding_size, self.hidden_size,
							  num_layers=self.nlayers,
							  dropout=(0 if self.nlayers == 1 else dropout),
							  bidirectional=bidirectional)
		else:
			self.rnn = nn.RNN(self.embedding_size, self.hidden_size,
							  num_layers=self.nlayers,
							  nonlinearity='tanh',							# ['relu', 'tanh']
							  dropout=(0 if self.nlayers == 1 else dropout),
							  bidirectional=bidirectional)
		self.ww_encoder = KnowEncoder(embedding_size*2, hidden_size, 2)
		self.wo_encoder = KnowEncoder1(embedding_size + hidden_size, hidden_size, 2)
		self.ww_gcn = GCN(hidden_size, hidden_size, hidden_size, dropout)
		self.norm = LayerNorm1(hidden_size)
		self.ww_trans = PositionwiseFeedForward(hidden_size, hidden_size, hidden_size, dropout)
		
	def encode_prior(self, input_seqs, ops_embed):
		# Note: we run this all at once (over multiple batches of multiple sequences)
		embedded = input_seqs  # S x B x E
		
		all_word = embedded.transpose(0, 1)
		_, ww_prob = self.ww_encoder(all_word, 0.5, True, 0.5)

		_, wo_prob = self.wo_encoder(all_word, ops_embed, 0.5, True, 0.5)
		return ww_prob, wo_prob
		
	def forward(self, sorted_seqs, sorted_len, orig_idx, device=None, ops_embed=None, hidden=None, sort_word_exist_mat=None, temp=0.5, thre=0.5, hard=False):
		'''
			Args:
				input_seqs (tensor) : input tensor | size : [Seq_len X Batch_size]
				input_lengths (list/tensor) : length of each input sentence | size : [Batch_size] 
				device (gpu) : Used for sorting the sentences and putting it to device

			Returns:
				output (tensor) : Last State representations of RNN [Seq_len X Batch_size X hidden_size]
				hidden (tuple)	: Hidden states and (cell states) of recurrent networks
		'''

		# sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seqs, input_lengths, device)
		# pdb.set_trace()

		all_word = sorted_seqs.transpose(0, 1)
		word_word, ww_prob = self.ww_encoder(all_word, temp, hard, thre)
		word_op, wo_prob = self.wo_encoder(all_word, ops_embed, temp, hard, thre)

		#embedded = self.embedding(sorted_seqs)  ### NO MORE IDS
		packed = torch.nn.utils.rnn.pack_padded_sequence(
			sorted_seqs, sorted_len)
		outputs, hidden = self.rnn(packed, hidden)
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
			outputs)  # unpack (back to padded)

		if self.bidirectional:
			outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
		outputs = outputs.transpose(0,1)

		ww_adj = word_word[:,:,:,0].squeeze(-1).masked_fill(sort_word_exist_mat!=1, 0)
		# ww_adj = word_word[:,:,:,0].squeeze(-1) * word_exist_mat
		outputs_1 = self.ww_gcn(outputs, ww_adj)
		outputs_2 = self.norm(outputs_1) + outputs
		outputs = self.ww_trans(outputs_2) + outputs_2

		outputs = outputs.transpose(0,1)
		outputs = outputs.index_select(1, orig_idx)
		word_word = word_word.index_select(0, orig_idx)
		ww_prob = ww_prob.index_select(0, orig_idx)
		word_op = word_op.index_select(0, orig_idx)
		wo_prob = wo_prob.index_select(0, orig_idx)

		return outputs, hidden, word_word, ww_prob, word_op, wo_prob 

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
        
# GCN
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
class GraphConvolution(nn.Module):
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
