import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from .encoder import GCN, LayerNorm1

# Luong attention layer
class Attn(nn.Module):
	def __init__(self, method, hidden_size):
		super(Attn, self).__init__()
		self.method = method
		if self.method not in ['dot', 'general', 'concat']:
			raise ValueError(self.method, "is not an appropriate attention method.")
		self.hidden_size = hidden_size
		if self.method == 'general':
			self.attn = nn.Linear(self.hidden_size, hidden_size)
		elif self.method == 'concat':
			self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
			self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

	def dot_score(self, hidden, encoder_outputs):
		return torch.sum(hidden * encoder_outputs, dim=2)

	def general_score(self, hidden, encoder_outputs):
		energy = self.attn(encoder_outputs)
		return torch.sum(hidden * energy, dim=2)

	def concat_score(self, hidden, encoder_outputs):
		energy = self.attn(torch.cat((hidden.expand(encoder_outputs.size(0), -1, -1), encoder_outputs), 2)).tanh()
		return torch.sum(self.v * energy, dim=2)

	def forward(self, hidden, encoder_outputs):
		# Calculate the attention weights (energies) based on the given method
		if self.method == 'general':
			attn_energies = self.general_score(hidden, encoder_outputs)
		elif self.method == 'concat':
			attn_energies = self.concat_score(hidden, encoder_outputs)
		elif self.method == 'dot':
			attn_energies = self.dot_score(hidden, encoder_outputs)

		# Transpose max_length and batch_size dimensions
		attn_energies = attn_energies.t()

		# Return the softmax normalized probability scores (with added dimension)
		return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
	def __init__(self, attn_model, embedding, cell_type, hidden_size, output_size, nlayers=1, dropout=0.1):
		super(LuongAttnDecoderRNN, self).__init__()

		# Keep for reference
		self.attn_model 	= attn_model
		self.hidden_size 	= hidden_size
		self.output_size 	= output_size
		self.nlayers 		= nlayers
		self.dropout 		= dropout
		self.cell_type 		= cell_type

		# Define layers
		self.embedding = embedding
		self.embedding_size  = self.embedding.embedding_dim
		self.embedding_dropout = nn.Dropout(self.dropout)
		if self.cell_type == 'gru':
			self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.nlayers, dropout=(0 if self.nlayers == 1 else self.dropout))
		else:
			self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.nlayers, dropout=(0 if self.nlayers == 1 else self.dropout))
		self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)

		op_nums = 4 # +,-,*,/
		self.ops = nn.Parameter(torch.nn.init.uniform_(torch.empty(op_nums, hidden_size), a=-1/(hidden_size), b=1/(hidden_size)))
		self.ops_bias = nn.Parameter(torch.nn.init.uniform_(torch.empty(op_nums), a=-1/(hidden_size), b=1/(hidden_size)))
		self.nop_out = nn.Linear(self.hidden_size, self.output_size-op_nums)

		self.attn = Attn(self.attn_model, self.hidden_size)

		self.know_net = Know_Net(hidden_size, hidden_size, dropout)

	def forward(self, input_step, last_hidden, encoder_outputs, word_word, word_op, word_exist_mat, seq_mask):
		# Note: we run this one step (word) at a time
		# Get embedding of current input word
		embedded = self.embedding(input_step)
		embedded = self.embedding_dropout(embedded)

		try:
			embedded = embedded.view(1, input_step.size(0), self.embedding_size)
		except:
			embedded = embedded.view(1, 1, self.embedding_size)

		rnn_output, hidden = self.rnn(embedded, last_hidden)
		# Calculate attention weights from the current GRU output
		attn_weights = self.attn(rnn_output, encoder_outputs)
		# Multiply attention weights to encoder outputs to get new "weighted sum" context vector
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
		# Concatenate weighted context vector and GRU output using Luong eq. 5
		rnn_output = rnn_output.squeeze(0)
		context = context.squeeze(1)
		concat_input = torch.cat((rnn_output, context), 1)
		concat_output = F.relu(self.concat(concat_input))
		representation = concat_output
		# Predict next word using Luong eq. 6
		now_ops = self.know_net(encoder_outputs.transpose(0,1), self.ops, rnn_output.unsqueeze(1), attn_weights, word_word, word_op, word_exist_mat, seq_mask)
		op_output = (concat_output.unsqueeze(1) * now_ops).sum(-1) + self.ops_bias

		nop_output = self.nop_out(concat_output)
		output = F.log_softmax(torch.cat([op_output, nop_output], dim=-1), dim=1)
		# Return output and final hidden state
		return output, hidden, attn_weights, representation

class Know_Net(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout):
        super(Know_Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.ww_gcn = GCN(hidden_size * 2, hidden_size, hidden_size * 2, dropout)
        self.norm = LayerNorm1(hidden_size * 2)
        self.norm1 = LayerNorm1(hidden_size)
        self.norm2 = LayerNorm1(hidden_size)
        self.w_o = nn.Linear(hidden_size * 2, hidden_size)
        # self.w_o = nn.Linear(hidden_size, hidden_size)
        self.o_trans = nn.Linear(hidden_size * 2, hidden_size)
        self.o_output = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.dropout = dropout
    
    def normalize_matrix(self, matrix):
        diag = torch.sum(matrix, dim=-1, keepdims=True)
        return matrix / (diag+1e-30)

    def forward(self, encoder_outputs, ops, s, current_attn, word_word, word_op, word_exist_mat, seq_mask):                    
        # encoder_outputs: B x seq x N, ops: op_num x N
        # s: B x 1 x N
        # current_attn: B x 1 x seq
        # word_word: B x seq x seq x 2, word_op: B x seq x op x 2
        # word_exist_mat: B x seq x seq, seq_mask: B x seq
        
        batch_size = encoder_outputs.size(0)
        # s -> word
        s2w = current_attn.transpose(1,2) * s  # B x seq x N
        w_all = torch.cat([s2w, encoder_outputs], dim=-1)  
        ww_adj = word_word[:,:,:,0].squeeze(-1).masked_fill(word_exist_mat!=1, 0)
        w2w = self.ww_gcn(w_all, ww_adj)
        w2w = self.norm(w2w) + w_all  # B x seq x 2N
        # w2w = encoder_outputs
        
        # word -> operator
        wo_adj = word_op[:,:,:,0].squeeze(-1).transpose(1,2).masked_fill(seq_mask.unsqueeze(1).bool(), 0)  # B x op x seq 
        op_trans = F.relu(self.w_o(torch.matmul(self.normalize_matrix(wo_adj), w2w)))
        op_trans = F.dropout(op_trans, self.dropout, training=self.training)  # B x op x N
        op_trans = self.norm1(op_trans)

        op_all = torch.cat([op_trans, torch.unsqueeze(ops, 0).repeat(batch_size, 1, 1)], dim=-1)  # B x op x 2N
        op_h = F.relu(self.o_trans(op_all))
        op_o = self.norm2(op_h) + ops
        return op_o