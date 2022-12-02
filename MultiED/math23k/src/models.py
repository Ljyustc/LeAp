import copy
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
from .knowledge_base import *

def replace_masked_values(tensor, mask, replace_with):
    return tensor.masked_fill((1 - mask).bool(), replace_with)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        # S x B x H
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, embedding_size, input_size, output_size, n_layers=2, op_nums=5, dropout=0.5):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.em_dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        # op_nums: -,*,+,/,^
        self.ops = nn.Parameter(torch.nn.init.uniform_(torch.empty(op_nums, hidden_size), a=-1/hidden_size, b=1/hidden_size))
        self.ops_bias = nn.Parameter(torch.nn.init.uniform_(torch.empty(op_nums), a=-1/hidden_size, b=1/hidden_size))
        self.nop_out = nn.Linear(self.hidden_size, self.output_size-op_nums)
        
        self.out = nn.Linear(hidden_size, output_size)
        
        # Choose attention model
        self.attn = Attn(hidden_size)
        
        self.know_net = Know_Net(hidden_size, hidden_size, hidden_size, dropout)

    def forward(self, input_seq, last_hidden, encoder_outputs, word_word, word_op, word_exist_mat, seq_mask):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.em_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_size)  # S=1 x B x N

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        
        # temp_output = torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1)))
        
        # now_ops = self.know_net(encoder_outputs.transpose(0,1), self.ops, rnn_output.transpose(0,1), attn_weights, word_word, word_op, word_exist_mat, seq_mask)
        # op_output = (temp_output.unsqueeze(1) * now_ops).sum(-1) + self.ops_bias

        # nop_output = self.nop_out(temp_output)
        
        # output = torch.cat([op_output, nop_output], dim=-1)
        
        output = self.out(torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

        # Return final output, hidden state
        return output, hidden


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask.bool(), -1e12)
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)


class EncoderSeq(nn.Module):
    def __init__(self, input1_size, input2_size, embed_model, embedding1_size, 
                 embedding2_size, hidden_size, n_layers=2, hop_size=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.embedding1_size = embedding1_size
        self.embedding2_size = embedding2_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.hop_size = hop_size

        self.embedding1 = embed_model
        self.embedding2 = nn.Embedding(input2_size, embedding2_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding1_size+embedding2_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.parse_gnn = clones(Parse_Graph_Module(hidden_size), hop_size)
        
        self.ww_encoder = KnowEncoder(embedding1_size*2, hidden_size, 2)
        self.wo_encoder = KnowEncoder1(embedding1_size + hidden_size*2, hidden_size, 2)
        self.ww_gcn = GCN(hidden_size, hidden_size, hidden_size, dropout=0.5)
        self.norm = LayerNorm1(hidden_size)
        self.ww_trans = PositionwiseFeedForward(hidden_size, hidden_size, hidden_size, dropout=0.5)

    def encode_prior(self, input_seqs, ops_embed):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding1(input_seqs)  # S x B x E

        all_word = embedded.transpose(0, 1)
        _, ww_prob = self.ww_encoder(all_word, 0.5, True, 0.5)

        _, wo_prob = self.wo_encoder(all_word, ops_embed, 0.5, True, 0.5)
        return ww_prob, wo_prob
        
    def forward(self, input1_var, input2_var, input_length, parse_graph, word_exist_mat, ops_embed, temp=0.5, thre=0.5, hidden=None, hard=False):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded1 = self.embedding1(input1_var)  # S x B x E
        
        all_word = embedded1.transpose(0, 1)
        word_word, ww_prob = self.ww_encoder(all_word, temp, hard, thre)
        word_op, wo_prob = self.wo_encoder(all_word, ops_embed, temp, hard, thre)
        
        embedded2 = self.embedding2(input2_var)
        embedded = torch.cat((embedded1, embedded2), dim=2)
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_length)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        pade_outputs = pade_outputs.transpose(0, 1)
        for i in range(self.hop_size):
            pade_outputs = self.parse_gnn[i](pade_outputs, parse_graph[:,2])
        
        ww_adj = word_word[:,:,:,0].squeeze(-1).masked_fill(word_exist_mat!=1, 0)
        # ww_adj = word_word[:,:,:,0].squeeze(-1) * word_exist_mat
        pade_outputs_1 = self.ww_gcn(pade_outputs, ww_adj)
        pade_outputs_2 = self.norm(pade_outputs_1) + pade_outputs
        pade_outputs = self.ww_trans(pade_outputs_2) + pade_outputs_2
        
        pade_outputs = pade_outputs.transpose(0, 1)
#        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        
        return pade_outputs, pade_hidden, word_word, ww_prob, word_op, wo_prob 


class Parse_Graph_Module(nn.Module):
    def __init__(self, hidden_size):
        super(Parse_Graph_Module, self).__init__()
        
        self.hidden_size = hidden_size
        self.node_fc1 = nn.Linear(hidden_size, hidden_size)
        self.node_fc2 = nn.Linear(hidden_size, hidden_size)
        self.node_out = nn.Linear(hidden_size * 2, hidden_size)
    
    def normalize(self, graph, symmetric=True):
        d = graph.sum(1)
        if symmetric:
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(graph).mm(D)
        else :
            D = torch.diag(torch.pow(d,-1))
            return D.mm(graph)
        
    def forward(self, node, graph):
        graph = graph.float()
        batch_size = node.size(0)
        for i in range(batch_size):
            graph[i] = self.normalize(graph[i])
        
        node_info = torch.relu(self.node_fc1(torch.matmul(graph, node)))
        node_info = torch.relu(self.node_fc2(torch.matmul(graph, node_info)))
        
        agg_node_info = torch.cat((node, node_info), dim=2)
        agg_node_info = torch.relu(self.node_out(agg_node_info))
        
        return agg_node_info


class NumEncoder(nn.Module):
    def __init__(self, node_dim, hop_size=2):
        super(NumEncoder, self).__init__()
        
        self.node_dim = node_dim
        self.hop_size = hop_size
        self.num_gnn = clones(Num_Graph_Module(node_dim), hop_size)

    def forward(self, encoder_outputs, num_encoder_outputs, num_pos_pad, num_order_pad):
        num_embedding = num_encoder_outputs.clone()
        batch_size = num_embedding.size(0)
        num_mask = (num_pos_pad > -1).long()
        node_mask = (num_order_pad > 0).long()
        greater_graph_mask = num_order_pad.unsqueeze(-1).expand(batch_size, -1, num_order_pad.size(-1)) > \
                        num_order_pad.unsqueeze(1).expand(batch_size, num_order_pad.size(-1), -1)
        lower_graph_mask = num_order_pad.unsqueeze(-1).expand(batch_size, -1, num_order_pad.size(-1)) <= \
                        num_order_pad.unsqueeze(1).expand(batch_size, num_order_pad.size(-1), -1)
        greater_graph_mask = greater_graph_mask.long()
        lower_graph_mask = lower_graph_mask.long()
        
        diagmat = torch.diagflat(torch.ones(num_embedding.size(1), dtype=torch.long, device=num_embedding.device))
        diagmat = diagmat.unsqueeze(0).expand(num_embedding.size(0), -1, -1)
        graph_ = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1-diagmat)
        graph_greater = graph_ * greater_graph_mask + diagmat
        graph_lower = graph_ * lower_graph_mask + diagmat
        
        for i in range(self.hop_size):
            num_embedding = self.num_gnn[i](num_embedding, graph_greater, graph_lower)
        
#        gnn_info_vec = torch.zeros((batch_size, 1, encoder_outputs.size(-1)),
#                                   dtype=torch.float, device=num_embedding.device)
#        gnn_info_vec = torch.cat((encoder_outputs.transpose(0, 1), gnn_info_vec), dim=1)
        gnn_info_vec = torch.zeros((batch_size, encoder_outputs.size(0)+1, encoder_outputs.size(-1)),
                                   dtype=torch.float, device=num_embedding.device)
        clamped_number_indices = replace_masked_values(num_pos_pad, num_mask, gnn_info_vec.size(1)-1)
        gnn_info_vec.scatter_(1, clamped_number_indices.unsqueeze(-1).expand(-1, -1, num_embedding.size(-1)), num_embedding)
        gnn_info_vec = gnn_info_vec[:, :-1, :]
        gnn_info_vec = gnn_info_vec.transpose(0, 1)
        gnn_info_vec = encoder_outputs + gnn_info_vec
        num_embedding = num_encoder_outputs + num_embedding
        problem_output = torch.max(gnn_info_vec, 0).values

        return gnn_info_vec, num_embedding, problem_output


class Num_Graph_Module(nn.Module):
    def __init__(self, node_dim):
        super(Num_Graph_Module, self).__init__()
        
        self.node_dim = node_dim
        self.node1_fc1 = nn.Linear(node_dim, node_dim)
        self.node1_fc2 = nn.Linear(node_dim, node_dim)
        self.node2_fc1 = nn.Linear(node_dim, node_dim)
        self.node2_fc2 = nn.Linear(node_dim, node_dim)
        self.graph_weight = nn.Linear(node_dim * 4, node_dim)
        self.node_out = nn.Linear(node_dim * 2, node_dim)
    
    def normalize(self, graph, symmetric=True):
        d = graph.sum(1)
        if symmetric:
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(graph).mm(D)
        else :
            D = torch.diag(torch.pow(d,-1))
            return D.mm(graph)

    def forward(self, node, graph1, graph2):
        graph1 = graph1.float()
        graph2 = graph2.float()
        batch_size = node.size(0)
        
        for i in range(batch_size):
            graph1[i] = self.normalize(graph1[i], False)
            graph2[i] = self.normalize(graph2[i], False)
        
        node_info1 = torch.relu(self.node1_fc1(torch.matmul(graph1, node)))
        node_info1 = torch.relu(self.node1_fc2(torch.matmul(graph1, node_info1)))
        node_info2 = torch.relu(self.node2_fc1(torch.matmul(graph2, node)))
        node_info2 = torch.relu(self.node2_fc2(torch.matmul(graph2, node_info2)))
        gate = torch.cat((node_info1, node_info2, node_info1+node_info2, node_info1-node_info2), dim=2)
        gate = torch.sigmoid(self.graph_weight(gate))
        node_info = gate * node_info1 + (1-gate) * node_info2
        agg_node_info = torch.cat((node, node_info), dim=2)
        agg_node_info = torch.relu(self.node_out(agg_node_info))
        
        return agg_node_info


class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        # self.ops = nn.Linear(hidden_size * 2, op_nums)
        self.ops = nn.Parameter(torch.nn.init.uniform_(torch.empty(op_nums, hidden_size * 2), a=-1/(hidden_size * 2), b=1/(hidden_size * 2)))
        self.ops_bias = nn.Parameter(torch.nn.init.uniform_(torch.empty(op_nums), a=-1/(hidden_size * 2), b=1/(hidden_size * 2)))

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)
        
        self.know_net = Know_Net(hidden_size, hidden_size, hidden_size*2, dropout)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, word_word, word_op, word_exist_mat, seq_mask, mask_nums):
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)

        # num_score = nn.functional.softmax(num_score, 1)

        now_ops = self.know_net(encoder_outputs.transpose(0,1), self.ops, current_embeddings, current_attn, word_word, word_op, word_exist_mat, seq_mask)
        op = (leaf_input.unsqueeze(1) * now_ops).sum(-1) + self.ops_bias

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


class Know_Net(nn.Module):
    def __init__(self, input_dim, hidden_size, output_size, dropout):
        super(Know_Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.ww_gcn = GCN(hidden_size * 2, hidden_size, hidden_size * 2, dropout)
        self.norm = LayerNorm1(hidden_size * 2)
        self.norm1 = LayerNorm1(hidden_size)
        self.norm2 = LayerNorm1(output_size)
        self.w_o = nn.Linear(hidden_size * 2, hidden_size)
        # self.w_o = nn.Linear(hidden_size, hidden_size)
        self.o_trans = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = dropout
    
    def normalize_matrix(self, matrix):
        diag = torch.sum(matrix, dim=-1, keepdims=True)
        return matrix / (diag+1e-30)

    def forward(self, encoder_outputs, ops, s, current_attn, word_word, word_op, word_exist_mat, seq_mask):                    
        # encoder_outputs: B x seq x N, ops: op_num x o_N
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
        op_trans = F.relu(self.w_o(torch.matmul(wo_adj, w2w)))
        op_trans = F.dropout(op_trans, self.dropout, training=self.training)  # B x op x N
        op_trans = self.norm1(op_trans)
        
        op_all = torch.cat([op_trans, torch.unsqueeze(ops, 0).repeat(batch_size, 1, 1)], dim=-1)  # B x op x 3N
        op_h = F.relu(self.o_trans(op_all))
        op_o = self.norm2(op_h) + ops
        return op_o

# Graph Module
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

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