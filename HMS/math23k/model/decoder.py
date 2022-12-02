# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

from .seq2seq import GCN, LayerNorm1
from .attention import get_mask, HierarchicalAttention

class GateNN(nn.Module):
    def __init__(self, hidden_size, input1_size, input2_size=0, dropout=0.4, single_layer=False):
        super(GateNN, self).__init__()
        self.single_layer = single_layer
        self.hidden_l1 = nn.Linear(input1_size+hidden_size, hidden_size)
        self.gate_l1 = nn.Linear(input1_size+hidden_size, hidden_size)
        if not single_layer:
            self.dropout = nn.Dropout(p=dropout)
            self.hidden_l2 = nn.Linear(input2_size+hidden_size, hidden_size)
            self.gate_l2 = nn.Linear(input2_size+hidden_size, hidden_size)
        return
    
    def forward(self, hidden, input1, input2=None):
        input1 = torch.cat((hidden, input1), dim=-1)
        h = torch.tanh(self.hidden_l1(input1))
        g = torch.sigmoid(self.gate_l1(input1))
        h = h * g
        if not self.single_layer:
            h1 = self.dropout(h)
            if input2 is not None:
                input2 = torch.cat((h1, input2), dim=-1)
            else:
                input2 = h1
            h = torch.tanh(self.hidden_l2(input2))
            g = torch.sigmoid(self.gate_l2(input2))
            h = h * g
        return h

class ScoreModel(nn.Module):
    def __init__(self, hidden_size):
        super(ScoreModel, self).__init__()
        self.w = nn.Linear(hidden_size * 3, hidden_size)
        self.score = nn.Linear(hidden_size, 1)
        return
    
    def forward(self, hidden, context, token_embeddings):
        # hidden/context: batch_size * hidden_size
        # token_embeddings: batch_size * class_size * hidden_size
        batch_size, class_size, _ = token_embeddings.size()
        hc = torch.cat((hidden, context), dim=-1)
        # (b, c, h)
        hc = hc.unsqueeze(1).expand(-1, class_size, -1)
        hidden = torch.cat((hc, token_embeddings), dim=-1)
        hidden = F.leaky_relu(self.w(hidden))
        score = self.score(hidden).view(batch_size, class_size)
        return score

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
        self.dropout = dropout
    
    def normalize_matrix(self, matrix):
        diag = torch.sum(matrix, dim=-1, keepdims=True)
        return matrix / (diag+1e-30)

    def forward(self, encoder_outputs, ops, s, current_attn, word_word, word_op, word_exist_mat, seq_mask):                    
        # encoder_outputs: B x seq x N, ops: B x op_num x N
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

        op_all = torch.cat([op_trans, ops], dim=-1)  # B x op x 2N
        op_h = F.relu(self.o_trans(op_all))
        op_o = self.norm2(op_h) + ops
        return op_o
        
class PredictModel(nn.Module):
    def __init__(self, hidden_size, class_size, dropout=0.4):
        super(PredictModel, self).__init__()
        self.class_size = class_size

        self.dropout = nn.Dropout(p=dropout)
        self.attn = HierarchicalAttention(hidden_size)
        
        self.score_pointer = ScoreModel(hidden_size)
        self.score_generator = ScoreModel(hidden_size)
        self.score_span = ScoreModel(hidden_size)
        self.gen_prob = nn.Linear(hidden_size*2, 1)
        
        self.know_net = Know_Net(hidden_size, hidden_size, dropout)
        return
    
    def score_pn(self, hidden, context, encoder_word_outputs, current_attn, embedding_masks, word_word, word_op, word_exist_mat, seq_mask):
        # embedding: batch_size * pointer_size * hidden_size
        # mask: batch_size * pointer_size
        (pointer_embedding, pointer_mask), generator_embedding, _ = embedding_masks
        hidden = self.dropout(hidden)
        context = self.dropout(context)
        pointer_embedding = self.dropout(pointer_embedding)
        pointer_score = self.score_pointer(hidden, context, pointer_embedding)
        pointer_score.data.masked_fill_(pointer_mask, -float('inf'))
        # batch_size * symbol_size
        # pointer
        pointer_prob = F.softmax(pointer_score, dim=-1)
        
        op_embeddings = generator_embedding[0] # batch_size * op_size * hidden_size
        now_ops = self.know_net(encoder_word_outputs, op_embeddings, hidden.unsqueeze(1), current_attn, word_word, word_op, word_exist_mat, seq_mask)
        generator_embeddings = torch.cat((now_ops, generator_embedding[1]), dim=1)
        # generator_embeddings = generator_embedding
        generator_embeddings = self.dropout(generator_embeddings)
        generator_score = self.score_generator(hidden, context, generator_embeddings)
        # batch_size * generator_size
        # generator
        generator_prob = F.softmax(generator_score, dim=-1)
        # batch_size * class_size, softmax
        return pointer_prob, generator_prob

    def forward(self, node_hidden, encoder_outputs, masks, embedding_masks, word_word, word_op):
        use_cuda = node_hidden.is_cuda
        node_hidden_dropout = self.dropout(node_hidden).unsqueeze(1)
        span_output, word_outputs = encoder_outputs
        span_mask, word_masks, seq_mask, word_exist_mat = masks
        output_attn, goal_word = self.attn(node_hidden_dropout, span_output, word_outputs, span_mask, word_masks)
        context = output_attn.squeeze(1)

        hc = torch.cat((node_hidden, context), dim=-1)
        # log(f(softmax(x)))
        # prob: softmax
        encoder_word_outputs = torch.cat(word_outputs, dim=1)
        pointer_prob, generator_prob = self.score_pn(node_hidden, context, encoder_word_outputs, goal_word.unsqueeze(1), embedding_masks, word_word, word_op, word_exist_mat, seq_mask)
        gen_prob = torch.sigmoid(self.gen_prob(hc))
        prob = torch.cat((gen_prob * generator_prob, (1 - gen_prob) * pointer_prob), dim=-1)
        # batch_size * class_size
        # generator + pointer + empty_pointer
        pad_empty_pointer = torch.zeros(prob.size(0), self.class_size - prob.size(-1))
        if use_cuda:
            pad_empty_pointer = pad_empty_pointer.cuda()
        prob = torch.cat((prob, pad_empty_pointer), dim=-1)
        output = torch.log(prob + 1e-30)
        return output, context

class TreeEmbeddingNode:
    def __init__(self, embedding, terminal):
        self.embedding = embedding
        self.terminal = terminal
        return

class TreeEmbeddingModel(nn.Module):
    def __init__(self, hidden_size, op_set, dropout=0.4):
        super(TreeEmbeddingModel, self).__init__()
        self.op_set = op_set
        self.dropout = nn.Dropout(p=dropout)
        self.combine = GateNN(hidden_size, hidden_size*2, dropout=dropout, single_layer=True)
        return
    
    def merge(self, op_embedding, left_embedding, right_embedding):
        te_input = torch.cat((left_embedding, right_embedding), dim=-1)
        te_input = self.dropout(te_input)
        op_embedding = self.dropout(op_embedding)
        tree_embed = self.combine(op_embedding, te_input)
        return tree_embed
    
    def forward(self, class_embedding, tree_stacks, embed_node_index):
        # embed_node_index: batch_size
        use_cuda = embed_node_index.is_cuda
        batch_index = torch.arange(embed_node_index.size(0))
        if use_cuda:
            batch_index = batch_index.cuda()
        labels_embedding = class_embedding[batch_index, embed_node_index]
        for node_label, tree_stack, label_embedding in zip(embed_node_index.cpu().tolist(), tree_stacks, labels_embedding):
            # operations
            if node_label in self.op_set:
                tree_node = TreeEmbeddingNode(label_embedding, terminal=False)
            # numbers
            else:
                right_embedding = label_embedding
                # on right tree => merge
                while len(tree_stack) >= 2 and tree_stack[-1].terminal and (not tree_stack[-2].terminal):
                    left_embedding = tree_stack.pop().embedding
                    op_embedding = tree_stack.pop().embedding
                    right_embedding = self.merge(op_embedding, left_embedding, right_embedding)
                tree_node = TreeEmbeddingNode(right_embedding, terminal=True)
            tree_stack.append(tree_node)
        return labels_embedding

class NodeEmbeddingNode:
    def __init__(self, node_hidden, node_context=None, label_embedding=None):
        self.node_hidden = node_hidden
        self.node_context = node_context
        self.label_embedding = label_embedding
        return

class DecomposeModel(nn.Module):
    def __init__(self, hidden_size, dropout=0.4, use_cuda=True):
        super(DecomposeModel, self).__init__()
        self.pad_hidden = torch.zeros(hidden_size)
        if use_cuda:
            self.pad_hidden = self.pad_hidden.cuda()

        self.dropout = nn.Dropout(p=dropout)
        self.l_decompose = GateNN(hidden_size, hidden_size*2, 0, dropout=dropout, single_layer=False)
        self.r_decompose = GateNN(hidden_size, hidden_size*2, hidden_size, dropout=dropout, single_layer=False)
        return

    def forward(self, node_stacks, tree_stacks, nodes_context, labels_embedding, pad_node=True):
        children_hidden = []
        for node_stack, tree_stack, node_context, label_embedding in zip(node_stacks, tree_stacks, nodes_context, labels_embedding):
            # start from encoder_hidden
            # len == 0 => finished decode
            if len(node_stack) > 0:
                # left
                if not tree_stack[-1].terminal:
                    node_hidden = node_stack[-1].node_hidden    # parent, still need for right
                    node_stack[-1] = NodeEmbeddingNode(node_hidden, node_context, label_embedding)   # add context and label of parent for right child
                    l_input = torch.cat((node_context, label_embedding), dim=-1)
                    l_input = self.dropout(l_input)
                    node_hidden = self.dropout(node_hidden)
                    child_hidden = self.l_decompose(node_hidden, l_input, None)
                    node_stack.append(NodeEmbeddingNode(child_hidden, None, None))  # only hidden for left child
                # right
                else:
                    node_stack.pop()    # left child, no need
                    if len(node_stack) > 0:
                        parent_node = node_stack.pop()  # parent, no longer need
                        node_hidden = parent_node.node_hidden
                        node_context = parent_node.node_context
                        label_embedding = parent_node.label_embedding
                        left_embedding = tree_stack[-1].embedding   # left tree
                        left_embedding = self.dropout(left_embedding)
                        r_input = torch.cat((node_context, label_embedding), dim=-1)
                        r_input = self.dropout(r_input)
                        node_hidden = self.dropout(node_hidden)
                        child_hidden = self.r_decompose(node_hidden, r_input, left_embedding)
                        node_stack.append(NodeEmbeddingNode(child_hidden, None, None))  # only hidden for right child
                    # else finished decode
            # finished decode, pad
            if len(node_stack) == 0:
                child_hidden = self.pad_hidden
                if pad_node:
                    node_stack.append(NodeEmbeddingNode(child_hidden, None, None))
            children_hidden.append(child_hidden)
        children_hidden = torch.stack(children_hidden, dim=0)
        return children_hidden

def copy_list(src_list):
    dst_list = [copy_list(item) if type(item) is list else item for item in src_list]
    return dst_list

class BeamNode:
    def __init__(self, score, nodes_hidden, node_stacks, tree_stacks, decoder_outputs_list, sequence_symbols_list):
        self.score = score
        self.nodes_hidden = nodes_hidden
        self.node_stacks = node_stacks
        self.tree_stacks = tree_stacks
        self.decoder_outputs_list = decoder_outputs_list
        self.sequence_symbols_list = sequence_symbols_list
        return
    
    def copy(self):
        node = BeamNode(
            self.score,
            self.nodes_hidden,
            copy_list(self.node_stacks),
            copy_list(self.tree_stacks),
            copy_list(self.decoder_outputs_list),
            copy_list(self.sequence_symbols_list)
        )
        return node

class Decoder(nn.Module):
    def __init__(self, embed_model, op_set, vocab_dict, class_list, hidden_size=512, dropout=0.4, use_cuda=True):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        embed_size = embed_model.embedding_dim
        class_size = len(class_list)
        
        self.op_set = op_set
        # 128 => 512
        self.op_hidden = nn.Linear(embed_size, hidden_size)
        self.embed_model = embed_model
        self.get_predict_meta(class_list, vocab_dict)

        self.predict = PredictModel(hidden_size, class_size, dropout=dropout)
        
        op_set = set(i for i, symbol in enumerate(class_list) if symbol in op_set)
        self.tree_embedding = TreeEmbeddingModel(hidden_size, op_set, dropout=dropout)
        self.decompose = DecomposeModel(hidden_size, dropout=dropout, use_cuda=use_cuda)
        return

    def get_predict_meta(self, class_list, vocab_dict):
        # embed order: generator + pointer, with original order
        # used in predict_model, tree_embedding
        pointer_list = [token for token in class_list if 'temp_' in token]
        generator_nop_list = [token for token in class_list if token not in pointer_list and token not in self.op_set]
        generator_op_list = [token for token in class_list if token not in pointer_list and token in self.op_set]
        generator_list = generator_op_list + generator_nop_list
        embed_list = generator_list + pointer_list  

        # pointer num index in class_list, for select only num pos from num_pos with op pos
        self.pointer_index = torch.LongTensor([class_list.index(token) for token in pointer_list])
        # generator symbol index in vocab, for generator symbol embedding
        self.generator_vocab = torch.LongTensor([vocab_dict[token] for token in generator_list])
        self.generator_nop_vocab = torch.LongTensor([vocab_dict[token] for token in generator_nop_list])
        self.generator_op_vocab = torch.LongTensor([vocab_dict[token] for token in generator_op_list])
        # class_index -> embed_index, for tree embedding
        # embed order -> class order, for predict_model output
        self.class_to_embed_index = torch.LongTensor([embed_list.index(token) for token in class_list])
        # self.generator_op_embedding = self.op_hidden(self.embed_model(self.generator_op_vocab))
        if self.use_cuda:
            self.pointer_index = self.pointer_index.cuda()
            self.generator_vocab = self.generator_vocab.cuda()
            self.generator_nop_vocab = self.generator_nop_vocab.cuda()
            self.generator_op_vocab = self.generator_op_vocab.cuda()
            self.class_to_embed_index = self.class_to_embed_index.cuda()
            # self.generator_op_embedding = self.generator_op_embedding.cuda()
        return

    def get_pad_masks(self, encoder_outputs, input_lengths, span_length=None):
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
        masks = (span_mask, word_masks, 1-word_exist_sequence, word_exist_matrix * torch.transpose(word_exist_matrix,1,2))
        return masks

    def get_pointer_meta(self, num_pos, sub_num_poses=None):
        batch_size = num_pos.size(0)
        pointer_num_pos = num_pos.index_select(dim=1, index=self.pointer_index)
        num_pos_occupied = pointer_num_pos.sum(dim=0) == -batch_size
        occupied_len = num_pos_occupied.size(-1)
        for i, elem in enumerate(reversed(num_pos_occupied.cpu().tolist())):
            if not elem:
                occupied_len = occupied_len - i
                break
        pointer_num_pos = pointer_num_pos[:, :occupied_len]
        # length of word_num_poses determined by span_num_pos
        if sub_num_poses is not None:
            sub_pointer_poses = [sub_num_pos.index_select(dim=1, index=self.pointer_index)[:, :occupied_len] for sub_num_pos in sub_num_poses]
        else:
            sub_pointer_poses = None
        return pointer_num_pos, sub_pointer_poses

    def get_pointer_embedding(self, pointer_num_pos, encoder_outputs):
        # encoder_outputs: batch_size * seq_len * hidden_size
        # pointer_num_pos: batch_size * pointer_size
        # subset of num_pos, invalid pos -1
        batch_size, pointer_size = pointer_num_pos.size()
        batch_index = torch.arange(batch_size)
        if self.use_cuda:
            batch_index = batch_index.cuda()
        batch_index = batch_index.unsqueeze(1).expand(-1, pointer_size)
        # batch_size * pointer_len * hidden_size
        pointer_embedding = encoder_outputs[batch_index, pointer_num_pos]
        # mask invalid pos -1
        pointer_embedding = pointer_embedding * (pointer_num_pos != -1).unsqueeze(-1)
        return pointer_embedding
    
    def get_pointer_mask(self, pointer_num_pos):
        # pointer_num_pos: batch_size * pointer_size
        # subset of num_pos, invalid pos -1
        pointer_mask = pointer_num_pos == -1
        return pointer_mask
    
    def get_generator_embedding_mask(self, batch_size):
        # generator_size * hidden_size
        generator_nop_embedding = self.op_hidden(self.embed_model(self.generator_nop_vocab))
        generator_op_embedding = self.op_hidden(self.embed_model(self.generator_op_vocab))
        # batch_size * generator_size * hidden_size
        generator_nop_embedding = generator_nop_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        generator_op_embedding = generator_op_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        generator_embedding = (generator_op_embedding, generator_nop_embedding)
        # generator_embedding = self.op_hidden(self.embed_model(self.generator_vocab))
        # generator_embedding = generator_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        # batch_size * generator_size
        generator_mask = (self.generator_vocab == -1).unsqueeze(0).expand(batch_size, -1)
        return generator_embedding, generator_mask
    
    def get_class_embedding_mask(self, num_pos, encoder_outputs):
        # embedding: batch_size * size * hidden_size
        # mask: batch_size * size
        _, word_outputs = encoder_outputs
        span_num_pos, word_num_poses = num_pos
        generator_embedding, generator_mask = self.get_generator_embedding_mask(span_num_pos.size(0))
        span_pointer_num_pos, word_pointer_num_poses = self.get_pointer_meta(span_num_pos, word_num_poses)
        pointer_mask = self.get_pointer_mask(span_pointer_num_pos)
        num_pointer_embeddings = []
        for word_output, word_pointer_num_pos in zip(word_outputs, word_pointer_num_poses):
            num_pointer_embedding = self.get_pointer_embedding(word_pointer_num_pos, word_output)
            num_pointer_embeddings.append(num_pointer_embedding)
        pointer_embedding = torch.cat([embedding.unsqueeze(0) for embedding in num_pointer_embeddings], dim=0).sum(dim=0)
        
        all_embedding = torch.cat((generator_embedding[0], generator_embedding[1], pointer_embedding), dim=1)
        # all_embedding = torch.cat((generator_embedding, pointer_embedding), dim=1)
        pointer_embedding_mask = (pointer_embedding, pointer_mask)
        return pointer_embedding_mask, generator_embedding, all_embedding

    def init_stacks(self, encoder_hidden):
        batch_size = encoder_hidden.size(0)
        node_stacks = [[NodeEmbeddingNode(hidden, None, None)] for hidden in encoder_hidden]
        tree_stacks = [[] for _ in range(batch_size)]
        return node_stacks, tree_stacks

    def forward_step(self, node_stacks, tree_stacks, nodes_hidden, encoder_outputs, masks, embedding_masks, word_word=None, word_op=None, decoder_nodes_class=None):
        nodes_output, nodes_context = self.predict(nodes_hidden, encoder_outputs, masks, embedding_masks, word_word, word_op)
        nodes_output = nodes_output.index_select(dim=-1, index=self.class_to_embed_index)
        predict_nodes_class = nodes_output.topk(1)[1]
        # teacher
        if decoder_nodes_class is not None:
            nodes_class = decoder_nodes_class.view(-1)
        # no teacher
        else:
            nodes_class = predict_nodes_class.view(-1)
        embed_nodes_index = self.class_to_embed_index[nodes_class]
        labels_embedding = self.tree_embedding(embedding_masks[-1], tree_stacks, embed_nodes_index)
        nodes_hidden = self.decompose(node_stacks, tree_stacks, nodes_context, labels_embedding)
        return nodes_output, predict_nodes_class, nodes_hidden
    
    def forward_teacher(self, decoder_nodes_label, decoder_init_hidden, encoder_outputs, masks, embedding_masks, max_length=None, word_word=None, word_op=None):
        decoder_outputs_list = []
        sequence_symbols_list = []
        decoder_hidden = decoder_init_hidden
        node_stacks, tree_stacks = self.init_stacks(decoder_init_hidden)
        if decoder_nodes_label is not None:
            seq_len = decoder_nodes_label.size(1)
        else:
            seq_len = max_length
        for di in range(seq_len):
            if decoder_nodes_label is not None:
                decoder_node_class = decoder_nodes_label[:, di]
            else:
                decoder_node_class = None
            decoder_output, symbols, decoder_hidden = self.forward_step(node_stacks, tree_stacks, decoder_hidden, encoder_outputs, masks, embedding_masks, word_word=word_word, word_op=word_op, decoder_nodes_class=decoder_node_class)
            decoder_outputs_list.append(decoder_output)
            sequence_symbols_list.append(symbols)
        return decoder_outputs_list, decoder_hidden, sequence_symbols_list

    def forward_beam(self, decoder_init_hidden, encoder_outputs, masks, embedding_masks, max_length, beam_width=1):
        # only support batch_size == 1
        node_stacks, tree_stacks = self.init_stacks(decoder_init_hidden)
        beams = [BeamNode(0, decoder_init_hidden, node_stacks, tree_stacks, [], [])]
        for _ in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                # finished stack-guided decoding
                if len(b.node_stacks) == 0:
                    current_beams.append(b)
                    continue
                # unfinished decoding
                # batch_size == 1
                # batch_size * class_size
                nodes_output, nodes_context = self.predict(b.nodes_hidden, encoder_outputs, masks, embedding_masks)
                nodes_output = nodes_output.index_select(dim=-1, index=self.class_to_embed_index)
                # batch_size * beam_width
                top_value, top_index = nodes_output.topk(beam_width)
                top_value = torch.exp(top_value)
                for predict_score, predicted_symbol in zip(top_value.split(1, dim=-1), top_index.split(1, dim=-1)):
                    nb = b.copy()
                    embed_nodes_index = self.class_to_embed_index[predicted_symbol.view(-1)]
                    labels_embedding = self.tree_embedding(embedding_masks[-1], nb.tree_stacks, embed_nodes_index)
                    nodes_hidden = self.decompose(nb.node_stacks, nb.tree_stacks, nodes_context, labels_embedding, pad_node=False)

                    nb.score = b.score + predict_score.item()
                    nb.nodes_hidden = nodes_hidden
                    nb.decoder_outputs_list.append(nodes_output)
                    nb.sequence_symbols_list.append(predicted_symbol)
                    current_beams.append(nb)
            beams = sorted(current_beams, key=lambda b:b.score, reverse=True)
            beams = beams[:beam_width]
            all_finished = True
            for b in beams:
                if len(b.node_stacks[0]) != 0:
                    all_finished = False
                    break
            if all_finished:
                break
        output = beams[0]
        return output.decoder_outputs_list, output.nodes_hidden, output.sequence_symbols_list

    def forward(self, targets=None, encoder_hidden=None, encoder_outputs=None, input_lengths=None, span_length=None, num_pos=None, word_word=None, word_op=None, max_length=None, beam_width=None):
        masks = self.get_pad_masks(encoder_outputs, input_lengths, span_length)
        embedding_masks = self.get_class_embedding_mask(num_pos, encoder_outputs)

        if type(encoder_hidden) is tuple:
            encoder_hidden = encoder_hidden[0]
        decoder_init_hidden = encoder_hidden[-1,:,:]

        if max_length is None:
            if targets is not None:
                max_length = targets.size(1)
            else:
                max_length = 40
        
        if beam_width is not None:
            return self.forward_beam(decoder_init_hidden, encoder_outputs, masks, embedding_masks, max_length, beam_width)
        else:
            return self.forward_teacher(targets, decoder_init_hidden, encoder_outputs, masks, embedding_masks, max_length, word_word, word_op)
