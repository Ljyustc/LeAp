# coding: utf-8
import os
import logging
from src.train_and_evaluate import *
from src.models import *
from src.config import args
import time
import torch
import torch.optim
from src.expressions_transfer import *
import json
import pickle as pkl

def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x
    
def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

def write_json(path,file):
    with open(path,'w') as f:
        json.dump(file,f)
        
def write_pkl(path,file):
    with open(path,'w') as f:
        pkl.dump(file,f)

batch_size = 64
embedding_size = 128
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2
prior_prob = args.prior_prob
kr = args.kr
ori_path = './data/'
prefix = '23k_processed.json'

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
USE_CUDA = torch.cuda.is_available()
# occumpy_mem(args.cuda_id)

def get_train_test_fold(ori_path,prefix,data,pairs,group):
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path = ori_path + mode_test + prefix
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []
    for item,pair,g in zip(data, pairs, group):
        pair = list(pair)
        pair.append(g['group_num'])
        pair = tuple(pair)
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold

def change_num(num):
    new_num = []
    for item in num:
        if '/' in item:
            new_str = item.split(')')[0]
            new_str = new_str.split('(')[1]
            a = float(new_str.split('/')[0])
            b = float(new_str.split('/')[1])
            value = a/b
            new_num.append(value)
        elif '%' in item:
            value = float(item[0:-1])/100
            new_num.append(value)
        else:
            new_num.append(float(item))
    return new_num


data = load_raw_data("data/Math_23K.json")
group_data = read_json("data/Math_23K_processed.json")

data = load_raw_data("data/Math_23K.json")

pairs, generate_nums, copy_nums = transfer_num(data)

temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs = temp_pairs

train_fold, test_fold, valid_fold = get_train_test_fold(ori_path,prefix,data,pairs,group_data)

'''
fold_size = int(len(pairs) * 0.2)
fold_pairs = []
for split_fold in range(4):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(pairs[fold_start:fold_end])
fold_pairs.append(pairs[(fold_size * 4):])
'''

best_acc_fold = []

pairs_tested = test_fold
pairs_valided = valid_fold
pairs_trained = train_fold
#pairs_trained = valid_fold

#for fold_t in range(5):
#    if fold_t == fold:
#        pairs_tested += fold_pairs[fold_t]
#    else:
#        pairs_trained += fold_pairs[fold_t]

input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                copy_nums, tree=True)
#_, _, _, valid_pairs = prepare_data(pairs_trained, pairs_valided, 5, generate_nums,copy_nums, tree=True)

#print('train_pairs[0]')
#print(train_pairs[0])
#exit()
# Initialize models
encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                     n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)

predict_1 = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate_1 = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge_1 = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
# the embedding layer is  only for generated number embeddings, operators, and paddings

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
predict_1_optimizer = torch.optim.Adam(predict_1.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_1_optimizer = torch.optim.Adam(generate_1.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_1_optimizer = torch.optim.Adam(merge_1.parameters(), lr=learning_rate, weight_decay=weight_decay)


encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)
predict_1_scheduler = torch.optim.lr_scheduler.StepLR(predict_1_optimizer, step_size=20, gamma=0.5)
generate_1_scheduler = torch.optim.lr_scheduler.StepLR(generate_1_optimizer, step_size=20, gamma=0.5)
merge_1_scheduler = torch.optim.lr_scheduler.StepLR(merge_1_optimizer, step_size=20, gamma=0.5)

# ground_truth knowledge
gt_know = np.load("data/math23k_know.npy")
gt_know = gt_know.tolist()

# Select ground_truth knowledge in input_lang
common_dict = {}
common_word_count = 0
eval_know_inputs = []
for triples in gt_know:
    w1, w2, rel = triples[0], triples[1], triples[2]
    if rel not in ['error']:
        if w1 in input_lang.word2index and w2 in input_lang.word2index:
            if input_lang.word2index[w1] not in common_dict:
                common_dict[input_lang.word2index[w1]] = common_word_count
                common_word_count += 1
                eval_know_inputs.append(input_lang.word2index[w1])
            if input_lang.word2index[w2] not in common_dict:
                common_dict[input_lang.word2index[w2]] = common_word_count
                common_word_count += 1
                eval_know_inputs.append(input_lang.word2index[w2])
logging.info(f"common_word_num: {common_word_count}")
gt_ww = torch.zeros((common_word_count, common_word_count))
for triples in gt_know:
    w1, w2, rel = triples[0], triples[1], triples[2]
    if rel not in ['error']:
        if w1 in input_lang.word2index and w2 in input_lang.word2index:
            gt_ww[common_dict[input_lang.word2index[w1]], common_dict[input_lang.word2index[w2]]] = 1
eval_know_inputs = torch.LongTensor(eval_know_inputs)

know_gt_ww = torch.zeros((common_word_count, common_word_count))
for i in range(common_word_count):
    i_true = torch.where(gt_ww[i]!=0)[0].tolist()
    keep_true = random.sample(i_true, int(kr*len(i_true)))
    know_gt_ww[i][keep_true] = 1
torch.save(know_gt_ww, str(0)+"_know_gt_ww.pt")

# prior of unknown knowledge
prior = np.array([prior_prob, 1-prior_prob])  # TODO: batch * word * word * 2
log_prior = torch.FloatTensor(np.log(prior))
log_prior = torch.unsqueeze(log_prior, 0)
log_prior = torch.unsqueeze(log_prior, 0)
log_prior = torch.unsqueeze(log_prior, 0)

# known knowledge
gt_prior = np.array([0.5, 0.5])  
log_gt_prior = torch.FloatTensor(np.log(gt_prior))
log_gt_prior = torch.unsqueeze(log_gt_prior, 0)
log_gt_prior = torch.unsqueeze(log_gt_prior, 0)
log_gt_prior = torch.unsqueeze(log_gt_prior, 0)

log_prior = Variable(log_prior)
log_gt_prior = Variable(log_gt_prior)
    
# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()
    predict_1.cuda()
    generate_1.cuda()
    merge_1.cuda()
    log_prior = log_prior.cuda()
    log_gt_prior = log_gt_prior.cuda()
    gt_ww = gt_ww.cuda()
    know_gt_ww = know_gt_ww.cuda()
    eval_know_inputs = eval_know_inputs.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

sftarget_path = 'data/best_node_seq_diverse.pt'
node = torch.load(sftarget_path)
node_idx_path = 'data/best_idx_seq_diverse.json'
node_idx = read_json(node_idx_path)
sf_dic = get_softtarget_dic(get_softtarget_list(node), node_idx)
mask_path = 'data/mask.pt'
encoder_mask = torch.cuda.FloatTensor(batch_size,115,hidden_size).uniform_() < 0.99
encoder_mask = encoder_mask.float()
torch.save(encoder_mask,mask_path)

def get_alpha(alpha,epoch):
    if epoch % anneal ==0 and epoch != 0 and epoch < bound:
        alpha = alpha+0.02
    elif epoch == 0:
        alpha = alpha
    elif epoch % anneal == 0:
        alpha = alpha/2
    return alpha
anneal = 10
bound = 30
alpha = 0.15
best_acc = 0
best_path = './data/pg_seq_'
best_node_path = './data/node_seq_'
best_idx_path = './data/idx_seq_'
for epoch in range(n_epochs):
    temp = args.temp - (args.temp - 0.1) / (n_epochs-1) * epoch
    prior_lam = 0.1
    gt_prior_lam = 0.1
    encoder_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()
    predict_1_scheduler.step()
    generate_1_scheduler.step()
    merge_1_scheduler.step()
    loss_total = 0
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, \
   num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, graph_batches, idx_batches = prepare_train_batch(train_pairs, batch_size)
    print("epoch:", epoch + 1)
    start = time.time()
    all_node_out = []
    #alpha = get_alpha(alpha,epoch)
    for idx in range(len(input_lengths)):
        loss = train_tree(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
            encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang, num_pos_batches[idx], graph_batches[idx],idx_batches[idx],sf_dic, predict_1, generate_1, merge_1, predict_1_optimizer, generate_1_optimizer, merge_1_optimizer, encoder_mask, alpha, temp=temp, log_prior=log_prior, log_gt_prior=log_gt_prior, eval_know_inputs=eval_know_inputs, know_gt_ww=know_gt_ww, common_dict=common_dict, prior_lam=prior_lam, gt_prior_lam=gt_prior_lam)
        loss_total += loss
        #all_node_out.append(node_out)

    print("loss:", loss_total / len(input_lengths))
    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    '''
    if epoch % 5 == 0 or epoch > n_epochs - 5:
        value_ac = 0
        value_ac_0 = 0
        value_ac_1 = 0
        value_ac_gate = 0
        equation_ac = 0
        equation_ac_0 = 0
        equation_ac_1 = 0
        equation_ac_gate = 0
        eval_total = 0
        start = time.time()
        record = []
        for test_batch in valid_pairs:
            #print(test_batch)
            batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5])
            test_res,res_score,test_res_1,res_score_1 = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate, merge, output_lang, test_batch[5], batch_graph, predict_1, generate_1, merge_1, beam_size=1)
            val_ac, equ_ac, test, tar = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            val_ac_1, equ_ac_1, test_1, tar_1 = compute_prefix_tree_result(test_res_1, test_batch[2], output_lang, test_batch[4], test_batch[6])
            record.append([[test,tar],[test_1,tar_1]])
            if val_ac or val_ac_1:
                value_ac += 1
            if equ_ac or equ_ac_1:
                equation_ac += 1
            if val_ac:
                value_ac_0 += 1
            if equ_ac:
                equation_ac_0 += 1
            if val_ac_1:
                value_ac_1 += 1
            if equ_ac_1:
                equation_ac_1 += 1
            if (val_ac and res_score > res_score_1) or (val_ac_1 and res_score <= res_score_1):
                value_ac_gate += 1
            if (equ_ac and res_score > res_score_1) or (equ_ac_1 and res_score <= res_score_1):
                equation_ac_gate += 1
            eval_total += 1
        print(equation_ac, value_ac, equation_ac_0, value_ac_0, equation_ac_1, value_ac_1, eval_total)
        print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("decoder1 test_answer_acc", float(equation_ac_0) / eval_total, float(value_ac_0) / eval_total)
        print("decoder2 test_answer_acc", float(equation_ac_1) / eval_total, float(value_ac_1) / eval_total)
        print("test_answer_acc", float(equation_ac_gate) / eval_total, float(value_ac_gate) / eval_total)
        print("testing time", time_since(time.time() - start))
        print("------------------------------------------------------")
    '''
    if epoch % 5 == 0 or epoch > n_epochs - 5:
        value_ac = 0
        value_ac_0 = 0
        value_ac_1 = 0
        value_ac_gate = 0
        equation_ac = 0
        equation_ac_0 = 0
        equation_ac_1 = 0
        equation_ac_gate = 0
        eval_total = 0
        start = time.time()
        record = []
        for test_batch in test_pairs:
            #print(test_batch)
            batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5])
            test_res,res_score,test_res_1,res_score_1 = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate, merge, output_lang, test_batch[5], batch_graph, predict_1, generate_1, merge_1, temp=temp, thre=args.thre, beam_size=beam_size)
            val_ac, equ_ac, test, tar = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            val_ac_1, equ_ac_1, test_1, tar_1 = compute_prefix_tree_result(test_res_1, test_batch[2], output_lang, test_batch[4], test_batch[6])
            record.append([[test,tar],[test_1,tar_1]])
            if val_ac or val_ac_1:
                value_ac += 1
            if equ_ac or equ_ac_1:
                equation_ac += 1
            if val_ac:
                value_ac_0 += 1
            if equ_ac:
                equation_ac_0 += 1
            if val_ac_1:
                value_ac_1 += 1
            if equ_ac_1:
                equation_ac_1 += 1
            if (val_ac and res_score > res_score_1) or (val_ac_1 and res_score <= res_score_1):
                value_ac_gate += 1
            if (equ_ac and res_score > res_score_1) or (equ_ac_1 and res_score <= res_score_1):
                equation_ac_gate += 1
            eval_total += 1
        print(equation_ac, value_ac, equation_ac_0, value_ac_0, equation_ac_1, value_ac_1, eval_total)
        print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("decoder1 test_answer_acc", float(equation_ac_0) / eval_total, float(value_ac_0) / eval_total)
        print("decoder2 test_answer_acc", float(equation_ac_1) / eval_total, float(value_ac_1) / eval_total)
        print("test_answer_acc", float(equation_ac_gate) / eval_total, float(value_ac_gate) / eval_total)
        print("testing time", time_since(time.time() - start))
        print("------------------------------------------------------")
        if value_ac > best_acc:
            best_acc = value_ac
            best_seq_path = best_path + str(value_ac) +'_'+str(value_ac_0) +'_' + str(value_ac_1) + '.json'
            #best_nodeout_path = best_node_path + str(value_ac) + '.pt'
            #best_idxout_path = best_idx_path + str(value_ac) + '.json'
            write_json(best_seq_path, record)
            #write_json(best_idxout_path, idx_batches)
            #torch.save(all_node_out,best_nodeout_path)
            torch.save(encoder.state_dict(), "model_traintest/encoder"+'_'+str(epoch)+'_'+str(best_acc))
            torch.save(predict.state_dict(), "model_traintest/predict"+'_'+str(epoch)+'_'+str(best_acc))
            torch.save(generate.state_dict(), "model_traintest/generate"+'_'+str(epoch)+'_'+str(best_acc))
            torch.save(merge.state_dict(), "model_traintest/merge"+'_'+str(epoch)+'_'+str(best_acc))
            torch.save(predict_1.state_dict(), "model_traintest/predict_1"+'_'+str(epoch)+'_'+str(best_acc))
            torch.save(generate_1.state_dict(), "model_traintest/generate_1"+'_'+str(epoch)+'_'+str(best_acc))
            torch.save(merge_1.state_dict(), "model_traintest/merge_1"+'_'+str(epoch)+'_'+str(best_acc))
        if epoch == n_epochs - 1:
            best_acc_fold.append((equation_ac, value_ac, eval_total))

    # if (epoch+1) % 10 == 0 or epoch ==0:
        # word_word = batch_evaluate_know(encoder, eval_know_inputs, gt_ww, know_gt_ww, args.temp, args.thre)
    # if epoch == n_epochs-1:
        # torch.save(eval_know_inputs, str(0)+"_eval_know_inputs.pt")
        # torch.save(word_word, str(0)+"_word_word.pt")
        # np.save(str(0)+"_word2index.npy", input_lang.word2index)

a, b, c = 0, 0, 0
for bl in range(len(best_acc_fold)):
    a += best_acc_fold[bl][0]
    b += best_acc_fold[bl][1]
    c += best_acc_fold[bl][2]
    print(best_acc_fold[bl])
print(a / float(c), b / float(c))
