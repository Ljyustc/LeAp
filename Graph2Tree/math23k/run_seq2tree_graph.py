# coding: utf-8
from src.train_and_evaluate import *
from src.models import *
from src.config import args
import time
import torch.optim
from torch.autograd import Variable
from src.expressions_transfer import *
import os
import json
import logging

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
    
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
USE_CUDA = torch.cuda.is_available()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", filename='log')
# occumpy_mem(args.cuda_id)

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file


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
logging.info(f"temp: {args.temp},thre: {args.thre},prior_prob: {args.prior_prob},kr: {kr}")

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


best_acc_fold = []

pairs_tested = test_fold
pairs_trained = train_fold

input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                copy_nums, tree=True)
# each element of train_pairs, test_pairs:
# [list of word_id,  len(list of word_id), list of operator_id,  len(list of operator_id), list of num, list of num_pos, temp_num, quantity_num]

encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                     n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
# the embedding layer is  only for generated number embeddings, operators, and paddings

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)

encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)

gt_know = np.load("data/math23k_know.npy")
gt_know = gt_know.tolist()

# Select ground_truth knowledge in input_lang
common_dict = {}
common_word_count = 0
eval_know_inputs = []
for triples in gt_know:
    w1, w2, rel = triples[0], triples[1], triples[2]
    # if rel in ['Antonym', 'IsA', 'MadeOf']:
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
    # if rel in ['Antonym', 'IsA', 'MadeOf']:
    if rel not in ['error']:
        if w1 in input_lang.word2index and w2 in input_lang.word2index:
            gt_ww[common_dict[input_lang.word2index[w1]], common_dict[input_lang.word2index[w2]]] = 1
eval_know_inputs = torch.LongTensor(eval_know_inputs)
    
know_gt_ww = torch.zeros((common_word_count, common_word_count))
for i in range(common_word_count):
    i_true = torch.where(gt_ww[i]!=0)[0].tolist()
    keep_true = random.sample(i_true, int(kr*len(i_true)))
    know_gt_ww[i][keep_true] = 1
torch.save(know_gt_ww, "0_know_gt_ww.pt")

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
    log_prior = log_prior.cuda()
    log_gt_prior = log_gt_prior.cuda()
    gt_ww = gt_ww.cuda()
    know_gt_ww = know_gt_ww.cuda()
    eval_know_inputs = eval_know_inputs.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

for epoch in range(n_epochs):
    temp = args.temp - (args.temp - 0.1) / (n_epochs-1) * epoch
    prior_lam = 0.1
    gt_prior_lam = 0.1
    encoder_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()
    loss_total = 0
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, \
   num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, graph_batches = prepare_train_batch(train_pairs, batch_size)
    logging.info(f"epoch: {epoch + 1}")
    start = time.time()
    for idx in range(len(input_lengths)):
        loss = train_tree(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
            encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang, num_pos_batches[idx], 
            graph_batches[idx], temp=temp, log_prior=log_prior, log_gt_prior=log_gt_prior, 
                eval_know_inputs=eval_know_inputs, know_gt_ww=know_gt_ww, common_dict=common_dict,
                prior_lam=prior_lam, gt_prior_lam=gt_prior_lam)
        loss_total += loss

    logging.info(f"loss: {loss_total / len(input_lengths)}")
    logging.info(f"training time: {time_since(time.time() - start)}")
    logging.info("--------------------------------")
    if epoch % 2 == 0 or epoch > n_epochs - 5:
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        for test_batch in test_pairs:
            #print(test_batch)
            batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5])
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                     merge, output_lang, test_batch[5], batch_graph, beam_size=beam_size, temp=temp, thre=args.thre)
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1
        logging.info(f"equation_ac: {equation_ac}, value_ac: {value_ac}, eval_total: {eval_total}")
        logging.info(f"test_equation_acc: {float(equation_ac) / eval_total}, test_value_acc: {float(value_ac) / eval_total}")
        logging.info(f"testing time: {time_since(time.time() - start)}")
        logging.info("------------------------------------------------------")
        torch.save(encoder.state_dict(), "model_traintest/encoder")
        torch.save(predict.state_dict(), "model_traintest/predict")
        torch.save(generate.state_dict(), "model_traintest/generate")
        torch.save(merge.state_dict(), "model_traintest/merge")
        if epoch == n_epochs - 1:
            best_acc_fold.append((equation_ac, value_ac, eval_total))
    
    # if (epoch+1) % 10 == 0 or epoch ==0:
        # word_word = batch_evaluate_know(encoder, eval_know_inputs, gt_ww, know_gt_ww, args.temp, args.thre)
    # if epoch == n_epochs-1:
        # torch.save(eval_know_inputs, str(0)+"_eval_know_inputs.pt")
        # torch.save(word_word, str(0)+"_word_word.pt")
        # np.save(str(0)+"_word2index.npy", input_lang.word2index)
            
        # word_op = encoder.wo_encoder.embed_know(encoder.embedding(eval_know_inputs), predict.ops, args.temp, hard=True) 
        # torch.save(word_op, str(0)+"_word_op.pt")
            
a, b, c = 0, 0, 0
for bl in range(len(best_acc_fold)):
    a += best_acc_fold[bl][0]
    b += best_acc_fold[bl][1]
    c += best_acc_fold[bl][2]
    print(best_acc_fold[bl])
print(a / float(c), b / float(c))
