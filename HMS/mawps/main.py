# -*- coding: utf-8 -*-

import os
import logging
import random
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from config import get_args
from model import Encoder, Decoder, Seq2seq
from utils import DataLoader, Checkpoint, Evaluator, SupervisedTrainer

def init():
    args = get_args()
    if args.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
        if not torch.cuda.is_available():
            args.use_cuda = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", filename=args.log)
    logging.info('\n' + '\n'.join([f"\t{'['+k+']':20}\t{v}" for k, v in dict(args._get_kwargs()).items()]))

    checkpoint_path = os.path.join("./experiment", args.checkpoint)
    if not os.path.exists(checkpoint_path):
        logging.info(f'create checkpoint directory {checkpoint_path} ...')
        os.makedirs(checkpoint_path)
    Checkpoint.set_ckpt_path(checkpoint_path)
    return args

def create_model(args,gt_know):
    trim_min_count = 5
    data_loader = DataLoader(args, trim_min_count=trim_min_count)
    
    common_dict = {}
    common_word_count = 0
    eval_know_inputs = []
    for triples in gt_know:
        w1, w2, rel = triples[0], triples[1], triples[2]
        # if rel in ['Antonym', 'IsA', 'MadeOf']:
        if rel not in ['error']:
            if w1 in data_loader.vocab_dict and w2 in data_loader.vocab_dict:
                if data_loader.vocab_dict[w1] not in common_dict:
                    common_dict[data_loader.vocab_dict[w1]] = common_word_count
                    common_word_count += 1
                    eval_know_inputs.append(data_loader.vocab_dict[w1])
                if data_loader.vocab_dict[w2] not in common_dict:
                    common_dict[data_loader.vocab_dict[w2]] = common_word_count
                    common_word_count += 1
                    eval_know_inputs.append(data_loader.vocab_dict[w2])
    logging.info(f"common_word_num: {common_word_count}")
    gt_ww = torch.zeros((common_word_count, common_word_count))
    for triples in gt_know:
        w1, w2, rel = triples[0], triples[1], triples[2]
        # if rel in ['Antonym', 'IsA', 'MadeOf']:
        if rel not in ['error']:
            if w1 in data_loader.vocab_dict and w2 in data_loader.vocab_dict:
                gt_ww[common_dict[data_loader.vocab_dict[w1]], common_dict[data_loader.vocab_dict[w2]]] = 1
    eval_know_inputs = torch.LongTensor(eval_know_inputs)
    
    know_gt_ww = torch.zeros((common_word_count, common_word_count))
    for i in range(common_word_count):
        i_true = torch.where(gt_ww[i]!=0)[0].tolist()
        keep_true = random.sample(i_true, int(args.kr*len(i_true)))
        know_gt_ww[i][keep_true] = 1
    torch.save(know_gt_ww, str(args.fold)+"_know_gt_ww.pt")
                
    embed_model = nn.Embedding(data_loader.vocab_len, args.embed)
    embed_model.weight.data.copy_(data_loader.embed_vectors)
    encode_model = Encoder(
        embed_model=embed_model,
        hidden_size=args.hidden,
        span_size=data_loader.span_size,
        dropout=args.dropout,
    )

    decode_model = Decoder(
        embed_model=embed_model,
        op_set=data_loader.op_set,
        vocab_dict=data_loader.vocab_dict,
        class_list=data_loader.class_list,
        hidden_size=args.hidden,
        dropout=args.dropout,
        use_cuda=args.use_cuda
    )
    
    if args.use_cuda:
        gt_ww = gt_ww.cuda()
        know_gt_ww = know_gt_ww.cuda()
        eval_know_inputs = eval_know_inputs.cuda()    
        
    seq2seq = Seq2seq(encode_model, decode_model, gt_ww, common_dict, know_gt_ww, eval_know_inputs)
    return seq2seq, data_loader

def train(args,gt_know):
    prior_prob = args.prior_prob
    kr = args.kr

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
    
    seq2seq, data_loader = create_model(args,gt_know)
    if args.use_cuda:
        seq2seq = seq2seq.cuda()
        log_prior = log_prior.cuda()
        log_gt_prior = log_gt_prior.cuda()    
    
    st = SupervisedTrainer(
        class_dict=data_loader.class_dict,
        class_list=data_loader.class_list,
        use_cuda=args.use_cuda
    )

    logging.info('start training ...')
    st.train(
        model=seq2seq, 
        data_loader=data_loader,
        batch_size=args.batch,
        n_epoch=args.epoch,
        resume=args.resume,
        optim_lr=args.lr,
        optim_weight_decay=args.weight_decay,
        scheduler_step_size=args.step,
        scheduler_gamma=args.gamma,
        temp=args.temp,
        thre=args.thre,
        log_prior=log_prior,
        log_gt_prior=log_gt_prior
    )
    return

def test(args, test_dataset="test"):
    seq2seq, data_loader = create_model(args)
    resume_checkpoint = Checkpoint.load(model_only=True)
    seq2seq.load_state_dict(resume_checkpoint.model)
    if args.use_cuda:
        seq2seq = seq2seq.cuda()

    evaluator = Evaluator(
        class_dict=data_loader.class_dict,
        class_list=data_loader.class_list,
        use_cuda=args.use_cuda
    )
    if test_dataset == "test":
        test_dataset = data_loader.test_list
    elif test_dataset == "train":
        test_dataset = data_loader.train_list
    seq2seq.eval()
    with torch.no_grad():
        test_temp_acc, test_ans_acc = evaluator.evaluate(
            model=seq2seq,
            data_loader=data_loader,
            data_list=test_dataset,
            template_flag=True,
            template_len=False,
            batch_size=1,
            beam_width=args.beam,
            test_log=args.test_log,
            print_probability=True
        )
    logging.info(f"temp_acc: {test_temp_acc}, ans_acc: {test_ans_acc}")
    return

if __name__ == "__main__":
    args = init()
    # ground_truth knowledge
    gt_know = np.load("data/mawps_know.npy")
    gt_know = gt_know.tolist()
    if args.run_flag == "test":
        test(args, "test")
    elif args.run_flag == 'train':
        train(args,gt_know)
    else:
        logging.info('unknown run_flag')
