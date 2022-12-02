# -*- coding: utf-8 -*-

import logging
import random
import numpy as np
import torch
from torch import nn, optim

from utils import Checkpoint, Evaluator

def kl_categorical(preds, log_prior, mask, eps=1e-16):
    # logging.info(f"pred0:{preds[0,:,:,0]}")
    kl_div = (preds * (torch.log(preds + eps) - log_prior)).masked_fill_(mask.bool(), 0)
    return kl_div.sum() / ((1-mask[:,:,:,0]).sum()+1e-30)

def evaluate_know(encoder, inputs, gt_ww, know_gt_ww, temp):
    encoder.eval()
    word_num = inputs.size(0)
    embedded = encoder.embedding(inputs)  # (word_num, dim)
    
    # off-diagonal, not consider self-loop
    off_diag = torch.FloatTensor(np.ones([word_num, word_num]) - np.eye(word_num))
    if inputs.is_cuda:
        off_diag = off_diag.cuda()
    off_diag = off_diag * (1 - know_gt_ww)
    
    word_word_prob = encoder.ww_encoder.embed_know(embedded, temp, hard=True)  # (word_num, word_num)
    pre, rec = eval_know_k(word_num, word_word_prob*(1-know_gt_ww), gt_ww*(1-know_gt_ww), 50)
    logging.info(f"know_pre: {pre}, know_recall: {rec}")
    return word_word_prob

def eval_know_k(common_word_count, ww, gt_ww, k):
    off_diag = torch.FloatTensor(np.ones([common_word_count, common_word_count]) - np.eye(common_word_count))
    if ww.is_cuda:
        off_diag = off_diag.cuda()
    ww_temp = ww * off_diag
    topk_ids = torch.topk(ww_temp.view(-1),k)[1]
    gt_ww_temp = gt_ww.view(-1)
    right = 0
    for i in range(k):
        if gt_ww_temp[topk_ids[i]] == 1:
            right += 1

    precision = right / k
    recall = right / torch.sum(gt_ww*off_diag)
    return precision, recall
    
class SupervisedTrainer(object):
    def __init__(self, class_dict, class_list, use_cuda):
        self.test_train_every = 10
        self.print_every = 30
        self.use_cuda = use_cuda

        self.pad_idx_in_class = class_dict['PAD_token']

        loss_weight = torch.ones(len(class_dict))
        loss_weight[self.pad_idx_in_class] = 0
        self.loss = nn.NLLLoss(weight=loss_weight, reduction="sum")
        if use_cuda:
            self.loss = self.loss.cuda()
        
        self.evaluator = Evaluator(
            class_dict=class_dict,
            class_list=class_list,
            use_cuda=use_cuda
        )
        return

    def _train_batch(self, input_variables, num_pos, input_lengths, span_length, target_variables, tree, model, batch_size, temp, log_prior, log_gt_prior):
        decoder_outputs, _, _, generator_op_embedding, _ = model(
            input_variable=input_variables,
            num_pos=num_pos,
            input_lengths=input_lengths, 
            span_length=span_length,
            target_variable=target_variables, 
            tree=tree, 
            temp=temp,
            hard=False
        )
        
        # batch_size = span_length.size(0)
        # input_batch = torch.cat(input_variables, dim=1)
        # words_length = sum(input_lengths).cpu()
        # max_len = sum([int(max(i)) for i in input_lengths])
        # know_mask = torch.zeros((batch_size,max_len,max_len,2))
        # for ids in range(batch_size):
            # seq_id = input_batch[ids]
            # poses = torch.where(seq_id!=0)[0]
            # for i in range(words_length[ids]):
                # i_pos = poses[i]
                # i_id = int(seq_id[i_pos])
                # if i_id in model.common_dict:
                    # for j in np.arange(i+1, words_length[ids]):
                        # j_pos = poses[j]
                        # j_id = int(seq_id[j_pos])
                        # if j_id in model.common_dict:
                            # know_mask[ids, i_pos, j_pos] = model.gt_ww[model.common_dict[i_id], model.common_dict[j_id]]
                            # know_mask[ids, j_pos, i_pos] = model.gt_ww[model.common_dict[j_id], model.common_dict[i_id]]  
        # if self.use_cuda: 
            # know_mask = know_mask.cuda()
        ww_prob_prior, wo_prob_prior = model.encoder.encode_prior(model.eval_know_inputs.unsqueeze(1), generator_op_embedding)
        if log_prior is not None:
        # prior kl divergence
            loss_kl = kl_categorical(ww_prob_prior, log_prior, model.know_gt_ww.unsqueeze(0).unsqueeze(-1))
            
            wo_know_mask = torch.zeros((1, model.know_gt_ww.size(0), generator_op_embedding.size(0), 1))
            if self.use_cuda:
                wo_know_mask = wo_know_mask.cuda()
            loss_wo_kl = kl_categorical(wo_prob_prior, log_prior, wo_know_mask)
        else:
            loss_kl = 0
            loss_wo_kl = 0
    
        if log_gt_prior is not None:
        # prior kl divergence
            loss_gt_kl = kl_categorical(ww_prob_prior, log_gt_prior, (1-model.know_gt_ww).unsqueeze(0).unsqueeze(-1))
        else:
            loss_gt_kl = 0
        
        batch_size = span_length.size(0)

        # loss
        loss = 0
        for step, step_output in enumerate(decoder_outputs):
            loss += self.loss(step_output.contiguous().view(batch_size, -1), target_variables[:, step].view(-1))
        
        total_target_length = (target_variables != self.pad_idx_in_class).sum().item()
        loss = loss / total_target_length + 0.1 * loss_kl + 0.1 * loss_wo_kl + 0.1 * loss_gt_kl

        model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _train_epoches(self, data_loader, model, batch_size, start_epoch, start_step, max_acc, n_epoch, temp0, thre, log_prior, log_gt_prior):
        train_list = data_loader.train_list
        test_list = data_loader.test_list

        step = start_step
        print_loss_total = 0
        max_ans_acc = max_acc

        for epoch_index, epoch in enumerate(range(start_epoch, n_epoch + 1)):
            model.train()
            temp = temp0 - (temp0 - 0.1) / (n_epoch-start_epoch) * epoch_index
            batch_generator = data_loader.get_batch(train_list, batch_size, template_flag=True)
            for batch_data_dict in batch_generator:
                step += 1
                input_variables = batch_data_dict['batch_span_encode_idx']
                input_lengths = batch_data_dict['batch_span_encode_len']
                span_length = batch_data_dict['batch_span_len']
                tree = batch_data_dict["batch_tree"]

                input_variables = [torch.LongTensor(input_variable) for input_variable in input_variables]
                input_lengths = [torch.LongTensor(input_length) for input_length in input_lengths]
                span_length = torch.LongTensor(span_length)
                if self.use_cuda:
                    input_variables = [input_variable.cuda() for input_variable in input_variables]
                    input_lengths = [input_length.cuda() for input_length in input_lengths]
                    span_length = span_length.cuda()
                
                span_num_pos = batch_data_dict["batch_span_num_pos"]
                word_num_poses = batch_data_dict["batch_word_num_poses"]
                span_num_pos = torch.LongTensor(span_num_pos)
                word_num_poses = [torch.LongTensor(word_num_pos) for word_num_pos in word_num_poses]
                if self.use_cuda:
                    span_num_pos = span_num_pos.cuda()
                    word_num_poses = [word_num_pose.cuda() for word_num_pose in word_num_poses]
                num_pos = (span_num_pos, word_num_poses)

                target_variables = batch_data_dict['batch_decode_idx']
                target_variables = torch.LongTensor(target_variables)
                if self.use_cuda:
                    target_variables = target_variables.cuda()

                loss = self._train_batch(
                    input_variables=input_variables,
                    num_pos=num_pos, 
                    input_lengths=input_lengths,
                    span_length=span_length, 
                    target_variables=target_variables, 
                    tree=tree,
                    model=model,
                    batch_size=batch_size,
                    temp=temp, 
                    log_prior=log_prior, 
                    log_gt_prior=log_gt_prior
                )

                print_loss_total += loss
                if step % self.print_every == 0:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    logging.info(f'step: {step}, Train loss: {print_loss_avg:.4f}')
                    if self.use_cuda:
                        torch.cuda.empty_cache()
            self.scheduler.step()

            model.eval()
            with torch.no_grad():
                test_temp_acc, test_ans_acc = self.evaluator.evaluate(
                    model=model,
                    data_loader=data_loader,
                    data_list=test_list,
                    template_flag=True,
                    template_len=True,
                    batch_size=batch_size,
                )
                if epoch_index % self.test_train_every == 0:
                    train_temp_acc, train_ans_acc = self.evaluator.evaluate(
                        model=model,
                        data_loader=data_loader,
                        data_list=train_list,
                        template_flag=True,
                        template_len=True,
                        batch_size=batch_size,
                    )

                    logging.info(f"Epoch: {epoch}, Step: {step}, test_acc: {test_temp_acc:.3f}, {test_ans_acc:.3f}, train_acc: {train_temp_acc:.3f}, {train_ans_acc:.3f}")
                else:
                    logging.info(f"Epoch: {epoch}, Step: {step}, test_acc: {test_temp_acc:.3f}, {test_ans_acc:.3f}")

                # if (epoch_index+1) % 10 == 0 or epoch_index ==0:
                    # word_word = evaluate_know(model.encoder, model.eval_know_inputs, model.gt_ww, model.know_gt_ww, temp)
                # if epoch == n_epoch:
                    # torch.save(model.eval_know_inputs, "0_eval_know_inputs.pt")
                    # torch.save(word_word, "0_word_word.pt")
                    # np.save("0_word2index.npy", data_loader.vocab_dict)
                    
                    # word_op = model.encoder.wo_encoder.embed_know(model.encoder.embedding(model.eval_know_inputs), model.decoder.op_hidden(model.decoder.embed_model(model.decoder.generator_op_vocab)), temp0, hard=True) 
                    # torch.save(word_op, str(0)+"_word_op.pt")
                
            if test_ans_acc > max_ans_acc:
                max_ans_acc = test_ans_acc
                logging.info("saving checkpoint ...")
                Checkpoint.save(epoch=epoch, step=step, max_acc=max_ans_acc, model=model, optimizer=self.optimizer, scheduler=self.scheduler, best=True)
            else:
                Checkpoint.save(epoch=epoch, step=step, max_acc=max_ans_acc, model=model, optimizer=self.optimizer, scheduler=self.scheduler, best=False)
        return

    def train(self, model, data_loader, batch_size, n_epoch, resume=False, 
              optim_lr=1e-3, optim_weight_decay=1e-5, scheduler_step_size=60, scheduler_gamma=0.6, temp=0.5,
              thre=0.5, log_prior=None, log_gt_prior=None):
        start_epoch = 1
        start_step = 0
        max_acc = 0
        self.optimizer = optim.Adam(model.parameters(), lr=optim_lr, weight_decay=optim_weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        if resume:
            resume_checkpoint = Checkpoint.load(model_only=False)
            model.load_state_dict(resume_checkpoint.model)
            resume_optimizer = resume_checkpoint.optimizer
            resume_scheduler = resume_checkpoint.scheduler
            if resume_optimizer is not None:
                start_epoch = resume_checkpoint.epoch
                start_step = resume_checkpoint.step
                max_acc = resume_checkpoint.max_acc
                self.optimizer.load_state_dict(resume_optimizer)
                self.scheduler.load_state_dict(resume_scheduler)

        self._train_epoches(
            data_loader=data_loader, 
            model=model, 
            batch_size=batch_size,
            start_epoch=start_epoch, 
            start_step=start_step, 
            max_acc=max_acc,
            n_epoch=n_epoch,
            temp0=temp,
            thre=thre,
            log_prior=log_prior,
            log_gt_prior=log_gt_prior
        )
        return
