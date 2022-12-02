import os
import sys
import math
import logging
import pdb
import random
import numpy as np
from attrdict import AttrDict
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import OrderedDict
try:
	import cPickle as pickle
except ImportError:
	import pickle

from src.args import build_parser
from src.utils.helper import *
from src.utils.logger import get_logger, print_log
from src.dataloader import TextDataset
from src.modelv2 import build_model, train_model, run_validation, estimate_confidence
from src.confidence_estimation import *

global log_folder
global model_folder
global result_folder
global data_path
global board_path

log_folder = 'logs'
model_folder = 'models'
outputs_folder = 'outputs'
result_folder = './out/'
data_path = './data/'
board_path = './runs/'

def load_data(config, logger):
	'''
		Loads the data from the datapath in torch dataset form

		Args:
			config (dict) : configuration/args
			logger (logger) : logger object for logging

		Returns:
			dataloader(s) 
	'''
	if config.mode == 'train':
		logger.debug('Loading Training Data...')

		'''Load Datasets'''
		train_set = TextDataset(data_path=data_path, dataset=config.dataset,
								datatype='train', max_length=config.max_length, is_debug=config.debug)
		val_set = TextDataset(data_path=data_path, dataset=config.dataset, datatype='dev', max_length=config.max_length, 
								is_debug=config.debug, grade_info=config.grade_disp, type_info=config.type_disp, 
								challenge_info=config.challenge_disp)
		
		'''In case of sort by length, write a different case with shuffle=False '''
		train_dataloader = DataLoader(
			train_set, batch_size=config.batch_size, shuffle=True, num_workers=5)
		val_dataloader = DataLoader(
			val_set, batch_size=config.batch_size, shuffle=True, num_workers=5)

		train_size = len(train_dataloader) * config.batch_size
		val_size = len(val_dataloader)* config.batch_size
		
		msg = 'Training and Validation Data Loaded:\nTrain Size: {}\nVal Size: {}'.format(train_size, val_size)
		logger.info(msg)

		return train_dataloader, val_dataloader

	elif config.mode == 'test' or config.mode == 'conf':
		logger.debug('Loading Test Data...')

		test_set = TextDataset(data_path=data_path, dataset=config.dataset,
							   datatype='test', max_length=config.max_length, is_debug=config.debug)
		test_dataloader = DataLoader(
			test_set, batch_size=config.batch_size, shuffle=True, num_workers=5)

		logger.info('Test Data Loaded...')
		return test_dataloader

	else:
		logger.critical('Invalid Mode Specified')
		raise Exception('{} is not a valid mode'.format(config.mode))

def main():
	'''read arguments'''
	parser = build_parser()
	args = parser.parse_args()
	config = args
	mode = config.mode
	if mode == 'train':
		is_train = True
	else:
		is_train = False

	''' Set seed for reproducibility'''
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	random.seed(config.seed)

	'''GPU initialization'''
	device = gpu_init_pytorch(config.gpu)

	# ground_truth knowledge
	gt_know = np.load("data/mawps_know.npy")
	gt_know = gt_know.tolist()
	prior_prob = config.prior_prob
	kr = config.kr
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

	if config.full_cv:
		global data_path 
		data_name = config.dataset
		data_path = data_path + data_name + '/'
		config.val_result_path = os.path.join(result_folder, 'CV_results_{}.json'.format(data_name))
		fold_acc_score = 0.0
		folds_scores = []
		for z in range(5):
			run_name = config.run_name + '_fold' + str(z)
			config.dataset = 'fold' + str(z)
			config.log_path = os.path.join(log_folder, run_name)
			config.model_path = os.path.join(model_folder, run_name)
			config.board_path = os.path.join(board_path, run_name)
			config.outputs_path = os.path.join(outputs_folder, run_name)

			vocab1_path = os.path.join(config.model_path, 'vocab1.p')
			vocab2_path = os.path.join(config.model_path, 'vocab2.p')
			config_file = os.path.join(config.model_path, 'config.p')
			log_file = os.path.join(config.log_path, 'log.txt')

			if config.results:
				config.result_path = os.path.join(result_folder, 'val_results_{}_{}.json'.format(data_name, config.dataset))

			if is_train:
				create_save_directories(config.log_path)
				create_save_directories(config.model_path)
				create_save_directories(config.outputs_path)
			else:
				create_save_directories(config.log_path)
				create_save_directories(config.result_path)

			logger = get_logger(run_name, log_file, logging.DEBUG)
			writer = SummaryWriter(config.board_path)

			logger.debug('Created Relevant Directories')
			logger.info('Experiment Name: {}'.format(config.run_name))

			'''Read Files and create/load Vocab'''
			if is_train:
				train_dataloader, val_dataloader = load_data(config, logger)

				logger.debug('Creating Vocab...')

				voc1 = Voc1()
				voc1.create_vocab_dict(config, train_dataloader)

				# To Do : Remove Later
				voc1.add_to_vocab_dict(config, val_dataloader)

				voc2 = Voc2(config)
				voc2.create_vocab_dict(config, train_dataloader)

				# To Do : Remove Later
				voc2.add_to_vocab_dict(config, val_dataloader)

				logger.info(
					'Vocab Created with number of words : {}'.format(voc1.nwords))

				with open(vocab1_path, 'wb') as f:
					pickle.dump(voc1, f, protocol=pickle.HIGHEST_PROTOCOL)
				with open(vocab2_path, 'wb') as f:
					pickle.dump(voc2, f, protocol=pickle.HIGHEST_PROTOCOL)

				logger.info('Vocab saved at {}'.format(vocab1_path))

			else:
				test_dataloader = load_data(config, logger)
				logger.info('Loading Vocab File...')

				with open(vocab1_path, 'rb') as f:
					voc1 = pickle.load(f)
				with open(vocab2_path, 'rb') as f:
					voc2 = pickle.load(f)

				logger.info('Vocab Files loaded from {}\nNumber of Words: {}'.format(vocab1_path, voc1.nwords))

			checkpoint = get_latest_checkpoint(config.model_path, logger)

			if is_train:
				# Select ground_truth knowledge in voc1
				common_dict = {}
				common_word_count = 0
				eval_know_inputs = []
				for triples in gt_know:
					w1, w2, rel = triples[0], triples[1], triples[2]
					# if rel in ['Antonym', 'IsA', 'MadeOf']:
					if rel not in ['error']:
						if w1 in voc1.w2id and w2 in voc1.w2id:
							if voc1.w2id[w1] not in common_dict:
								common_dict[voc1.w2id[w1]] = common_word_count
								common_word_count += 1
								eval_know_inputs.append(voc1.w2id[w1])
							if voc1.w2id[w2] not in common_dict:
								common_dict[voc1.w2id[w2]] = common_word_count
								common_word_count += 1
								eval_know_inputs.append(voc1.w2id[w2])
				logging.info(f"common_word_num: {common_word_count}")
				gt_ww = torch.zeros((common_word_count, common_word_count))
				for triples in gt_know:
					w1, w2, rel = triples[0], triples[1], triples[2]
					# if rel in ['Antonym', 'IsA', 'MadeOf']:
					if rel not in ['error']:
						if w1 in voc1.w2id and w2 in voc1.w2id:
							gt_ww[common_dict[voc1.w2id[w1]], common_dict[voc1.w2id[w2]]] = 1
				eval_know_inputs = torch.LongTensor(eval_know_inputs)

				know_gt_ww = torch.zeros((common_word_count, common_word_count))
				for i in range(common_word_count):
					i_true = torch.where(gt_ww[i]!=0)[0].tolist()
					keep_true = random.sample(i_true, int(kr*len(i_true)))
					know_gt_ww[i][keep_true] = 1
				torch.save(know_gt_ww, str(z)+"_know_gt_ww.pt")
				
				if device:
					log_prior = log_prior.to(device)
					log_gt_prior = log_gt_prior.to(device)
					gt_ww = gt_ww.to(device)
					know_gt_ww = know_gt_ww.to(device)
					eval_know_inputs = eval_know_inputs.to(device)
                
				model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger, num_iters=len(train_dataloader))

				logger.info('Initialized Model')
				
				if checkpoint == None:
					min_val_loss = torch.tensor(float('inf')).item()
					min_train_loss = torch.tensor(float('inf')).item()
					max_val_bleu = 0.0
					max_val_acc = 0.0
					max_train_acc = 0.0
					best_epoch = 0
					epoch_offset = 0
				else:
					epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = load_checkpoint(config, model, config.mode, checkpoint, logger, device)

				with open(config_file, 'wb') as f:
					pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

				logger.debug('Config File Saved')

				logger.info('Starting Training Procedure')
				max_val_acc = train_model(model, z, train_dataloader, val_dataloader, voc1, voc2,
							device, config, logger, epoch_offset, min_val_loss, max_val_bleu, max_val_acc, min_train_loss, max_train_acc, best_epoch, writer,
							log_prior=log_prior, log_gt_prior=log_gt_prior, eval_know_inputs=eval_know_inputs, know_gt_ww=know_gt_ww, gt_ww=gt_ww, common_dict=common_dict)
			else:
				gpu = config.gpu

				with open(config_file, 'rb') as f:
					config = AttrDict(pickle.load(f))
					config.gpu = gpu

				model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

				epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = load_checkpoint(config, model, config.mode, checkpoint, logger, device)

				logger.info('Prediction from')
				od = OrderedDict()
				od['epoch'] = epoch_offset
				od['min_train_loss'] = min_train_loss
				od['min_val_loss'] = min_val_loss
				od['max_train_acc'] = max_train_acc
				od['max_val_acc'] = max_val_acc
				od['max_val_bleu'] = max_val_bleu
				od['best_epoch'] = best_epoch
				print_log(logger, od)

				test_acc_epoch, test_loss_epoch = run_validation(config, model, test_dataloader, voc1, voc2, device, logger)
				logger.info('Accuracy: {} \t Loss: {}'.format(test_acc_epoch, test_loss_epoch))

			fold_acc_score += max_val_acc
			folds_scores.append(max_val_acc)

		fold_acc_score = fold_acc_score/5
		logger.info('Final Val score: {}'.format(fold_acc_score))
			

	else:
		'''Run Config files/paths'''
		run_name = config.run_name
		config.log_path = os.path.join(log_folder, run_name)
		config.model_path = os.path.join(model_folder, run_name)
		config.board_path = os.path.join(board_path, run_name)
		config.outputs_path = os.path.join(outputs_folder, run_name)

		vocab1_path = os.path.join(config.model_path, 'vocab1.p')
		vocab2_path = os.path.join(config.model_path, 'vocab2.p')
		config_file = os.path.join(config.model_path, 'config.p')
		log_file = os.path.join(config.log_path, 'log.txt')

		if config.results:
			config.result_path = os.path.join(result_folder, 'val_results_{}.json'.format(config.dataset))

		if is_train:
			create_save_directories(config.log_path)
			create_save_directories(config.model_path)
			create_save_directories(config.outputs_path)
		else:
			create_save_directories(config.log_path)
			create_save_directories(config.result_path)

		logger = get_logger(run_name, log_file, logging.DEBUG)
		writer = SummaryWriter(config.board_path)

		logger.debug('Created Relevant Directories')
		logger.info('Experiment Name: {}'.format(config.run_name))

		'''Read Files and create/load Vocab'''
		if is_train:
			train_dataloader, val_dataloader = load_data(config, logger)

			logger.debug('Creating Vocab...')

			voc1 = Voc1()
			voc1.create_vocab_dict(config, train_dataloader)

			# To Do : Remove Later
			voc1.add_to_vocab_dict(config, val_dataloader)

			voc2 = Voc2(config)
			voc2.create_vocab_dict(config, train_dataloader)

			# To Do : Remove Later
			voc2.add_to_vocab_dict(config, val_dataloader)

			logger.info(
				'Vocab Created with number of words : {}'.format(voc1.nwords))

			with open(vocab1_path, 'wb') as f:
				pickle.dump(voc1, f, protocol=pickle.HIGHEST_PROTOCOL)
			with open(vocab2_path, 'wb') as f:
				pickle.dump(voc2, f, protocol=pickle.HIGHEST_PROTOCOL)

			logger.info('Vocab saved at {}'.format(vocab1_path))

		else:
			test_dataloader = load_data(config, logger)
			logger.info('Loading Vocab File...')

			with open(vocab1_path, 'rb') as f:
				voc1 = pickle.load(f)
			with open(vocab2_path, 'rb') as f:
				voc2 = pickle.load(f)

			logger.info('Vocab Files loaded from {}\nNumber of Words: {}'.format(vocab1_path, voc1.nwords))

		checkpoint = get_latest_checkpoint(config.model_path, logger)

		if is_train:
			# Select ground_truth knowledge in voc1
			common_dict = {}
			common_word_count = 0
			eval_know_inputs = []
			for triples in gt_know:
				w1, w2, rel = triples[0], triples[1], triples[2]
				# if rel in ['Antonym', 'IsA', 'MadeOf']:
				if rel not in ['error']:
					if w1 in voc1.w2id and w2 in voc1.w2id:
						if voc1.w2id[w1] not in common_dict:
							common_dict[voc1.w2id[w1]] = common_word_count
							common_word_count += 1
							eval_know_inputs.append(voc1.w2id[w1])
						if voc1.w2id[w2] not in common_dict:
							common_dict[voc1.w2id[w2]] = common_word_count
							common_word_count += 1
							eval_know_inputs.append(voc1.w2id[w2])
			logging.info(f"common_word_num: {common_word_count}")
			gt_ww = torch.zeros((common_word_count, common_word_count))
			for triples in gt_know:
				w1, w2, rel = triples[0], triples[1], triples[2]
				# if rel in ['Antonym', 'IsA', 'MadeOf']:
				if rel not in ['error']:
					if w1 in voc1.w2id and w2 in voc1.w2id:
						gt_ww[common_dict[voc1.w2id[w1]], common_dict[voc1.w2id[w2]]] = 1
			eval_know_inputs = torch.LongTensor(eval_know_inputs)

			know_gt_ww = torch.zeros((common_word_count, common_word_count))
			for i in range(common_word_count):
				i_true = torch.where(gt_ww[i]!=0)[0].tolist()
				keep_true = random.sample(i_true, int(kr*len(i_true)))
				know_gt_ww[i][keep_true] = 1
			torch.save(know_gt_ww, "know_gt_ww.pt")
				
			if device:
				log_prior = log_prior.to(device)
				log_gt_prior = log_gt_prior.to(device)
				gt_ww = gt_ww.to(device)
				know_gt_ww = know_gt_ww.to(device)
				eval_know_inputs = eval_know_inputs.to(device)

			model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger, num_iters=len(train_dataloader))

			logger.info('Initialized Model')
			
			if checkpoint == None:
				min_val_loss = torch.tensor(float('inf')).item()
				min_train_loss = torch.tensor(float('inf')).item()
				max_val_bleu = 0.0
				max_val_acc = 0.0
				max_train_acc = 0.0
				best_epoch = 0
				epoch_offset = 0
			else:
				epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = load_checkpoint(config, model, config.mode, checkpoint, logger, device)

			with open(config_file, 'wb') as f:
				pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

			logger.debug('Config File Saved')

			logger.info('Starting Training Procedure')
			train_model(model, 'full', train_dataloader, val_dataloader, voc1, voc2,
						device, config, logger, epoch_offset, min_val_loss, max_val_bleu, max_val_acc, min_train_loss, max_train_acc, best_epoch, writer,
						log_prior=log_prior, log_gt_prior=log_gt_prior, eval_know_inputs=eval_know_inputs, know_gt_ww=know_gt_ww, gt_ww=gt_ww, common_dict=common_dict)

		else :
			gpu = config.gpu
			conf = config.conf
			sim_criteria = config.sim_criteria
			adv = config.adv
			mode = config.mode
			dataset = config.dataset
			batch_size = config.batch_size
			with open(config_file, 'rb') as f:
				config = AttrDict(pickle.load(f))
				config.gpu = gpu
				config.conf = conf
				config.sim_criteria = sim_criteria
				config.adv = adv
				config.mode = mode
				config.dataset = dataset
				config.batch_size = batch_size

			model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger,num_iters=len(test_dataloader))

			epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = load_checkpoint(config, model, config.mode, checkpoint, logger, device)

			logger.info('Prediction from')
			od = OrderedDict()
			od['epoch'] = epoch_offset
			od['min_train_loss'] = min_train_loss
			od['min_val_loss'] = min_val_loss
			od['max_train_acc'] = max_train_acc
			od['max_val_acc'] = max_val_acc
			od['max_val_bleu'] = max_val_bleu
			od['best_epoch'] = best_epoch
			print_log(logger, od)

			if config.mode == 'test':
				test_acc_epoch = run_validation(config, model, test_dataloader, voc1, voc2, device, logger, 0)
				logger.info('Accuracy: {}'.format(test_acc_epoch))
			else:
				estimate_confidence(config, model, test_dataloader, logger)


if __name__ == '__main__':
	main()


''' Just docstring format '''
# class Vehicles(object):
# 	'''
# 	The Vehicle object contains a lot of vehicles

# 	Args:
# 		arg (str): The arg is used for...
# 		*args: The variable arguments are used for...
# 		**kwargs: The keyword arguments are used for...

# 	Attributes:
# 		arg (str): This is where we store arg,
# 	'''
# 	def __init__(self, arg, *args, **kwargs):
# 		self.arg = arg

# 	def cars(self, distance,destination):
# 		'''We can't travel distance in vehicles without fuels, so here is the fuels

# 		Args:
# 			distance (int): The amount of distance traveled
# 			destination (bool): Should the fuels refilled to cover the distance?

# 		Raises:
# 			RuntimeError: Out of fuel

# 		Returns:
# 			cars: A car mileage
# 		'''
# 		pass