# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2016-10-17 14:31:13
# @Last Modified by:   rokeer
# @Last Modified time: 2018-11-11 16:26:13
import gzip
import logging
import sys
import gensim
# from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
# import theano
import numpy as np
##import tensorflow as tf
import os, errno
import logging
# import matplotlib.pyplot as plt

#import mxnet as mx
#from bert_embedding import BertEmbedding
import codecs
import re

import tensorflow as tf
import keras.backend as K

from nltk import tokenize
#from bertUtil import *

def ranking_loss(y_true, y_pred):
	
	y_pred = tf.convert_to_tensor(y_pred)
	y_true = tf.cast(y_true, y_pred.dtype)
	#assert y_true.shape == y_pred.shape
	y_true = K.exp(y_true) / K.sum(K.exp(y_true), axis=0)
	y_pred = K.exp(y_pred) / K.sum(K.exp(y_pred), axis=0)
	
	return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
	

def regression_loss(y_true, y_pred):
	
	y_pred = tf.convert_to_tensor(y_pred)
	y_true = tf.cast(y_true, y_pred.dtype)
	return K.mean(tf.math.squared_difference(y_pred, y_true), axis=-1)


def regression_and_ranking(cur_epoch, total_epoch):

	def r_square(y_true, y_pred):
		
		L_m = regression_loss(y_true, y_pred)
		L_r = ranking_loss(y_true, y_pred)
	
		t_1 = 1e-6
		gamma = K.log(1/t_1 - 1) / (total_epoch/2 - 1)
		#gamma = 0.99999 #as per paper 2020.findings.emnlp
	
		t_e = 1 / ( 1 + K.exp(gamma*(total_epoch/2 - cur_epoch)))
		loss = t_e * L_m + (1-t_e) * L_r
		return loss

	return r_square


def set_logger(out_dir=None):
	logger = logging.getLogger()
	if out_dir:
		file_format = '[%(levelname)s] (%(name)s) %(message)s'
		log_file = logging.FileHandler(out_dir + '/log.txt', mode='w')
		log_file.setLevel(logging.DEBUG)
		log_file.setFormatter(logging.Formatter(file_format))
		logger.addHandler(log_file)

def mkdir_p(path):
	if path == '':
		return
	try:
		os.makedirs(path)
	except OSError as exc: # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else: raise

def get_root_dir():
	return os.path.dirname(sys.argv[0])


def get_logger(name, level=logging.INFO, handler=sys.stdout,
		formatter='%(name)s - %(levelname)s - %(message)s'):
	logger = logging.getLogger(name)
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter(formatter)
	stream_handler = logging.StreamHandler(handler)
	stream_handler.setLevel(level)
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	return logger


def padding_sentence_sequences(index_sequences, scores, max_sentnum, max_sentlen, post_padding=True):

	X = np.empty([len(index_sequences), max_sentnum, max_sentlen], dtype=np.int32)
	Y = np.empty([len(index_sequences), 1], dtype=np.float32)
	mask = np.zeros([len(index_sequences), max_sentnum, max_sentlen], dtype=np.float32)

	for i in range(len(index_sequences)):
		sequence_ids = index_sequences[i]
		num = len(sequence_ids)

		for j in range(num):
			word_ids = sequence_ids[j]
			length = len(word_ids)
			# X_len[i] = length
			for k in range(length):
				wid = word_ids[k]
				# print wid
				X[i, j, k] = wid

			# Zero out X after the end of the sequence
			X[i, j, length:] = 0
			# Make the mask for this sample 1 within the range of length
			mask[i, j, :length] = 1

		X[i, num:, :] = 0
		Y[i] = scores[i]
	return X, Y, mask


def padding_sequences(word_indices, char_indices, scores, max_sentnum, max_sentlen, maxcharlen, post_padding=True):
	# support char features
	X = np.empty([len(word_indices), max_sentnum, max_sentlen], dtype=np.int32)
	Y = np.empty([len(word_indices), 1], dtype=np.float32)
	mask = np.zeros([len(word_indices), max_sentnum, max_sentlen], dtype=np.float32)

	char_X = np.empty([len(char_indices), max_sentnum, max_sentlen, maxcharlen], dtype=np.int32)

	for i in range(len(word_indices)):
		sequence_ids = word_indices[i]
		num = len(sequence_ids)

		for j in range(num):
			word_ids = sequence_ids[j]
			length = len(word_ids)
			# X_len[i] = length
			for k in range(length):
				wid = word_ids[k]
				# print wid
				X[i, j, k] = wid

			# Zero out X after the end of the sequence
			X[i, j, length:] = 0
			# Make the mask for this sample 1 within the range of length
			mask[i, j, :length] = 1

		X[i, num:, :] = 0
		Y[i] = scores[i]

	for i in range(len(char_indices)):
		sequence_ids = char_indices[i]
		num = len(sequence_ids)
		for j in range(num):
			word_ids = sequence_ids[j]
			length = len(word_ids)
			for k in range(length):
				wid = word_ids[k]
				charlen = len(wid)
				for l in range(charlen):
					cid = wid[l]
					char_X[i, j, k, l] = cid
				char_X[i, j, k, charlen:] = 0
			char_X[i, j, length:, :] = 0
		char_X[i, num:, :] = 0
	return X, char_X, Y, mask

def preprocess_essay_bert(text):
	text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', '<url>', text) ##replace_url
	text = text.replace(u'"', u'')
	if "..." in text:
		text = re.sub(r'\.{3,}(\s+\.{3,})*', '...', text)
		# print text
	if "??" in text:
		text = re.sub(r'\?{2,}(\s+\?{2,})*', '?', text)
		# print text
	if "!!" in text:
		text = re.sub(r'\!{2,}(\s+\!{2,})*', '!', text)
		# print text
	return text

def prepare_bert_data(datapath,norm_score, prompt_id, logger, embedd_dim=100):
	# logger.info('Creating Bert Embedding ---------')

	datasets=[]
	with codecs.open(datapath, mode='r', encoding='utf-8') as input_file:
		next(input_file)
		for line in input_file:
			tokens = line.strip().split('\t')
			bert_abstract = tokens[2].strip()
			bert_abstract = preprocess_essay_bert(bert_abstract)
			sentences = bert_abstract.split('\n')
			essay_set = int(tokens[1])

			if essay_set == prompt_id or prompt_id <= 0 or essay_set==9:
				#print("essay set", essay_set)
				datasets.append(sentences[0])

	max_seq_len = findMaximumSeqLen(datasets)
	input_ids_vals, input_mask_vals, segment_ids_vals = convert_essays_to_features(datasets, tokenizer, max_seq_len)

	print("input_word_ids: ", np.array(input_ids_vals).shape)
	print("input_mask_vals: ", np.array(input_mask_vals).shape)
	print("segment_ids_vals: ", np.array(segment_ids_vals).shape)

	datasets = {
      'input_word_ids': np.array(input_ids_vals),
       'input_mask': np.array(input_mask_vals),
       'input_type_ids': np.array(segment_ids_vals)
  	}
	#print('input_ids_vals: ', x['input_word_ids'].shape)
	#print('input_mask: ', x['input_mask'].shape)
	#print('input_type_ids: ', x['input_type_ids'].shape)
	'''
	datasets = tf.data.Dataset.from_tensor_slices((datasets, norm_score))
	data = (datasets.map(to_feature_map,
                           	num_parallel_calls=tf.data.experimental.AUTOTUNE
                           )
				.shuffle(1000)
  				.batch(batch, drop_remainder=True)
  				.prefetch(tf.data.experimental.AUTOTUNE))
	#print(train_data)
	#print(tokenizer.wordpiece_tokenizer.tokenize("hi, how, are you"))
	#print(tokenizer.convert_tokens_to_ids(tokenizer.wordpiece_tokenizer.tokenize("hi, how, are you")))
	'''
	return datasets

def load_word_embedding_dict_bert(datapaths, prompt_id, logger, embedd_dim=100):
	#train_path, dev_path, test_path = datapaths[0], datapaths[1], datapaths[2]
	#logger.info("Loading Bert ---------")
	
	#import tensorflow as tf
	from transformers import BertTokenizer, BertTokenizerFast, DistilBertTokenizer, AlbertTokenizer, RobertaTokenizer
	#import torch
	max_bert_tokens=512

	logger.info('Creating bert tokens from: ' + datapaths)

	#ctx = mx.gpu(1)
	#bert_embedding = BertEmbedding()
	embedd_dict = {}

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="/data/rahulk/transformer_models/") #DistilBertTokenizer.from_pretrained('distilbert-base-uncased') 

	#tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
	#model = TFBertModel.from_pretrained('bert-base-uncased')
	max_sequence_len = 0 #This length is supported by this bert model.
	count=0
	input_bertTokens = []
	input_mask = []

	# logger.info('Creating Embedding dict ---------')
	with codecs.open(datapaths, mode='r', encoding='utf-8') as input_file:
		next(input_file)
		for line in input_file:
			tokens = line.strip().split('\t')
			bert_abstract = tokens[2].strip()
			bert_abstract = preprocess_essay_bert(bert_abstract)
			sentences = bert_abstract.split('\n')
			

			essay_set = int(tokens[1])

			if essay_set == prompt_id or prompt_id <= 0 or essay_set==9:
				#print("essay set", essay_set)

				count+=1
				#print("count: ", count)
				#if count == 101: break

				##sent_list = tokenize.sent_tokenize(sentences[0])
				tokenizers = tokenizer.batch_encode_plus(sentences, return_token_type_ids=False, return_attention_mask=True, pad_to_max_length=True)

				#input_ids = torch.tensor(tokenizer.encode(bert_abstract)).detach().numpy() 
				#tokenizers = tokenizer.encode(sentences[0])
				in_arr = np.array(tokenizers['input_ids']).flatten()
				in_mask = np.array(tokenizers['attention_mask']).flatten()
				if max_sequence_len < len(in_arr): max_sequence_len = len(in_arr)

				#if(len(in_arr) > 512):
					#first_128_tokens = in_arr[1:129] #Without considering first [CLS] token
					#last_382_tokens = in_arr[-383:len(in_arr)-1] #Without considering last [SEP] token
					#in_arr = np.concatenate([first_128_tokens, last_382_tokens])

				input_bertTokens.append(in_arr)
				input_mask.append(in_mask)

	#logger.info('Bert Embedding is done --------- ')
	print("max_sequence_len: ", max_sequence_len)
	for i, token in enumerate(input_bertTokens):
		t = max_sequence_len - len(token)
		input_bertTokens[i] = np.pad(input_bertTokens[i], pad_width=(0, t), mode='constant')
		input_mask[i] = np.pad(input_mask[i], pad_width=(0, t), mode='constant')
	
	input_bertTokens = np.array(input_bertTokens, dtype='int32')
	input_mask = np.array(input_mask, dtype='int32')
	print("List input_bert_tokens: ", input_bertTokens.shape)

	#Considering First 512 bert tokens
	input_bert_tokens = input_bertTokens[:, :max_bert_tokens]
	input_mask = input_mask[:, :max_bert_tokens]
	
	input_data= {
		'input_ids': input_bert_tokens, 
		'input_mask': input_mask
	}

	return input_data




def load_word_embedding_dict(embedding, embedding_path, word_alphabet, logger, embedd_dim=100):
	"""
	load word embeddings from file
	:param embedding:
	:param embedding_path:
	:param logger:
	:return: embedding dict, embedding dimention, caseless
	"""
	if embedding == 'word2vec':
		# loading word2vec
		logger.info("Loading word2vec ...")
		word2vec = KeyedVectors.load_word2vec_format(embedding_path, binary=False, unicode_errors='ignore')
		embedd_dim = word2vec.vector_size
		return word2vec, embedd_dim, False
	elif embedding == 'glove':
		# loading GloVe
		logger.info("Loading GloVe ...")
		embedd_dim = -1
		embedd_dict = dict()
		with gzip.open(embedding_path, 'r') as file:
			for line in file:
				line = line.strip()
				if len(line) == 0:
					continue

				tokens = line.split()
				if embedd_dim < 0:
					embedd_dim = len(tokens) - 1
				else:
					assert (embedd_dim + 1 == len(tokens))
				embedd = np.empty([1, embedd_dim], dtype=np.float32)
				embedd[:] = tokens[1:]
				embedd_dict[tokens[0]] = embedd
		return embedd_dict, embedd_dim, True
	elif embedding == 'senna':
		# loading Senna
		logger.info("Loading Senna ...")
		embedd_dim = -1
		embedd_dict = dict()
		with gzip.open(embedding_path, 'r') as file:
			for line in file:
				line = line.strip()
				if len(line) == 0:
					continue

				tokens = line.split()
				if embedd_dim < 0:
					embedd_dim = len(tokens) - 1
				else:
					assert (embedd_dim + 1 == len(tokens))
				embedd = np.empty([1, embedd_dim], dtype=np.float32)
				embedd[:] = tokens[1:]
				embedd_dict[tokens[0]] = embedd
		return embedd_dict, embedd_dim, True
	# elif embedding == 'random':
	#     # loading random embedding table
	#     logger.info("Loading Random ...")
	#     embedd_dict = dict()
	#     words = word_alphabet.get_content()
	#     scale = np.sqrt(3.0 / embedd_dim)
	#     # print words, len(words)
	#     for word in words:
	#         embedd_dict[word] = np.random.uniform(-scale, scale, [1, embedd_dim])
	#     return embedd_dict, embedd_dim, False
	else:
		raise ValueError("embedding should choose from [word2vec, senna]")


def build_embedd_table(word_alphabet,train_wordFeaturesDict, embedd_dict, embedd_dim, logger, caseless):
	scale = np.sqrt(3.0 / embedd_dim)
	#print(train_wordFeaturesDict)
	wordFeaturesLen=0
	if train_wordFeaturesDict:
		wordFeaturesLen = len(list(train_wordFeaturesDict.values())[0])
		for word, feat in train_wordFeaturesDict.items():
			wordFeaturesLen = max(wordFeaturesLen, len(feat))

	embedd_table = np.empty([len(word_alphabet), embedd_dim+wordFeaturesLen], dtype=np.float32)
	embedd_table[0, :] = np.zeros([1, embedd_dim+wordFeaturesLen])
	oov_num = 0
	logger.info("Word Features Len = %s" % (wordFeaturesLen))
	for word, index in word_alphabet.items():
		ww = word.lower() if caseless else word
		# show oov ratio
		if ww in embedd_dict:
			embedd = embedd_dict[ww]
		else:
			embedd = np.random.uniform(-scale, scale, [1, embedd_dim])
			oov_num += 1
		if train_wordFeaturesDict:
			if ww in train_wordFeaturesDict:
				feat=np.array(train_wordFeaturesDict[ww])
				embedd = np.concatenate([embedd[0], feat])
			else:
				zeros = np.zeros([1,wordFeaturesLen])
				embedd = np.concatenate([embedd[0], zeros[0]])
		
			if(len(embedd) < embedd_dim+wordFeaturesLen):
				zeros = np.zeros(embedd_dim+wordFeaturesLen-len(embedd))
				embedd = np.concatenate([embedd, zeros])

		embedd = np.array([embedd])
		embedd_table[index, :] = embedd
	oov_ratio = float(oov_num)/(len(word_alphabet)-1)
	logger.info("OOV number =%s, OOV ratio = %f" % (oov_num, oov_ratio))
	return embedd_table

def rescale_tointscore(scaled_scores, set_ids, score_index, overall_score_column):
	'''
	rescale scaled scores range[0,1] to original integer scores based on  their set_ids
	:param scaled_scores: list of scaled scores range [0,1] of essays
	:param set_ids: list of corresponding set IDs of essays, integer from 1 to 8
	# '''
	# print (type(scaled_scores))
	# print (scaled_scores)
	scaled_scores = np.array(scaled_scores)
	if isinstance(set_ids, int):
		prompt_id = set_ids
		set_ids = np.ones(scaled_scores.shape[0],) * prompt_id
	assert scaled_scores.shape[0] == len(set_ids)
	int_scores = np.zeros((scaled_scores.shape[0], 1))
	# print ("int_score_shape: ", int_scores.shape)
	for k, i in enumerate(set_ids):
		assert i in range(1, 9)
		# TODO
		#For overall score column asap_ranges[prompt_id]
		if score_index==overall_score_column:
			if i == 1:
				minscore = 2
				maxscore = 12
			elif i == 2:
				minscore = 1
				maxscore = 6
			elif i in [3, 4]:
				minscore = 0
				maxscore = 3
			elif i in [5, 6]:
				minscore = 0
				maxscore = 4
			elif i == 7:
				minscore = 0
				maxscore = 30
			elif i == 8:
				minscore = 0
				maxscore = 60
			else:
				print("Set ID error")
		else:
			#traits min and max score
			if i in [1, 2]:
				minscore = 1
				maxscore = 6
			elif i in [3, 4]:
				minscore = 0
				maxscore = 3
			elif i in [5, 6]:
				minscore = 0
				maxscore = 4
			elif i == 7:
				minscore = 0
				maxscore = 6
			elif i == 8:
				minscore = 0
				maxscore = 12
		# minscore = 0
		# maxscore = 60

		int_scores[k] = scaled_scores[k]*(maxscore-minscore) + minscore

	return np.around(int_scores).astype(int)


def rescale_tointscore_for_attr(scaled_scores,count,norm_shape, set_ids):
	'''
	rescale scaled scores range[0,1] to original integer scores based on  their set_ids
	:param scaled_scores: list of scaled scores range [0,1] of essays
	:param set_ids: list of corresponding set IDs of essays, integer from 1 to 8
	'''


	# if(count):
	#     new_scaled_scores=[]
	#     j=0
	#     for i in range(len(scaled_scores)):
	#         if (i==0):
	#             temp=[]
	#             temp.append([x[0] for x in scaled_scores[j]])
	#             temp = sum(temp, [])
	#             new_scaled_scores.append(temp)
	#         else:
	#             j=j+1
	#             temp=scaled_scores[j]
	#             temp=temp.flatten()
	#             new_scaled_scores.append(temp)

	#     new_scaled_scores=np.array(new_scaled_scores)
	#     scaled_scores = new_scaled_scores




	if isinstance(set_ids, int):
		prompt_id = set_ids
		# set_ids = np.ones(scaled_scores.shape[0],) * prompt_id
		set_ids = np.ones(shape=(norm_shape[0],norm_shape[1])) * prompt_id


	# assert scaled_scores.shape[0] == len(set_ids)
	int_scores = np.zeros(shape=(norm_shape[0],norm_shape[1]))

	for k, j in enumerate(set_ids):
		for x,i in enumerate(set_ids[k]): 
			assert i in range(1, 9)
			# TODO
			if k==0:
				if i == 1:
					minscore = 2
					maxscore = 12
				elif i == 2:
					minscore = 1
					maxscore = 6
				elif i in [3, 4]:
					minscore = 0
					maxscore = 3
				elif i in [5, 6]:
					minscore = 0
					maxscore = 4
				elif i == 7:
					minscore = 0
					maxscore = 30
				elif i == 8:
					minscore = 0
					maxscore = 60
			#traits min and max score
			elif k!=0:
				if i in [1, 2]:
					minscore = 1
					maxscore = 6
				elif i in [3, 4]:
					minscore = 0
					maxscore = 3
				elif i in [5, 6]:
					minscore = 0
					maxscore = 4
				elif i == 7:
					minscore = 0
					maxscore = 6
				elif i == 8:
					minscore = 0
					maxscore = 12
			else:
				print("Set ID error")
			# minscore = 0
			# maxscore = 60
			# print("scaled_scores[k][x]: ", scaled_scores[k][x])
			# if(count):
			#     int_scores[0][k][x] = scaled_scores[0][k][x]*(maxscore-minscore) + minscore
			# else:
			int_scores[k][x] = scaled_scores[k][x]*(maxscore-minscore) + minscore

	return np.around(int_scores).astype(int)


def domain_specific_rescale(y_true, y_pred, set_ids):
	'''
	rescaled scores to original integer scores based on their set ids
	and partition the score list based on its specific prompot
	return 8-prompt int score list for y_true and y_pred respectively
	:param y_true: true score list, contains all 8 prompts
	:param y_pred: pred score list, also contains 8 prompts
	:param set_ids: list that indicates the set/prompt id for each essay
	'''
	# prompts_truescores = []
	# prompts_predscores = []
	y_true = y_true.flatten()
	y_pred = y_pred.flatten()

	y1_true, y1_pred = [], []
	y2_true, y2_pred = [], []
	y3_true, y3_pred = [], []
	y4_true, y4_pred = [], []
	y5_true, y5_pred = [], []
	y6_true, y6_pred = [], []
	y7_true, y7_pred = [], []
	y8_true, y8_pred = [], []

	for k, i in enumerate(set_ids):
		assert i in range(1, 9)
		if i == 1:
			minscore = 0
			maxscore = 3
			y1_true.append(y_true[k]*(maxscore-minscore) + minscore)
			y1_pred.append(y_pred[k]*(maxscore-minscore) + minscore)
		elif i == 2:
			minscore = 1
			maxscore = 6
			y2_true.append(y_true[k]*(maxscore-minscore) + minscore)
			y2_pred.append(y_pred[k]*(maxscore-minscore) + minscore)
		elif i == 3:
			minscore = 0
			maxscore = 3
			y3_true.append(y_true[k]*(maxscore-minscore) + minscore)
			y3_pred.append(y_pred[k]*(maxscore-minscore) + minscore)
		elif i == 4:
			minscore = 0
			maxscore = 3
			y4_true.append(y_true[k]*(maxscore-minscore) + minscore)
			y4_pred.append(y_pred[k]*(maxscore-minscore) + minscore)

		elif i == 5:
			minscore = 0
			maxscore = 4
			y5_true.append(y_true[k]*(maxscore-minscore) + minscore)
			y5_pred.append(y_pred[k]*(maxscore-minscore) + minscore)
		elif i == 6:
			minscore = 0
			maxscore = 4
			y6_true.append(y_true[k]*(maxscore-minscore) + minscore)
			y6_pred.append(y_pred[k]*(maxscore-minscore) + minscore)

		elif i == 7:
			minscore = 0
			maxscore = 3
			y7_true.append(y_true[k]*(maxscore-minscore) + minscore)
			y7_pred.append(y_pred[k]*(maxscore-minscore) + minscore)

		elif i == 8:
			minscore = 0
			maxscore = 60
			y8_true.append(y_true[k]*(maxscore-minscore) + minscore)
			y8_pred.append(y_pred[k]*(maxscore-minscore) + minscore)

		else:
			print("Set ID error")
	prompts_truescores = [np.around(y1_true), np.around(y2_true), np.around(y3_true), np.around(y4_true), \
							np.around(y5_true), np.around(y6_true), np.around(y7_true), np.around(y8_true)]
	prompts_predscores = [np.around(y1_pred), np.around(y2_pred), np.around(y3_pred), np.around(y4_pred), \
							np.around(y5_pred), np.around(y6_pred), np.around(y7_pred), np.around(y8_pred)]

	return prompts_truescores, prompts_predscores
# def plot_convergence(train_stats, dev_stats, test_stats, metric_type='mse'):
#     '''
#     Plot convergence curve of training process
#     :param train_stats: list of train metrics at each epoch
#     :param dev_stats: list of dev metrics at each epoch
#     :param test_stas: list of test metrics at each epoch
#     '''
#     num_epochs = len(train_stats)
#     x = range(1, num_epochs+1)

#     plt.plot(x, train_stats)
#     plt.plot(x, dev_stats)
#     plt.plot(x, test_stats)
#     plt.legend(['train', 'dev', 'test'], loc='upper right')
#     plt.xlabel('num of epochs')
#     if metric_type == 'kappa':
#         y_label = 'Kappa value'
#     else:
#         y_label = 'Mean square error'
#     plt.ylabel('%s' % y_label)
#     plt.show()

import re

class BColors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	WHITE = '\033[37m'
	YELLOW = '\033[33m'
	GREEN = '\033[32m'
	BLUE = '\033[34m'
	CYAN = '\033[36m'
	RED = '\033[31m'
	MAGENTA = '\033[35m'
	BLACK = '\033[30m'
	BHEADER = BOLD + '\033[95m'
	BOKBLUE = BOLD + '\033[94m'
	BOKGREEN = BOLD + '\033[92m'
	BWARNING = BOLD + '\033[93m'
	BFAIL = BOLD + '\033[91m'
	BUNDERLINE = BOLD + '\033[4m'
	BWHITE = BOLD + '\033[37m'
	BYELLOW = BOLD + '\033[33m'
	BGREEN = BOLD + '\033[32m'
	BBLUE = BOLD + '\033[34m'
	BCYAN = BOLD + '\033[36m'
	BRED = BOLD + '\033[31m'
	BMAGENTA = BOLD + '\033[35m'
	BBLACK = BOLD + '\033[30m'
	
	@staticmethod
	def cleared(s):
		return re.sub("\033\[[0-9][0-9]?m", "", s)

def red(message):
	return BColors.RED + str(message) + BColors.ENDC

def b_red(message):
	return BColors.BRED + str(message) + BColors.ENDC

def blue(message):
	return BColors.BLUE + str(message) + BColors.ENDC

def b_yellow(message):
	return BColors.BYELLOW + str(message) + BColors.ENDC

def green(message):
	return BColors.GREEN + str(message) + BColors.ENDC

def b_green(message):
	return BColors.BGREEN + str(message) + BColors.ENDC

#-----------------------------------------------------------------------------------------------------------#

def print_args(args, path=None):
	if path:
		output_file = open(path, 'w')
	logger = logging.getLogger(__name__)
	logger.info("Arguments:")
	args.command = ' '.join(sys.argv)
	items = vars(args)
	for key in sorted(items.keys(), key=lambda s: s.lower()):
		value = items[key]
		if not value:
			value = "None"
		logger.info("  " + key + ": " + str(items[key]))
		if path is not None:
			output_file.write("  " + key + ": " + str(items[key]) + "\n")
	if path:
		output_file.close()
	del args.command

