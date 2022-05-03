# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-01-10 11:57:22
# @Last Modified by:   rokeer
# @Last Modified time: 2018-11-11 16:26:13


import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
import sys
#sys.path.append('/home/rahulee16/Rahul/System/')
import argparse
import random
import time
import numpy as np
from utils import *
import data_prepare_STL as data_prepare
from evaluator_STL import Evaluator
from keras.models import Model
import joblib
import pickle
from bert import bert_model, build_bert_model, load_bert_embedd, regression_and_ranking
import keras
import keras.backend as K
import tensorflow as tf
# from tf.keras.optimizers import RMSprop

# import tensorflow_hub as hub
# from official.nlp.data import classifier_data_lib
# from official.nlp.bert import tokenization
# from official.nlp import optimization

import logging
tf.get_logger().setLevel(logging.ERROR)

# from bertUtil import *

from hand_crafted_features import read_hand_crafted_features, read_word_feat, read_sent_feat, read_essay_feat

logger = get_logger("ATTN")
np.random.seed(100)

#class NewCallback(keras.callbacks.Callback):
#    def __init__(self, current_epoch):
#        self.current_epoch = current_epoch

#    def on_epoch_begin(self, epoch, logs={}):
#        K.set_value(self.current_epoch,  K.get_value(self.current_epoch) * epoch ** 0.95)
#        print("Start epoch {} of training; got log keys: {}".format(epoch, self.current_epoch))


alpha = K.variable(1.)
class NewCallback(keras.callbacks.Callback):
    def __init__(self, alpha):
        self.alpha = alpha       
    def on_epoch_end(self, epoch, logs={}):
        K.set_value(self.alpha, K.get_value(self.alpha) * epoch**0.95)
        print("Start epoch {} of training; got log keys: {}".format(epoch, self.alpha))

def main():
	#arguments
	
	t0_run_time = time.time()

	parser = argparse.ArgumentParser(description="sentence Hi_BERT model")
	parser.add_argument('--train_flag', action='store_true', help='Train or eval')
	parser.add_argument('--fine_tune', action='store_true', help='Fine tune word embeddings')
	parser.add_argument('--embedding', type=str, default='bert', help='Word embedding type, bert')
	parser.add_argument('--embedding_dict', type=str, choices='glove/glove.6B.50d.txt.gz', help='Pretrained embedding path')
	parser.add_argument('--embedding_dim', type=int, default=50, help='Only useful when embedding is randomly initialised')
	parser.add_argument('--char_embedd_dim', type=int, default=30, help='char embedding dimension if using char embedding')

	parser.add_argument('--use_char', action='store_false', help='Whether use char embedding or not')
	parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for training')
	parser.add_argument('--batch_size', type=int, default=1, help='Number of texts in each batch')
	parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")


	parser.add_argument('--word_feat_path', type=str,default=None, help='Word Features')
	parser.add_argument('--sent_feat_path', type=str,default=None, help='Sentence Features')
	parser.add_argument('--essay_feat_path', type=str, help='Hand crafted Essay Features')
	parser.add_argument('--prompt_filePath', type=str, help='For cross domain, prompt-specific vocab')
	parser.add_argument('--ridley_feat_path', help='Hand crafted Ridely Essay Features', nargs='+')
	parser.add_argument('--bert_repr_path', type=str, help='Load bert embedding .npy files')

	parser.add_argument('--word_feat_flag', type=int,default=0, help='Use Word Features or not')
	parser.add_argument('--sent_feat_flag', type=int,default=0, help='Use Sentence Features or not')
	parser.add_argument('--essay_feat_flag', type=int,default=0, help='Use Hand crafted Essay Features or not')
	parser.add_argument('--ridley_feat_flag', type=int,default=0, help='Use Hand crafted Ridely Essay Features or not')

	parser.add_argument('--nbfilters', type=int, default=100, help='Num of filters in conv layer')

	parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, 
									help="The path to the output directory")
	parser.add_argument("-wt", "--wt-dir",default='model_weights', dest="model_weights", help="Model weights for each traits")

	parser.add_argument('--score_index', type=int, default=6, help='Overall Score Column index')
	parser.add_argument('--score_weight', type=float, default=1.0, help='Overall Score Weight')

	parser.add_argument('--loss', type=str, default='regression', choices=['regression','ranking','regression_and_ranking'], 
						help='Choose which loss function to use')
	parser.add_argument('--checkpoint_path', type=str, help='checkpoint directory', default='checkpoints')
	parser.add_argument('--train', type=str, help='train file', default='data/fold_0/train.tsv')  # "data/word-level/*.trpreprocess_asap.pyain"
	parser.add_argument('--dev', type=str, help='dev file', default='data/fold_0/dev.tsv')
	parser.add_argument('--test', type=str, help='test file', default='data/fold_0/test.tsv')
	parser.add_argument('--prompt_id', type=int, default=1, help='prompt id of essay set')
	parser.add_argument('--init_bias', action='store_true', help='init the last layer bias with average score of training data')
	parser.add_argument('--mode', type=str, choices=['mot', 'att', 'merged'], default='att', \
						help='Mean-over-Time pooling or attention-pooling, or two pooling merged')

	args = parser.parse_args()
	args.use_char = False
	train_flag = args.train_flag
	fine_tune = args.fine_tune
	USE_CHAR = args.use_char

	batch_size = args.batch_size
	checkpoint_dir = args.checkpoint_path
	num_epochs = args.num_epochs

	#output dir. for preds files
	out_dir = args.out_dir_path
	mkdir_p(out_dir + '/preds')
	set_logger(out_dir)

	# Required Attributes for Multi-task learning 
	overall_loss_weight = 1.0
	overall_score_column = 6
	score_index = args.score_index
	score_weight = args.score_weight

	modelname = "attn-%s.prompt%s.%sfilters.bs%s.hdf5" % (args.mode, args.prompt_id, args.nbfilters, batch_size)
	imgname = "attn-%s.prompt%s.%sfilters.bs%s.png" % (args.mode, args.prompt_id, args.nbfilters, batch_size)

	#Model name for each traits
	#model_wt_dir = args.model_weights
	#mkdir_p(model_wt_dir + '/Prompt-'+ str(args.prompt_id))
	# set_logger(model_wt_dir)

	# print("trait_modelname: ", tr_modelname)
	modelpath = os.path.join(checkpoint_dir, modelname)
	imgpath = os.path.join(checkpoint_dir, imgname)

	datapaths = [args.train, args.dev, args.test]
	embedding_path = args.embedding_dict
	embedding = args.embedding
	embedd_dim = args.embedding_dim
	prompt_id = args.prompt_id

	wordFeaturesDict, sentenceFeaturesDict, essayFeaturesDict = {}, {}, {}
	if args.word_feat_flag:
		wordFeaturesDict = read_word_feat(args.word_feat_path)
	sentenceFeaturesDict = read_sent_feat(args.sent_feat_path)
	essayFeaturesDict = read_essay_feat(args.essay_feat_path)
	#exit()

	# Preparing Train, Dev and Test Datasets
	(X_train, Y_train, train_sentenceFeatures, train_essayFeatures, train_essayIDList, mask_train, train_true_score, train_norm_score), (X_dev, Y_dev, dev_sentenceFeatures, dev_essayFeatures, dev_essayIDList, mask_dev, dev_true_score, dev_norm_score), (X_test, Y_test, test_sentenceFeatures, test_essayFeatures, test_essayIDList, mask_test, test_true_score, test_norm_score), \
			vocab, vocab_size, embed_table, overal_maxlen, overal_maxnum, init_mean_value = data_prepare.prepare_sentence_data(datapaths, \
			args.prompt_filePath, overall_score_column, wordFeaturesDict, sentenceFeaturesDict, essayFeaturesDict, \
			embedding_path, embedding, embedd_dim, prompt_id, args.vocab_size, tokenize_text=True, \
			to_lower=True, sort_by_len=False, vocab_path=None, score_index=score_index)

		
	max_sentnum = overal_maxnum
	max_sentlen = overal_maxlen

	train_sentenceFeatures = np.array(train_sentenceFeatures)
	dev_sentenceFeatures = np.array(dev_sentenceFeatures)
	test_sentenceFeatures = np.array(test_sentenceFeatures)
	logger.info("Embedding Dimen.: %s" % str(embedd_dim))
	logger.info("train_sentenceFeatures shape: %s" % str(train_sentenceFeatures.shape))
	logger.info("dev_sentenceFeatures shape: %s"% str(dev_sentenceFeatures.shape))
	logger.info("test_sentenceFeatures shape: %s" % str(test_sentenceFeatures.shape))

	train_essayFeatures = np.array(train_essayFeatures)
	dev_essayFeatures = np.array(dev_essayFeatures)
	test_essayFeatures = np.array(test_essayFeatures)
	logger.info("train_essayFeatures shape: %s" % str(train_essayFeatures.shape))
	logger.info("dev_essayFeatures shape: %s"% str(dev_essayFeatures.shape))
	logger.info("test_essayFeatures shape: %s" % str(test_essayFeatures.shape))


	#Hand Crafted Features
	hand_feat_path = args.ridley_feat_path[0]
	readability_path = args.ridley_feat_path[1]
	feat_train, feat_dev, feat_test = read_hand_crafted_features(datapaths,hand_feat_path,readability_path, prompt_id)
	train_feat, dev_feat, test_feat = feat_train['features_x'], feat_dev['features_x'], feat_test['features_x'] 
	train_read_feat, dev_read_feat, test_read_feat = feat_train['readability_x'], feat_dev['readability_x'], feat_test['readability_x'] 
	#X_train = np.concatenate([X_train, train_feat], axis=1)
	#X_dev = np.concatenate([X_dev, dev_feat], axis=1)
	#X_test = np.concatenate([X_test, test_feat], axis=1)

	logger.info("train_feat shape: %s" % str(train_feat.shape))
	logger.info("dev_feat_shape: %s"% str(dev_feat.shape))
	logger.info("test_feat_shape: %s" % str(test_feat.shape))
	
	logger.info("train_read_feat_shape: %s" % str(train_read_feat.shape))
	logger.info("dev_read_feat_shape: %s"% str(dev_read_feat.shape))
	logger.info("test_read_feat_shape: %s" % str(test_read_feat.shape))

	train_norm_score = np.array(train_norm_score)
	dev_norm_score = np.array(dev_norm_score)
	test_norm_score = np.array(test_norm_score)


	print("max_sentnum: ", max_sentnum)
	print("max_sentlen: ", max_sentlen)

	start_time = time.time()
	logger.info("Loading BERT -------------------------------")

	train_path, dev_path, test_path = datapaths[0], datapaths[1], datapaths[2]

	flag=1
	if(flag==1):
		'''
		train_essay_repr = prepare_bert_data(train_path, train_norm_score, prompt_id, logger, embedd_dim)
		dev_essay_repr = prepare_bert_data(dev_path, dev_norm_score, prompt_id, logger, embedd_dim)
		test_essay_repr = prepare_bert_data(test_path, test_norm_score, prompt_id, logger, embedd_dim)
		'''
		train_essay_repr = load_word_embedding_dict_bert(train_path, prompt_id, logger, embedd_dim)['input_ids']
		dev_essay_repr = load_word_embedding_dict_bert(dev_path, prompt_id, logger, embedd_dim)['input_ids']
		test_essay_repr = load_word_embedding_dict_bert(test_path, prompt_id, logger, embedd_dim)['input_ids']

		'''
		bert_start_time = time.time()
		train_essay_repr = bert_model(train_bert_tokens)
		dev_essay_repr = bert_model(dev_bert_tokens)
		test_essay_repr = bert_model(test_bert_tokens)
		bert_total_time = time.time() - bert_start_time
		print("Pretrained Bert Model took time: ", bert_total_time)

		train_essay_repr = train_essay_repr[:, 0, :]
		dev_essay_repr = dev_essay_repr[:, 0, :]
		test_essay_repr = test_essay_repr[:, 0, :]
		'''

		#logger.info("Saving Numpy array into .npy format")
		#np.save("./bert/Prompt-"+str(prompt_id)+"/train_essay_repr.npy", train_essay_repr)
		#np.save("./bert/Prompt-"+str(prompt_id)+"/dev_essay_repr.npy", dev_essay_repr)
		#np.save("./bert/Prompt-"+str(prompt_id)+"/test_essay_repr.npy", test_essay_repr)

	else:
		logger.info("Loading .npy format Bert representations")
		#train_essay_repr = np.load('../bert_array/Prompt-3/fold-1/train_essay_repr.npy')
		#dev_essay_repr = np.load('../bert_array/Prompt-3/fold-1/dev_essay_repr.npy')
		#test_essay_repr = np.load('../bert_array/Prompt-3/fold-1/test_essay_repr.npy')
		train_essay_repr = load_bert_embedd(args.bert_repr_path, train_essayIDList)
		dev_essay_repr = load_bert_embedd(args.bert_repr_path, dev_essayIDList)
		test_essay_repr = load_bert_embedd(args.bert_repr_path, test_essayIDList)

	#train_essay_repr = np.expand_dims(train_essay_repr, 1)
	#dev_essay_repr = np.expand_dims(dev_essay_repr, 1)
	#test_essay_repr = np.expand_dims(test_essay_repr, 1)

	logger.info("train_essay_repr_shape: %s" % str(np.array(train_essay_repr).shape))
	logger.info("dev_essay_repr_shape: %s"% str(np.array(dev_essay_repr).shape))
	logger.info("test_essay_repr_Shape: %s" % str(np.array(test_essay_repr).shape))
	#exit()

	#assert train_essay_repr.shape[1] == dev_essay_repr.shape[1] == test_essay_repr.shape[1]

	tt_time = time.time() - start_time
	logger.info("Loading word embedding in %.3f s" % tt_time)

	#train_feat = np.expand_dims(train_feat, 1)
	#dev_feat = np.expand_dims(dev_feat, 1)
	#test_feat = np.expand_dims(test_feat, 1)

	#train_read_feat = np.expand_dims(train_read_feat, 1)
	#dev_read_feat = np.expand_dims(dev_read_feat, 1)
	#test_read_feat = np.expand_dims(test_read_feat, 1)

	
	model = build_bert_model(args,score_weight, train_essay_repr.shape[1:], 
											train_sentenceFeatures.shape[1:],train_essayFeatures.shape[1:], 
											train_feat.shape[1:],train_read_feat.shape[1:])
	
	evl = Evaluator(args.prompt_id, score_index,overall_score_column,out_dir, checkpoint_dir, modelname,
				train_essay_repr, dev_essay_repr, test_essay_repr,train_feat, dev_feat, test_feat,
				train_read_feat, dev_read_feat, test_read_feat, 
				train_sentenceFeatures, dev_sentenceFeatures, test_sentenceFeatures,
				train_essayFeatures, dev_essayFeatures, test_essayFeatures,
				train_norm_score, test_norm_score, dev_norm_score)


	train_sentenceFeatures_ = train_sentenceFeatures
	train_essayFeatures_ = train_essayFeatures
	train_feat_ = train_feat
	train_read_feat_ = train_read_feat
	train_essay_repr_ = train_essay_repr

	print("Train Norm Shape: ", train_norm_score.shape)
	#print(train_true_score)

	if args.loss == 'regression':
		train_norm_score_ = train_norm_score # np.transpose(train_norm_score)
		#train_norm_score_ = train_norm_score_.tolist()
		shuffle=True

	total_train_time = 0
	total_eval_time = 0

	# Train model
	#h = model.fit(train_essay_repr_, validation_data=dev_essay_repr, epochs=2)

	history=[]

	for ii in range(args.num_epochs):
		logger.info('Running Prompt %s' % (prompt_id))
		logger.info('Epoch %s/%s' % (str(ii+1), args.num_epochs))
		start_time = time.time()
		logger.info("Using %s Loss" % str(args.loss))
		if args.loss == 'ranking' or args.loss == 'regression_and_ranking':
			perm = np.random.permutation(train_essay_repr.shape[0])
			train_essay_repr_ = train_essay_repr[perm]
			'''
			train_essay_repr_ = {
				'input_ids': train_essay_repr['input_ids'][perm],
				'input_mask': train_essay_repr['input_mask'][perm],
    		}
			'''
			train_norm_score_ = train_norm_score[perm]
			if args.sent_feat_flag:
				train_sentenceFeatures_ = train_sentenceFeatures[perm,:]
			if args.essay_feat_flag:
				train_essayFeatures_ = train_essayFeatures[perm,:]
			if args.ridley_feat_flag:
				train_feat_ = train_feat[perm, :]
				train_read_feat_ = train_read_feat[perm, :]
			shuffle=False

			#train_norm_score_ = train_norm_score_.tolist()

		
		#Fiting the model 
		t0 = time.time()
		if args.loss == 'ranking':
			loss_fn = ranking_loss
		elif args.loss == 'regression':
			loss_fn = 'mse'
		elif args.loss == 'regression_and_ranking':
			loss_fn = regression_and_ranking(ii+1, args.num_epochs)
		# optimizer = RMSprop(lr=4e-5)

		#COmpile the model
		compile_start_time = time.time()
		model.compile(optimizer='rmsprop', loss=loss_fn, loss_weights=[score_weight])
		total_compile_time = time.time() - compile_start_time
		logger.info("Model compiled in %.3f s" % total_compile_time)


		model.fit([train_essay_repr_, train_sentenceFeatures_, train_essayFeatures_, train_feat_, train_read_feat_], 
					train_norm_score_, batch_size=args.batch_size, epochs=1, verbose=0, shuffle=shuffle)
		tr_time = time.time() - t0

		total_train_time += tr_time
		tt_time = time.time() - start_time
		logger.info("Training one epoch in %.3f s" % tt_time)

		t0 = time.time()
		evl.evaluate(model, ii+1)
		evl_time = time.time() - t0
		logger.info("Evaluating one epoch in %.3f s" % evl_time)

		total_eval_time += evl_time
		evl.print_info()
	
	logger.info('Training:   %i seconds in total' % total_train_time)
	logger.info('Evaluation: %i seconds in total' % total_eval_time)

	evl.print_final_info()
	#evl.predict_final_score(model)
	total_run_time = time.time() - t0_run_time
	logger.info("Total time taken in executing the program: %.3f s" % total_run_time)

if __name__ == '__main__':
	main()


