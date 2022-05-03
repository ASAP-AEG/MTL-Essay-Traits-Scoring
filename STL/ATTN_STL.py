# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-01-10 11:57:22
# @Last Modified by:   rokeer
# @Last Modified time: 2018-11-11 16:26:13


import os
import sys
sys.path.append('/home/rahulee16/Rahul/System/')
import argparse
import random
import time
import numpy as np
from utils import *
import data_prepare_STL as data_prepare

#Self-Attention Without traits
from networks.hier_networks_STL import build_hrcnn_model
from evaluator_STL import Evaluator

from hand_crafted_features import read_hand_crafted_features, read_word_feat, read_sent_feat, read_essay_feat

logger = get_logger("ATTN")
np.random.seed(100)


def main():

	t0_run_time = time.time()
	#arguments
	parser = argparse.ArgumentParser(description="sentence Hi_CNN model")
	parser.add_argument('--train_flag', action='store_true', help='Train or eval')
	parser.add_argument('--fine_tune', action='store_true', help='Fine tune word embeddings')
	parser.add_argument('--embedding', type=str, default='glove', help='Word embedding type, word2vec, senna or glove')
	parser.add_argument('--embedding_dict', type=str, default='../glove/glove.6B.50d.txt.gz', help='Pretrained embedding path')
	parser.add_argument('--embedding_dim', type=int, default=50, help='Only useful when embedding is randomly initialised')
	parser.add_argument('--char_embedd_dim', type=int, default=30, help='char embedding dimension if using char embedding')

	parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for training')
	parser.add_argument('--batch_size', type=int, default=1, help='Number of texts in each batch')
	parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")


	parser.add_argument('--score_index', type=int, default=6, help='Overall Score Column')
	parser.add_argument('--score_weight', type=float, default=1.0, help='Overall Score Weight')
	parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, 
						help="The path to the output directory")
	
	parser.add_argument('--word_feat_path', type=str,default=None, help='Word Features')
	parser.add_argument('--sent_feat_path', type=str,default=None, help='Sentence Features')
	parser.add_argument('--essay_feat_path', type=str, help='Hand crafted Essay Features')
	parser.add_argument('--ridley_feat_path', help='Hand crafted Ridely Essay Features', nargs='+')

	parser.add_argument('--word_feat_flag', type=int,default=0, help='Use Word Features or not')
	parser.add_argument('--sent_feat_flag', type=int,default=0, help='Use Sentence Features or not')
	parser.add_argument('--essay_feat_flag', type=int,default=0, help='Use Hand crafted Essay Features or not')
	parser.add_argument('--ridley_feat_flag', type=int,default=0, help='Use Hand crafted Ridely Essay Features or not')

	
	parser.add_argument('--nbfilters', type=int, default=100, help='Num of filters in conv layer')
	parser.add_argument('--char_nbfilters', type=int, default=20, help='Num of char filters in conv layer')
	parser.add_argument('--filter1_len', type=int, default=5, help='filter length in 1st conv layer')
	parser.add_argument('--filter2_len', type=int, default=3, help='filter length in 2nd conv layer or char conv layer')
	parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'Bi-LSTM'], help='Recurrent type')
	parser.add_argument('--lstm_units', type=int, default=100, help='Num of hidden units in recurrent layer')

	# parser.add_argument('--project_hiddensize', type=int, default=100, help='num of units in projection layer')
	parser.add_argument('--optimizer', choices=['sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop'], help='updating algorithm', default='rmsprop')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
	parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for layers')
	parser.add_argument('--oov', choices=['random', 'embedding'], default='embedding', help="Embedding for oov word")
	parser.add_argument('--l2_value', type=float, help='l2 regularizer value')
	parser.add_argument('--checkpoint_path', type=str, help='checkpoint directory', default='checkpoints')

	parser.add_argument('--prompt_filePath', type=str, help='Prompt specific file', default='data/1.txt')
	parser.add_argument('--train', type=str, help='train file', default='data/fold_0/train.tsv')  # "data/word-level/*.trpreprocess_asap.pyain"
	parser.add_argument('--dev', type=str, help='dev file', default='data/fold_0/dev.tsv')
	parser.add_argument('--test', type=str, help='test file', default='data/fold_0/test.tsv')
	parser.add_argument('--prompt_id', type=int, default=1, help='prompt id of essay set')
	parser.add_argument('--init_bias', action='store_true', help='init the last layer bias with average score of training data')
	parser.add_argument('--mode', type=str, choices=['mot', 'att', 'merged'], default='att', \
						help='Mean-over-Time pooling or attention-pooling, or two pooling merged')

	args = parser.parse_args()
	train_flag = args.train_flag
	fine_tune = args.fine_tune

	batch_size = args.batch_size
	checkpoint_dir = args.checkpoint_path
	num_epochs = args.num_epochs

	#output dir. for preds files
	out_dir = args.out_dir_path
	mkdir_p(out_dir + '/preds')
	set_logger(out_dir)

	overall_score_column = 6
	score_index = args.score_index
	score_weight = args.score_weight
	mkdir_p(out_dir + '/Prompt-'+str(args.prompt_id))


	modelname = "attn-%s.prompt%s.%sfilters.bs%s.hdf5" % (args.mode, args.prompt_id, args.nbfilters, batch_size)
	imgname = "attn-%s.prompt%s.%sfilters.bs%s.png" % (args.mode, args.prompt_id, args.nbfilters, batch_size)

	modelpath = os.path.join(checkpoint_dir, modelname)
	imgpath = os.path.join(checkpoint_dir, imgname)

	datapaths = [args.train, args.dev, args.test]
	embedding_path = args.embedding_dict
	oov = args.oov
	embedding = args.embedding
	embedd_dim = args.embedding_dim
	prompt_id = args.prompt_id

	# debug mode
	# debug = True
	# if debug:
	# 	nn_model = build_concat_model(args, args.vocab_size, 71, 20, embedd_dim, None, True)

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

	# print type(embed_table)
	if embed_table is not None:
		embedd_dim = embed_table.shape[1]
		embed_table = [embed_table]
		
	max_sentnum = overal_maxnum
	max_sentlen = overal_maxlen

	#print(train_norm_score[:10])
	#print(train_true_score[:10])
	#print(dev_norm_score)
	#print(dev_true_score)
	##print(test_norm_score[:10])
	#print(test_true_score[:10])
	#exit()

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
	


	#Reshaping all the train, dev and test files into 2d array from 3d array
	X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
	X_dev = X_dev.reshape((X_dev.shape[0], X_dev.shape[1]*X_dev.shape[2]))
	X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
	#logger.info("X_train shape: %s" % str(X_train.shape))

	#Hand Crafted Features
	hand_feat_path = args.ridley_feat_path[0]
	readability_path = args.ridley_feat_path[1]
	feat_train, feat_dev, feat_test = read_hand_crafted_features(datapaths,hand_feat_path,readability_path, prompt_id)
	train_feat, dev_feat, test_feat = feat_train['features_x'], feat_dev['features_x'], feat_test['features_x'] 
	train_read_feat, dev_read_feat, test_read_feat = feat_train['readability_x'], feat_dev['readability_x'], feat_test['readability_x'] 
	#X_train = np.concatenate([X_train, train_feat], axis=1)
	#X_dev = np.concatenate([X_dev, dev_feat], axis=1)
	#X_test = np.concatenate([X_test, test_feat], axis=1)


	logger.info("X_train shape: %s" % str(X_train.shape))
	logger.info("X_dev_shape: %s"% str(X_dev.shape))
	logger.info("X_test_shape: %s" % str(X_test.shape))

	logger.info("train_feat shape: %s" % str(train_feat.shape))
	logger.info("dev_feat_shape: %s"% str(dev_feat.shape))
	logger.info("test_feat_shape: %s" % str(test_feat.shape))
	
	logger.info("train_read_feat_shape: %s" % str(train_read_feat.shape))
	logger.info("dev_read_feat_shape: %s"% str(dev_read_feat.shape))
	logger.info("test_read_feat_shape: %s" % str(test_read_feat.shape))


	train_norm_score = np.array(train_norm_score)
	dev_norm_score = np.array(dev_norm_score)
	test_norm_score = np.array(test_norm_score)

	# Defining Model
	model = build_hrcnn_model(args,score_weight,train_sentenceFeatures.shape[1:],train_essayFeatures.shape[1:],
							train_feat.shape[1:],train_read_feat.shape[1:], 
							args.rnn_type, vocab_size, 
							max_sentnum, max_sentlen, embedd_dim, embed_table, True, init_mean_value)

	#exit()
	#Initilalize all the parameters 
	evl = Evaluator(args.prompt_id,score_index,overall_score_column, out_dir, checkpoint_dir, modelname,
					X_train, X_dev, X_test, train_feat, dev_feat, test_feat,
					train_read_feat, dev_read_feat, test_read_feat,
					train_sentenceFeatures, dev_sentenceFeatures, test_sentenceFeatures,
					train_essayFeatures, dev_essayFeatures, test_essayFeatures,
					train_norm_score, test_norm_score, dev_norm_score)

	# Initial evaluation
	logger.info("Initial evaluation: ")
	evl.evaluate(model, -1, print_info=True)
	logger.info("Train model")


	train_norm_score = (train_norm_score).tolist()

	total_train_time = 0
	total_eval_time = 0

	for ii in range(args.num_epochs):
		logger.info('Epoch %s/%s' % (str(ii+1), args.num_epochs))
		start_time = time.time()
			
		# Without traits only Self-attn model 
		t0 = time.time()
		model.fit([X_train,train_sentenceFeatures, train_essayFeatures, train_feat,train_read_feat],
					train_norm_score,batch_size=args.batch_size,epochs=1,verbose=0, shuffle=True)
		tr_time = time.time() - t0
		total_train_time += tr_time

		tt_time = time.time() - start_time
		logger.info("Training one epoch in %.3f s" % tt_time)

		t0 = time.time()
		evl.evaluate(model, ii+1)

		evl_time = time.time() - t0
		total_eval_time += evl_time
		evl.print_info()
	# model.save(model_wt_dir + '/Prompt-'+ str(args.prompt_id)+'/'+'final_model_prompt_id_'+str(args.prompt_id)+'.h5')

	logger.info('Training:   %i seconds in total' % total_train_time)
	logger.info('Evaluation: %i seconds in total' % total_eval_time)

	evl.print_final_info()
	#evl.predict_final_score(model)
	total_run_time = time.time() - t0_run_time
	logger.info("Total time taken in executing the program: %.3f s" % total_run_time)

if __name__ == '__main__':
	main()


