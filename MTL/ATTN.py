import os
import sys
import argparse
import random
import time
import numpy as np
from utils import *
#from networks.CTS import build_CTS
from networks.hier_networks import build_hrcnn_model
import data_prepare
from evaluator import Evaluator
from keras.models import Model
import joblib
import pickle

from hand_crafted_features import read_hand_crafted_features, read_word_feat, read_sent_feat, read_essay_feat


logger = get_logger("ATTN")
np.random.seed(100)


def main():
	#arguments
	
	t0_run_time = time.time()

	parser = argparse.ArgumentParser(description="sentence Hi_CNN model")
	parser.add_argument('--train_flag', action='store_true', help='Train or eval')
	parser.add_argument('--fine_tune', action='store_true', help='Fine tune word embeddings')
	parser.add_argument('--embedding', type=str, default='glove', help='Word embedding type, word2vec, senna or glove')
	parser.add_argument('--embedding_dict', type=str,default='glove/glove.6B.50d.txt.gz', help='Pretrained embedding path')
	parser.add_argument('--embedding_dim', type=int, default=50, help='Only useful when embedding is randomly initialised')
	parser.add_argument('--char_embedd_dim', type=int, default=30, help='char embedding dimension if using char embedding')
	parser.add_argument('--use_char', action='store_false', help='Whether use char embedding or not')
	parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for training')
	parser.add_argument('--batch_size', type=int, default=1, help='Number of texts in each batch')
	parser.add_argument('--word_feat', type=str, help='Word Features')
	parser.add_argument('--sent_feat', type=str, help='Sentence Features')
	parser.add_argument('--essay_feat', type=str, help='Essay Features')
	parser.add_argument('--hand_feat', type=str, help='Hand crafted Features')
	parser.add_argument('--readability_feat', type=str, help='Readability Features')
	parser.add_argument('--prompt_filePath', type=str, help='For cross domain, prompt-specific vocab')
	parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")
	
	parser.add_argument("--auxiliary_task_weight",type=float, help='auxiliary task weight',nargs='+')
	parser.add_argument("--auxiliary_task_column",help='auxiliary task columns',nargs='+')
	parser.add_argument("--primary_task_weight",type=float, help='primary_task weight')
	parser.add_argument("--primary_task_column",help='primary_task_columns')

	parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, 
						help="The path to the output directory")
	parser.add_argument("-wt", "--wt-dir",default='model_weights', dest="model_weights", help="Model weights for each traits")
	parser.add_argument('--score_index', type=int, default=6, help='Overall Score Column index')
	parser.add_argument('--nbfilters', type=int, default=100, help='Num of filters in conv layer')
	parser.add_argument('--char_nbfilters', type=int, default=20, help='Num of char filters in conv layer')
	parser.add_argument('--filter1_len', type=int, default=5, help='filter length in 1st conv layer')
	parser.add_argument('--filter2_len', type=int, default=3, help='filter length in 2nd conv layer or char conv layer')
	parser.add_argument('--lstm_units', type=int, default=100, help='Num of hidden units in recurrent layer')
	parser.add_argument('--optimizer', choices=['sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop'], help='updating algorithm', default='rmsprop')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
	parser.add_argument('--rnn_type', type=str, default='BiLSTM', choices=['LSTM', 'BiLSTM'], help='Recurrent type')

	parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for layers')
	parser.add_argument('--oov', choices=['random', 'embedding'], default='embedding', help="Embedding for oov word")
	parser.add_argument('--l2_value', type=float, help='l2 regularizer value')
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
	prompt_filePath = args.prompt_filePath

	#output dir. for preds files
	out_dir = args.out_dir_path
	mkdir_p(out_dir + '/preds')
	set_logger(out_dir)

	# Required Attributes for Multi-task learning 
	overall_loss_weight = int(args.primary_task_weight)
	overall_score_column = int(args.primary_task_column)
	attr_loss_weights = args.auxiliary_task_weight
	attr_score_columns = args.auxiliary_task_column#[int(args.attr_column)+overall_score_column]

	modelname = "attn-%s.prompt%s.%sfilters.bs%s.hdf5" % (args.mode, args.prompt_id, args.nbfilters, batch_size)
	imgname = "attn-%s.prompt%s.%sfilters.bs%s.png" % (args.mode, args.prompt_id, args.nbfilters, batch_size)

	#Model name for each traits
	model_wt_dir = args.model_weights
	mkdir_p(model_wt_dir + '/Prompt-'+ str(args.prompt_id))
	# set_logger(model_wt_dir)
	tr_modelname = [("attn-%s.prompt%s.%sfilters.bs%s.trait%s.hdf5" % (args.mode, args.prompt_id, args.nbfilters, batch_size, i)) 
							for i in range(len(attr_loss_weights))]

	# print("trait_modelname: ", tr_modelname)
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


	word_path = args.word_feat
	sentence_path = args.sent_feat
	# essayFeaturesPath = [word_path, sentence_path]
	wordFeaturesDict = [] #read_word_feat(args.word_feat)
	sentenceFeaturesDict = [] #read_sent_feat(args.sent_feat)
	essayFeaturesDict = [] #read_essay_feat(args.essay_feat)

	# Preparing Train, Dev and Test Datasets
	(X_train, Y_train, train_sentenceFeatures, train_essayFeatures, train_essayIDList, mask_train, train_true_score, train_norm_score), \
	 (X_dev, Y_dev, dev_sentenceFeatures, dev_essayFeatures, dev_essayIDList, mask_dev, dev_true_score, dev_norm_score), \
	  (X_test, Y_test, test_sentenceFeatures, test_essayFeatures, test_essayIDList, mask_test, test_true_score, test_norm_score), \
			vocab, vocab_size, embed_table, overal_maxlen, overal_maxnum, init_mean_value = data_prepare.prepare_sentence_data(datapaths, prompt_filePath, \
			attr_score_columns, overall_score_column, wordFeaturesDict, sentenceFeaturesDict, essayFeaturesDict, \
			embedding_path, embedding, embedd_dim, prompt_id, args.vocab_size, tokenize_text=True, \
			to_lower=True, sort_by_len=False, vocab_path=None, score_index=6)


	# print type(embed_table)
	if embed_table is not None:
		embedd_dim = embed_table.shape[1]
		embed_table = [embed_table]
		
	logger.info("Embedding dim: %s" % str(embedd_dim))
	max_sentnum = overal_maxnum
	max_sentlen = overal_maxlen
	# train_sentenceFeatures = np.array(train_sentenceFeatures)
	# dev_sentenceFeatures = np.array(dev_sentenceFeatures)
	# test_sentenceFeatures = np.array(test_sentenceFeatures)
	# logger.info("train_sentenceFeatures shape: %s" % str(train_sentenceFeatures.shape))
	# logger.info("dev_sentenceFeatures shape: %s"% str(dev_sentenceFeatures.shape))
	# logger.info("test_sentenceFeatures shape: %s" % str(test_sentenceFeatures.shape))

	# train_essayFeatures = np.array(train_essayFeatures)
	# dev_essayFeatures = np.array(dev_essayFeatures)
	# test_essayFeatures = np.array(test_essayFeatures)
	# logger.info("train_essayFeatures shape: %s" % str(train_essayFeatures.shape))
	# logger.info("dev_essayFeatures shape: %s"% str(dev_essayFeatures.shape))
	# logger.info("test_essayFeatures shape: %s" % str(test_essayFeatures.shape))


	#Reshaping all the train, dev and test files into 2d array from 3d array
	X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
	X_dev = X_dev.reshape((X_dev.shape[0], X_dev.shape[1]*X_dev.shape[2]))
	X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

	#Hand Crafted Features
	# hand_feat_path = args.hand_feat
	# readability_path = args.readability_feat
	# feat_train, feat_dev, feat_test = read_hand_crafted_features(datapaths,hand_feat_path,readability_path, prompt_id)
	# train_feat, dev_feat, test_feat = feat_train['features_x'], feat_dev['features_x'], feat_test['features_x'] 
	# train_read_feat, dev_read_feat, test_read_feat = feat_train['readability_x'], feat_dev['readability_x'], feat_test['readability_x'] 


	logger.info("X_train shape: %s" % str(X_train.shape))
	logger.info("X_dev_shape: %s"% str(X_dev.shape))
	logger.info("X_test_shape: %s" % str(X_test.shape))

	# logger.info("train_linguistic_feat shape: %s" % str(train_feat.shape))
	# logger.info("dev_linguistic_feat shape: %s"% str(dev_feat.shape))
	# logger.info("test_linguistic_feat shape: %s" % str(test_feat.shape))
	
	# logger.info("train_readability_feat_shape: %s" % str(train_read_feat.shape))
	# logger.info("dev_readability_feat_shape: %s"% str(dev_read_feat.shape))
	# logger.info("test_readability_feat_shape: %s" % str(test_read_feat.shape))

	train_norm_score = np.array(train_norm_score)
	dev_norm_score = np.array(dev_norm_score)
	test_norm_score = np.array(test_norm_score)

	# Defining Model
	# model = build_CTS(vocab_size,max_sentnum, max_sentlen, train_sentenceFeatures.shape[1:], train_essayFeatures.shape[1:], train_read_feat.shape[1],
	# 	  train_feat.shape[1], len(attr_loss_weights)+1, embedd_dim, args.dropout, args.nbfilters, args.filter1_len, args.lstm_units)
	model = build_hrcnn_model(args,attr_loss_weights,overall_loss_weight,
									 args.rnn_type, vocab_size, max_sentnum, max_sentlen, embedd_dim, embed_table, True, init_mean_value)

	#Initilalize all the parameters 
	evl = Evaluator(args.prompt_id, args.use_char,out_dir, checkpoint_dir,model_wt_dir, modelname, tr_modelname,
					X_train, X_dev, X_test,
					Y_train,Y_dev, Y_test,
					train_norm_score, test_norm_score, dev_norm_score, 
					overall_score_column, attr_score_columns, 6)

	# Initial evaluation
	logger.info("Initial evaluation: ")
	#evl.evaluate(model, -1, print_info=True)
	logger.info("Train model")
	print(train_norm_score.shape)	

	#Env - tensorflow == 1.14
	train_norm_score = train_norm_score.tolist()

	#Env - tensorflow == 2.3.1
	#train_norm_score = np.transpose(train_norm_score)


	total_train_time = 0
	total_eval_time = 0
 
	for ii in range(args.num_epochs):
		logger.info('Epoch %s/%s' % (str(ii+1), args.num_epochs))
		start_time = time.time()

		t0 = time.time()
		model.fit(X_train, train_norm_score , batch_size=args.batch_size, epochs=1, verbose=0, shuffle=True)

		tr_time = time.time() - t0
		total_train_time += tr_time
		tt_time = time.time() - start_time
		logger.info("Training one epoch in %.3f s" % tt_time)

		t0 = time.time()
		evl.evaluate(model, ii+1)

		evl_time = time.time() - t0
		total_eval_time += evl_time
		evl.print_info()
	
	logger.info('Training:   %i seconds in total' % total_train_time)
	logger.info('Evaluation: %i seconds in total' % total_eval_time)

	evl.print_final_info()
	evl.predict_final_score(model)
	total_run_time = time.time() - t0_run_time
	logger.info("Total time taken in executing the program: %.3f s" % total_run_time)

if __name__ == '__main__':
	main()


