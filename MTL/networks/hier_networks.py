# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-01-10 11:40:53
# @Last Modified by:   rokeer
# @Last Modified time: 2018-11-11 16:26:13

from keras.models import *
from keras.optimizers import *
from keras.layers.core import *
from keras.layers import Input, Embedding, LSTM, Dense, merge, Concatenate, RepeatVector, Add
from keras.layers import TimeDistributed, Conv1D, Bidirectional, Flatten
#from tensorflow.keras import regularizers

from keras.layers.convolutional import MaxPooling1D, AveragePooling1D
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.regularizers import l2
import keras.backend as K

from networks.softattention import Attention, CoAttention, CoAttentionWithoutBi
from networks.zeromasking import ZeroMaskedEntries
from utils import get_logger
import time
from networks.matrix_attention import MatrixAttention
from networks.masked_softmax import MaskedSoftmax
from networks.weighted_sum import WeightedSum
from networks.max import Max
from networks.repeat_like import RepeatLike
from networks.complex_concat import ComplexConcat

#from LambdaRankTF.lambdarank import *
#from networks.attention import Attention_CTS

# from networks.BertLayer import BertLayer
# from bert_embedding import BertEmbedding
#import tensorflow_ranking as tfr
import tensorflow as tf

import numpy as np
logger = get_logger("Build model")

"""
Hierarchical networks, the following function contains several models:
(1)build_hcnn_model: hierarchical CNN model
(2)build_hrcnn_model: hierarchical Recurrent CNN model, LSTM stack over CNN,
 it supports two pooling methods
	(a): Mean-over-time pooling
	(b): attention pooling
(3)build_shrcnn_model: source based hierarchical Recurrent CNN model, LSTM stack over CNN
"""

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
	
		t_1 = 0.000001
		gamma = K.log(1/t_1 - 1) / (total_epoch/2 - 1)
		#gamma = 0.99999 #as per paper 2020.findings.emnlp
	
		t_e = 1 / ( 1 + K.exp(gamma*(total_epoch/2 - cur_epoch)))
		loss = t_e * L_m + (1-t_e) * L_r
		return loss

	return r_square


def build_hcnn_model(opts, vocab_size=0, maxnum=50, maxlen=50, embedd_dim=50, embedding_weights=None, verbose=False):

	N = maxnum
	L = maxlen

	logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, filter2_len = %s, drop rate = %s, l2 = %s" % (N, L, embedd_dim,
		opts.nbfilters, opts.filter1_len, opts.filter2_len, opts.dropout, opts.l2_value))

	word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
	x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, name='x')(word_input)
	drop_x = Dropout(opts.dropout, name='drop_x')(x)

	resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

	z = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='z')(resh_W)

	avg_z = TimeDistributed(AveragePooling1D(pool_length=L-opts.filter1_len+1), name='avg_z')(z)	# shape= (N, 1, nbfilters)

	resh_z = Reshape((N, opts.nbfilters), name='resh_z')(avg_z)		# shape(N, nbfilters)

	hz = Conv1D(opts.nbfilters, opts.filter2_len, padding='valid', name='hz')(resh_z)
	# avg_h = MeanOverTime(mask_zero=True, name='avg_h')(hz)

	avg_hz = GlobalAveragePooling1D(name='avg_hz')(hz)
	y = Dense(output_dim=1, activation='sigmoid', name='output')(avg_hz)

	model = Model(input=word_input, output=y)

	if verbose:
		model.summary()

	start_time = time.time()
	model.compile(loss='mse', optimizer='rmsprop')
	total_time = time.time() - start_time
	logger.info("Model compiled in %.4f s" % total_time)

	return model


def build_hrcnn_model(opts,attr_loss_weights,overall_loss_weight, rnn_type, vocab_size=0, maxnum=50, maxlen=50, embedd_dim=50, embedding_weights=None, verbose=False, init_mean_value=None):
	# LSTM stacked over CNN based on sentence level
	N = maxnum
	L = maxlen

	logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s" % (N, L, embedd_dim,
		opts.nbfilters, opts.filter1_len, opts.dropout))

	#linguistic_feature_count=51
	#readability_feature_count=35
	
	word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
	# features_input = Input(shape=linguistic_shape, dtype='float32', name='features_input')
	# readability_features_input = Input(shape=readability_shape, dtype='float32', name='readability_features_input')
	# sentence_feat_input = Input(shape=sentenceFeatures_shape, dtype='float32', name='sentence_feat_input')
	# essay_feat_input = Input(shape=essayFeatures_shape, dtype='float32', name='essay_feat_input')


	x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
	x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
	drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)
	resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

	#zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)
	#logger.info('Use attention-pooling on sentence')
	#avg_zcnn = TimeDistributed(Attention(), name='avg_zcnn')(zcnn)


	attribute_scores=[]
	pos_avg_hz_lstm = []
	for i in range(len(attr_loss_weights)+1):
		print ("stack: ",i)

		# add char-based CNN, concatenating with word embedding to compose word representation
		if opts.use_char:
			char_input = Input(shape=(N*L*maxcharlen,), dtype='int32', name='char_input')
			xc = Embedding(output_dim=opts.char_embedd_dim, input_dim=char_vocabsize, input_length=N*L*maxcharlen, mask_zero=True, name='xc')(char_input)
			xc_masked = ZeroMaskedEntries(name='xc_masked')(xc)
			drop_xc = Dropout(opts.dropout, name='drop_xc')(xc_masked)
			res_xc = Reshape((N*L, maxcharlen, opts.char_embedd_dim), name='res_xc')(drop_xc)
			cnn_xc = TimeDistributed(Conv1D(opts.char_nbfilters, opts.filter2_len, padding='valid'), name='cnn_xc')(res_xc)
			max_xc = TimeDistributed(GlobalMaxPooling1D(), name='avg_xc')(cnn_xc)
			res_xc2 = Reshape((N, L, opts.char_nbfilters), name='res_xc2')(max_xc)

			w_repr = merge([resh_W, res_xc2], mode='concat', name='w_repr')
			zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(w_repr)
		else:
			logger.info("Use CNN")
			zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn'+str(i))(resh_W)

		# pooling mode
		if opts.mode == 'mot':
			logger.info("Use mean-over-time pooling on sentence")
			avg_zcnn = TimeDistributed(GlobalAveragePooling1D(), name='avg_zcnn')(zcnn)
		elif opts.mode == 'att':
			#logger.info('Using CNN Shared')
			logger.info('Use attention-pooling on sentence')
			avg_zcnn = TimeDistributed(Attention(), name='avg_zcnn'+str(i))(zcnn)
			#logger.info('Use Dropout on sentence')
			#avg_zcnn = Dropout(rate=0.2, name='drop_CNN'+str(i))(avg_zcnn) #rate means nodes will depreciate at prob of (1-rate)
			#if opts.sent_feat_flag: 
			# logger.info('Adding sentence features')
			# avg_zcnn = Concatenate()([avg_zcnn, sentence_feat_input])
		elif opts.mode == 'merged':
			logger.info('Use mean-over-time and attention-pooling together on sentence')
			avg_zcnn1 = TimeDistributed(GlobalAveragePooling1D(), name='avg_zcnn1'+str(i))(zcnn)
			avg_zcnn2 = TimeDistributed(Attention(), name='avg_zcnn2'+str(i))(zcnn)
			# avg_zcnn = merge([avg_zcnn1, avg_zcnn2], mode='concat', name='avg_zcnn')
			avg_zcnn = Concatenate()([avg_zcnn1,avg_zcnn2])
		else:
			raise NotImplementedError

		if opts.rnn_type == 'LSTM':
			hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm'+str(i))(avg_zcnn)
		elif opts.rnn_type == 'BiLSTM':
			logger.info('Use BiLSTM on sentence')
			hz_lstm = Bidirectional(LSTM(opts.lstm_units, return_sequences=True, name='hz_bilstm'+str(i)))(avg_zcnn)

			#combine_lstm = Concatenate()([features_input, readability_features_input])
			#essay_feat_lstm1 = Bidirectional(LSTM(opts.lstm_units, return_sequences=True, name='hz_bilstm_n'+str(i)))(sentence_feat_input)
			#essay_feat_lstm = Bidirectional(LSTM(opts.lstm_units, return_sequences=True, name='hz_bilstm_n'+str(i)))(combine_lstm)
			#essay_feat_lstm = Concatenate()([essay_feat_lstm1, essay_feat_lstm2])
			

		if opts.mode == 'mot':
			logger.info('Use mean-over-time pooling on text')
			avg_hz_lstm = GlobalAveragePooling1D(name='avg_hz_lstm')(hz_lstm)
		elif opts.mode == 'att':
			logger.info('Use attention-pooling on text')
			avg_hz_lstm = Attention(name='avg_hz_lstm'+str(i))(hz_lstm)
			#logger.info('Use Dropout on text')
			#avg_hz_lstm = Dropout(rate=0.2, name='drop_LSTM'+str(i))(avg_hz_lstm)
			#avg_hz_lstm = Concatenate()([avg_hz_lstm, features_input, readability_features_input])
			#essay_avg_hz_lstm = Attention(name='essay_avg_hz_lstm'+str(i))(essay_feat_lstm)
			#if opts.essay_feat_flag: 
			# logger.info('Adding Essay features')
			# avg_hz_lstm = Concatenate()([avg_hz_lstm, essay_feat_input, readability_features_input])
		elif opts.mode == 'merged':
			logger.info('Use mean-over-time and attention-poolisng together on text')
			avg_hz_lstm1 = GlobalAveragePooling1D(name='avg_hz_lstm1'+str(i))(hz_lstm)
			avg_hz_lstm2 = Attention(name='avg_hz_lstm2'+str(i))(hz_lstm)
			# avg_hz_lstm = merge([avg_hz_lstm1, avg_hz_lstm2], mode='concat', name='avg_hz_lstm')
			avg_hz_lstm = Concatenate()([avg_hz_lstm1,avg_hz_lstm2])
		else:
			raise NotImplementedError

		pos_avg_hz_lstm.append(avg_hz_lstm)

##################################################################################################################
	#Multi-Task Learning
		if i==0:
			new_avg_hz_lstm=avg_hz_lstm
		else:
			score = Dense(units=1, activation='sigmoid')(avg_hz_lstm)
			attribute_scores.append(score)

	
	x = Concatenate()(attribute_scores + [new_avg_hz_lstm])
	overall_score = Dense(units=1, activation='sigmoid')(x)
	model = Model(inputs=word_input, outputs=[overall_score]+attribute_scores)
	optimizer = RMSprop(lr=0.001, rho=0.9, clipnorm=10)

	# if opts.loss == 'ranking':
	# 	loss_fn = ranking_loss
	# elif opts.loss == 'regression':
	# 	loss_fn = 'mse'
	# elif opts.loss == 'regression_and_ranking':
	# 	current_epoch = K.variable(0.)
	# 	loss_fn = regression_and_ranking(current_epoch, opts.num_epochs)

	start_time = time.time()
	model.compile(optimizer='rmsprop', loss=['mse' for _ in range(len(attr_loss_weights)+1)],
					loss_weights=[overall_loss_weight]+attr_loss_weights)
	total_time = time.time() - start_time
	logger.info("Model compiled in %.4f s" % total_time)
	if verbose:
	    model.summary()
	return model

	# if opts.l2_value:
	#     logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
	#     y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(avg_hz_lstm)
	# else:
	#     y = Dense(units=1, activation='sigmoid', name='output')(avg_hz_lstm)

	# if opts.use_char:
	#     model = Model(inputs=[word_input, char_input], outputs=y)
	# else:
	#     model = Model(inputs=word_input, outputs=y)

	# if opts.init_bias and init_mean_value:
	#     logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
	#     bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
	#     model.layers[-1].b.set_value(bias_value)

	# if verbose:
	#     model.summary()

	# start_time = time.time()
	# model.compile(loss='mse', optimizer='rmsprop')
	# total_time = time.time() - start_time
	# logger.info("Model compiled in %.4f s" % total_time)

	# return model


def build_shrcnn_model(opts,attr_loss_weights,overall_loss_weight, vocab_size=0, char_vocabsize=0, maxnum=50, maxlen=50, maxcnum=50, maxclen=50, maxcharlen=20, embedd_dim=50, embedding_weights=None,attr_size=4, verbose=False, init_mean_value=None):
	# LSTM stacked over CNN based on sentence level
	N = maxnum
	L = maxlen

	cN = maxcnum
	cL = maxclen

	logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s" % (N, L, embedd_dim,
		opts.nbfilters, opts.filter1_len, opts.dropout))

	word_input = Input(shape=(N * L,), dtype='int32', name='word_input')
	context_input = Input(shape=(cN*cL,), dtype='int32', name='context_input')

	emb = Embedding(output_dim=embedd_dim, input_dim=vocab_size, weights=embedding_weights, mask_zero=True, name='cx')
	cx = emb(context_input)
	cx_maskedout = ZeroMaskedEntries(name='cx_maskedout')(cx)
	drop_cx = Dropout(opts.dropout, name='drop_cx')(cx_maskedout)

	resh_C = Reshape((cN, cL, embedd_dim), name='resh_C')(drop_cx)

	czcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='czcnn')(resh_C)


	x = emb(word_input)
	x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
	drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)

	resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)

	# add char-based CNN, concatenating with word embedding to compose word representation
	zcnn = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn')(resh_W)

	'''
	encoded_essay = Reshape((zcnn.shape[1].value*zcnn.shape[2].value, opts.nbfilters))(zcnn)
	encoded_context = Reshape((czcnn.shape[1].value*czcnn.shape[2].value, opts.nbfilters))(czcnn)
	# bidaf
	# Now we compute a similarity between the passage words and the question words, and
	# normalize the matrix in a couple of different ways for input into some more layers.
	matrix_attention_layer = MatrixAttention(name='essay_context_similarity')
	# matrix_attention_layer = LinearMatrixAttention(name='passage_question_similarity')

	# Shape: (batch_size, num_passage_words, num_question_words)
	essay_context_similarity = matrix_attention_layer([encoded_essay, encoded_context])


	# Shape: (batch_size, num_passage_words, num_question_words), normalized over question
	# words for each passage word.
	essay_context_attention = MaskedSoftmax()(essay_context_similarity)
	# Shape: (batch_size, num_passage_words, embedding_dim * 2)
	weighted_sum_layer = WeightedSum(name="essay_context_vectors", use_masking=False)
	essay_context_vectors = weighted_sum_layer([encoded_context, essay_context_attention])

	
	# Min's paper finds, for each document word, the most similar question word to it, and
	# computes a single attention over the whole document using these max similarities.
	# Shape: (batch_size, num_passage_words)
	context_essay_similarity = Max(axis=-1)(essay_context_similarity)
	# Shape: (batch_size, num_passage_words)
	context_essay_attention = MaskedSoftmax()(context_essay_similarity)
	# Shape: (batch_size, embedding_dim * 2)
	weighted_sum_layer = WeightedSum(name="question_passage_vector", use_masking=False)
	context_essay_vector = weighted_sum_layer([encoded_essay, context_essay_attention])

	# Then he repeats this question/passage vector for every word in the passage, and uses it
	# as an additional input to the hidden layers above.
	repeat_layer = RepeatLike(axis=1, copy_from_axis=1)
	# Shape: (batch_size, num_passage_words, embedding_dim * 2)
	tiled_context_essay_vector = repeat_layer([context_essay_vector, encoded_essay])

	complex_concat_layer = ComplexConcat(combination='1*2,1*3', name='final_merged_passage')
	final_merged_passage = complex_concat_layer([encoded_essay,
												 essay_context_vectors,
												 tiled_context_essay_vector])
	

	complex_concat_layer = ComplexConcat(combination='1*2', name='final_merged_passage')
	final_merged_passage = complex_concat_layer([encoded_essay,
												 essay_context_vectors])


	mcnn = Reshape((zcnn.shape[1].value, zcnn.shape[2].value, opts.nbfilters), name='mcnn')(final_merged_passage)
	'''

	# pooling mode
	if opts.mode == 'mot':
		logger.info("Use mean-over-time pooling on sentence")
		avg_zcnn = TimeDistributed(GlobalAveragePooling1D(), name='avg_zcnn')(zcnn)
	elif opts.mode == 'att':
		logger.info('Use attention-pooling on sentence')
		avg_zcnn = TimeDistributed(Attention(), name='avg_zcnn')(zcnn)
		avg_czcnn = TimeDistributed(Attention(), name='avg_czcnn')(czcnn)
	elif opts.mode == 'merged':
		logger.info('Use mean-over-time and attention-pooling together on sentence')
		avg_zcnn1 = TimeDistributed(GlobalAveragePooling1D(), name='avg_zcnn1')(zcnn)
		avg_zcnn2 = TimeDistributed(Attention(), name='avg_zcnn2')(zcnn)
		avg_zcnn = merge([avg_zcnn1, avg_zcnn2], mode='concat', name='avg_zcnn')
	else:
		raise NotImplementedError

	hz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm')(avg_zcnn)
	chz_lstm = LSTM(opts.lstm_units, return_sequences=True, name='chz_lstm')(avg_czcnn)

	if opts.mode == 'mot':
		logger.info('Use mean-over-time pooling on text')
		avg_hz_lstm = GlobalAveragePooling1D(name='avg_hz_lstm')(hz_lstm)
	elif opts.mode == 'att':
		logger.info('Use co-attention on text')

		# PART 2:
		# Now we compute a similarity between the passage words and the question words, and
		# normalize the matrix in a couple of different ways for input into some more layers.
		matrix_attention_layer = MatrixAttention(name='essay_context_similarity')
		# Shape: (batch_size, num_passage_words, num_question_words)
		essay_context_similarity = matrix_attention_layer([hz_lstm, chz_lstm])

		# Shape: (batch_size, num_passage_words, num_question_words), normalized over question
		# words for each passage word.
		essay_context_attention = MaskedSoftmax()(essay_context_similarity)
		weighted_sum_layer = WeightedSum(name="essay_context_vectors", use_masking=False)
		# Shape: (batch_size, num_passage_words, embedding_dim * 2)
		weighted_hz_lstm = weighted_sum_layer([chz_lstm, essay_context_attention])

		# Min's paper finds, for each document word, the most similar question word to it, and
		# computes a single attention over the whole document using these max similarities.
		# Shape: (batch_size, num_passage_words)
		context_essay_similarity = Max(axis=-1)(essay_context_similarity)
		# Shape: (batch_size, num_passage_words)
		context_essay_attention = MaskedSoftmax()(context_essay_similarity)
		# Shape: (batch_size, embedding_dim * 2)
		weighted_sum_layer = WeightedSum(name="context_essay_vector", use_masking=False)
		context_essay_vector = weighted_sum_layer([hz_lstm, context_essay_attention])

		# Then he repeats this question/passage vector for every word in the passage, and uses it
		# as an additional input to the hidden layers above.
		repeat_layer = RepeatLike(axis=1, copy_from_axis=1)
		# Shape: (batch_size, num_passage_words, embedding_dim * 2)
		tiled_context_essay_vector = repeat_layer([context_essay_vector, hz_lstm])

		complex_concat_layer = ComplexConcat(combination='1,2,1*2,1*3', name='final_merged_passage')
		final_merged_passage = complex_concat_layer([hz_lstm,
													 weighted_hz_lstm,
													 tiled_context_essay_vector])

		avg_hz_lstm = LSTM(opts.lstm_units, return_sequences=False, name='avg_hz_lstm')(final_merged_passage)

		# hz_lstm_fr = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm_fr')(final_merged_passage)
		# hz_lstm_bk = LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm_bk', go_backwards=True)(final_merged_passage)
		# avg_hz_lstm1 = Concatenate()([hz_lstm_fr, hz_lstm_bk])
		# avg_hz_lstm = Attention(name='avg_hz_lstm')(avg_hz_lstm1)

		# avg_hz_lstm = CoAttentionWithoutBi(name='avg_hz_lstm')([hz_lstm, weighted_hz_lstm])

		# avg_hz_lstm = Attention(name='avg_hz_lstm')(avg_hz_lstm1)
	elif opts.mode == 'merged':
		logger.info('Use mean-over-time and attention-pooling together on text')
		avg_hz_lstm1 = GlobalAveragePooling1D(name='avg_hz_lstm1')(hz_lstm)
		avg_hz_lstm2 = Attention(name='avg_hz_lstm2')(hz_lstm)
		avg_hz_lstm = merge([avg_hz_lstm1, avg_hz_lstm2], mode='concat', name='avg_hz_lstm')
	else:
		raise NotImplementedError

#############################################################################################################
	#Multi-Task Learning
	# avg_hz_lstm is the final representations of essays
	attribute_scores=[]
	for i in range(len(attr_loss_weights)):
		score = Dense(units=1, activation='sigmoid')(avg_hz_lstm)
		attribute_scores.append(score)
	 # attribute_scores = Dense(units=len(attr_loss_weights), activation='sigmoid', name='attr_score')(avg_hz_lstm)
	
	x = Concatenate()(attribute_scores + [avg_hz_lstm])
	overall_score = Dense(units=1, activation='sigmoid')(x)
	model = Model(inputs=[word_input, context_input], outputs=[overall_score]+attribute_scores)
	optimizer = RMSprop(lr=0.001, rho=0.9, clipnorm=10)
	model.compile(optimizer='rmsprop', loss=['mse' for _ in range(len(attr_loss_weights)+1)],
					loss_weights=[overall_loss_weight]+attr_loss_weights)

	return model
###################################################################################################################

	# attr_score = Dense(units=attr_size, activation='relu', name='attr_score')(avg_hz_lstm)
	# attr_output = Concatenate(axis=1)([attr_score, avg_hz_lstm])
	# final_attr_score = Dense(units=attr_size, activation='relu',name='final_attr_score')(attr_output)


	# if opts.l2_value:
	#     logger.info("Use l2 regularizers, l2 value = %s" % opts.l2_value)
	#     y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(avg_hz_lstm)
	#     # y = Dense(units=1, activation='sigmoid', name='output', W_regularizer=l2(opts.l2_value))(final_attr_score)
	# else:
	#     y = Dense(units=1, activation='sigmoid', name='output')(avg_hz_lstm)
	#     # y = Dense(units=1, activation='sigmoid', name='output')(final_attr_score)

	# model = Model(inputs=[word_input, context_input], outputs=y)

	# if opts.init_bias and init_mean_value:
	#     logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
	#     bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
	#     model.layers[-1].b.set_value(bias_value)

	# if verbose:
	#     model.summary()

	# start_time = time.time()
	# model.compile(loss='mse', optimizer='rmsprop')
	# total_time = time.time() - start_time
	# logger.info("Model compiled in %.4f s" % total_time)

	# return model
