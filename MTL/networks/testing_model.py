# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-01-10 11:40:53
# @Last Modified by:   rokeer
# @Last Modified time: 2018-11-11 16:26:13

from keras.models import *
from keras.optimizers import *
from keras.layers.core import *
from keras.layers import Input, Embedding, LSTM, GRU, Dense, merge, Concatenate, RepeatVector, Add, CuDNNLSTM, CuDNNGRU
from keras.layers import TimeDistributed, Conv1D, Bidirectional

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

# from networks.BertLayer import BertLayer
# from bert_embedding import BertEmbedding


import numpy as np
logger = get_logger("Build model")




def build_hrcnn_model(opts,attr_loss_weights,overall_loss_weight, vocab_size=0, char_vocabsize=0, maxnum=50, maxlen=50, maxcharlen=20, embedd_dim=50, embedding_weights=None, verbose=False, init_mean_value=None):
	# LSTM stacked over CNN based on sentence level
	N = maxnum
	L = maxlen

	logger.info("Model parameters: max_sentnum = %d, max_sentlen = %d, embedding dim = %s, nbfilters = %s, filter1_len = %s, drop rate = %s" % (N, L, embedd_dim,
		opts.nbfilters, opts.filter1_len, opts.dropout))

	
	word_input = Input(shape=(N*L,), dtype='int32', name='word_input')
	x = Embedding(output_dim=embedd_dim, input_dim=vocab_size, input_length=N*L, weights=embedding_weights, mask_zero=True, name='x')(word_input)
	x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
	drop_x = Dropout(opts.dropout, name='drop_x')(x_maskedout)
	resh_W = Reshape((N, L, embedd_dim), name='resh_W')(drop_x)


	attribute_scores=[]
	# for i in range(len(attr_loss_weights)+1):
	# 	print ("stack: ",i)


	# add char-based CNN, concatenating with word embedding to compose word representation
	zcnn0 = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn'+str(0))(resh_W)
	logger.info('Use attention-pooling on sentence')
	avg_zcnn0 = TimeDistributed(Attention(), name='avg_zcnn'+str(0))(zcnn0)
	hz_lstm0 = Bidirectional(LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm'+str(0)))(avg_zcnn0)
	logger.info('Use attention-pooling on text')
	avg_hz_lstm0 = Attention(name='avg_hz_lstm'+str(0))(hz_lstm0)
	score = Dense(units=1, activation='relu')(avg_hz_lstm0)
	attribute_scores.append(score)

	zcnn1 = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn'+str(1))(resh_W)
	logger.info('Use attention-pooling on sentence')
	avg_zcnn1 = TimeDistributed(Attention(), name='avg_zcnn'+str(1))(zcnn1)
	hz_lstm1 = Bidirectional(LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm'+str(1)))(avg_zcnn1)
	logger.info('Use attention-pooling on text')
	avg_hz_lstm1 = Attention(name='avg_hz_lstm'+str(1))(hz_lstm1)
	score = Dense(units=1, activation='relu')(avg_hz_lstm1)
	attribute_scores.append(score)

	zcnn2 = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn'+str(2))(resh_W)
	logger.info('Use attention-pooling on sentence')
	avg_zcnn2 = TimeDistributed(Attention(), name='avg_zcnn'+str(2))(zcnn2)
	hz_lstm2 = Bidirectional(LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm'+str(2)))(avg_zcnn2)
	logger.info('Use attention-pooling on text')
	avg_hz_lstm2 = Attention(name='avg_hz_lstm'+str(2))(hz_lstm2)
	score = Dense(units=1, activation='relu')(avg_hz_lstm2)
	attribute_scores.append(score)

	zcnn3 = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn'+str(3))(resh_W)
	logger.info('Use attention-pooling on sentence')
	avg_zcnn3 = TimeDistributed(Attention(), name='avg_zcnn'+str(3))(zcnn3)
	hz_lstm3 = Bidirectional(LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm'+str(3)))(avg_zcnn3)
	logger.info('Use attention-pooling on text')
	avg_hz_lstm3 = Attention(name='avg_hz_lstm'+str(3))(hz_lstm3)
	score = Dense(units=1, activation='relu')(avg_hz_lstm3)
	attribute_scores.append(score)

	zcnn4 = TimeDistributed(Conv1D(opts.nbfilters, opts.filter1_len, padding='valid'), name='zcnn'+str(4))(resh_W)
	logger.info('Use attention-pooling on sentence')
	avg_zcnn4 = TimeDistributed(Attention(), name='avg_zcnn'+str(4))(zcnn4)
	hz_lstm4 = Bidirectional(LSTM(opts.lstm_units, return_sequences=True, name='hz_lstm'+str(4)))(avg_zcnn4)
	logger.info('Use attention-pooling on text')
	avg_hz_lstm4 = Attention(name='avg_hz_lstm'+str(4))(hz_lstm4)
	score = Dense(units=1, activation='relu')(avg_hz_lstm4)
	attribute_scores.append(score)

	
	model = Model(inputs=word_input, outputs=attribute_scores)
	optimizer = RMSprop(lr=0.001, rho=0.9, clipnorm=10)

	# if opts.init_bias and init_mean_value:
	#     logger.info("Initialise output layer bias with log(y_mean/1-y_mean)")
	#     bias_value = (np.log(init_mean_value) - np.log(1 - init_mean_value)).astype(K.floatx())
	#     model.layers[-1].b.set_value(bias_value)

	start_time = time.time()
	model.compile(optimizer='rmsprop', loss=['mse' for _ in range(len(attr_loss_weights)+1)],
					loss_weights=[overall_loss_weight]+attr_loss_weights)
	total_time = time.time() - start_time
	logger.info("Model compiled in %.4f s" % total_time)
	
	return model