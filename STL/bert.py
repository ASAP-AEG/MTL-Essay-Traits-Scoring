import tensorflow as tf
import numpy as np
from transformers import TFBertModel, LongformerModel, TFDistilBertModel, BertConfig, TFRobertaModel
import joblib
import time

from keras.models import *
from keras.optimizers import *
from keras.layers.core import *
from keras.layers import Input, Dense, Concatenate, Bidirectional, LSTM, Dropout
import keras.backend as K

# config = BertConfig.from_pretrained('../../../../../Models/BERT', output_hidden_states=True, output_attentions=True)
#bert_layer = TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config, trainable=False) 
# bert_layer = TFBertModel.from_pretrained('../../../../../Models/BERT',config=config, trainable=False)
bert_layer = TFBertModel.from_pretrained('bert-base-uncased', cache_dir="/data/rahulk/transformer_models/", trainable=False)

def ranking_loss(y_true, y_pred):
	
	y_pred = tf.convert_to_tensor(y_pred)
	y_true = tf.cast(y_true, y_pred.dtype)
	#assert y_true.shape == y_pred.shape
	y_true = K.exp(y_true) / K.sum(K.exp(y_true), axis=0)
	y_pred = K.exp(y_pred) / K.sum(K.exp(y_pred), axis=0)
	
	return -K.sum(y_true * K.log(y_pred), axis=-1)
	

def regression_loss(y_true, y_pred):
	
	y_pred = tf.convert_to_tensor(y_pred)
	y_true = tf.cast(y_true, y_pred.dtype)
	return K.mean(tf.math.squared_difference(y_pred, y_true), axis=-1)

def regression_and_ranking(current_epoch, total_epoch):

	#print("Calling reegr_rank loss")
	#print("Current Epochs: ", current_epoch)
	#print("Total Epochs: ", total_epoch)

	def r_square(y_true, y_pred):

		#print("Calling r_square loss")
		
		L_m = regression_loss(y_true, y_pred)
		L_r = ranking_loss(y_true, y_pred)
	
		t_1 = 0.000001
		gamma = K.log(1/t_1 - 1) / (total_epoch/2 - 1)
		#gamma = 0.99999 #as per paper 2020.findings.emnlp
	
		t_e = 1 / ( 1 + K.exp(gamma*(total_epoch/2 - current_epoch)))
		loss = t_e * L_m + (1-t_e) * L_r
		return loss

	return r_square

def load_bert_embedd(path, essayIDList):
	
	essay_repr=[]
	for essayID in essayIDList:
		bert_repr = np.load(path+'/'+str(essayID)+'.txt-bert_essay_repr.npy')[0]
		#print(bert_repr.shape)
		#bert_repr = bert_repr.ravel()
		#if bert_repr.shape[0]!=512*768:
		#	bert_repr = np.pad(bert_repr, (0,512*768-bert_repr.shape[0]), 'constant')
		essay_repr.append(bert_repr[0]) #Considering first token only from bert representations
	return np.array(essay_repr)

def bert_model(bert_tokens):
	
	bert_max_seq_len = 512	
	bert_tensor = tf.constant(bert_tokens[:, :bert_max_seq_len])
	#print("bert_tensor: ", bert_tensor)
	input_ids = tf.convert_to_tensor(bert_tensor, dtype=tf.int32)
	print('input_ds_shape: ', input_ids.shape)
	model = TFBertModel.from_pretrained('bert-base-uncased') #TFBertModel.from_pretrained('../../../../../Models/BERT/')
	essay_repr_arr=[]
	count=1
	for input in bert_tokens[:, :bert_max_seq_len]:
		#print("Essay Count: ", count)
		bert_tensor = tf.constant([input])
		input_ids = tf.convert_to_tensor(bert_tensor, dtype=tf.int32)
		last_hidden_states = model(input_ids)
		essay_repr = last_hidden_states[0].numpy()
		#print('essay_rer_shape: ', essay_repr[0].shape)
		essay_repr_arr.append(essay_repr[0])
		count+=1
	return np.array(essay_repr_arr)

	#WOrking Code
	########################################
	#import torch
	#bert_embedd = torch.load('../glove/bert-embeddings/embd/bert-base-uncased-word_embeddings.pkl')
	#print(bert_embedd)
	#inputs = torch.tensor(bert_tokens, dtype=torch.long, device='cpu')
	#essay_repr = bert_embedd(inputs)
	#return essay_repr.detach().numpy()
	######################################
	
	#model = TFLongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096')
	#bert_tokens = np.expand_dims(bert_tokens, 1)
	#input_ids = tf.convert_to_tensor(bert_tokens, dtype=tf.int32)	
	#print("bert_tokens.shape: ", bert_tokens.shape)
	#outputs = model(input_ids)
	#sequence_output = outputs.last_hidden_state
	#print(sequence_output.shape)
	##return sequence_output

def build_bert_model(opts, score_weight, input_shape, 
								sentenceFeatures_shape, essayFeatures_shape,linguistic_shape,readability_shape):

	#from tensorflow.keras.layers import Concatenate

	

	attribute_scores = []
	print("Input Shape: " ,input_shape)

	#word_input = Input(shape=input_shape, dtype='int32', name='word_input')
	input_word_ids = Input(shape=input_shape, dtype='int32', name="input_word_ids")
	input_mask = Input(shape=input_shape, dtype='int32', name="input_mask")
	features_input = Input(shape=linguistic_shape, dtype='float32', name='features_input')
	readability_features_input = Input(shape=readability_shape, dtype='float32', name='readability_features_input')
	sentence_feat_input = Input(shape=sentenceFeatures_shape, dtype='float32', name='sentence_feat_input')
	essay_feat_input = Input(shape=essayFeatures_shape, dtype='float32', name='essay_feat_input')

	bert_output = bert_layer(input_word_ids)[0]
	flat = K.squeeze(bert_output[:, 0:1, :], axis=1) #Considering [CLS] token only
	#flat = tf.keras.layers.Flatten()(bert_output.last_hidden_state)
	# drop = Dropout(0.5)(flat)
	concat_input=flat #word_input
	if opts.ridley_feat_flag and opts.essay_feat_flag:
		print("Adding Ridely and Essay features")
		concat_input = Concatenate()([concat_input, essay_feat_input, readability_features_input])
	elif opts.essay_feat_flag:
		print("Adding essay features")
		concat_input = Concatenate()([essay_feat_input, concat_input])
	elif opts.ridley_feat_flag:
		print("Adding essay features")
		concat_input = Concatenate()([readability_features_input, concat_input])
	elif opts.sent_feat_flag:
		print("Adding sentence features")
		concat_input = Concatenate()([sentence_feat_input, concat_input])

	overall_score = Dense(units=1, activation='sigmoid', name='output')(concat_input)
	model = Model(inputs=[input_word_ids, sentence_feat_input, essay_feat_input, features_input, readability_features_input], outputs=overall_score)
	
	#if opts.loss == 'ranking':
	#	loss_fn = ranking_loss
	#elif opts.loss == 'regression':
	#	loss_fn = 'mse'
	#elif opts.loss == 'regression_and_ranking':
	#	print("Calling opts.loss loss")
	#	current_epoch = K.variable(1.)
	#	loss_fn = regression_and_ranking(current_epoch, opts.num_epochs)

	#start_time = time.time()
	#optimizer = RMSprop(lr=0.00004, rho=0.9, clipnorm=10)
	#model.compile(optimizer=optimizer, loss=[loss_fn]+['mse' for _ in range(len(attr_loss_weights))],
	#				loss_weights=[overall_loss_weight]+attr_loss_weights)
	#total_time = time.time() - start_time
	#print("Model compiled in: ", total_time)
	model.summary()
	return model
