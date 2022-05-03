import tensorflow_hub as hub
from official.nlp.data import classifier_data_lib
from official.nlp.bert import tokenization
from official.nlp import optimization
import tensorflow as tf
import numpy as np

from transformers import TFBertModel, BertTokenizerFast, BertTokenizer

max_bert_tokens = 512
batch = 64
label_list = np.arange(61).tolist()


#: https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2
bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2', trainable=False)
vocab = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab, do_lower_case)


'''
bert_layer2 = TFBertModel.from_pretrained('bert-base-uncased', trainable=False)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
'''
'''
# This provides a function to convert row to input features and label
def to_feature(text, label, label_list=label_list, max_seq_length=max_seq_len, tokenizer=tokenizer):
   example = classifier_data_lib.InputExample(guid=None,
											  text_a=text.numpy(),
											  text_b=None,
											  label=label.numpy()
											  )
   feature = classifier_data_lib.convert_single_example(0, example, label_list, max_seq_length, tokenizer)
   return (feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id)

def to_feature_map(text, label):
  input_ids, input_mask, segment_ids, label_id = tf.py_function(to_feature, inp=[text, label],
																Tout=[tf.int32, tf.int32, tf.int32, tf.int32]
																)
  input_ids.set_shape([max_seq_len])
  input_mask.set_shape([max_seq_len])
  segment_ids.set_shape([max_seq_len])
  label_id.set_shape([])

  x = {
	  'input_word_ids': input_ids,
	   'input_mask':input_mask,
	   'input_type_ids':segment_ids
  }
  return (x, label_id)
'''

def convert_sentence_to_features(sentence, tokenizer, max_seq_len):
	tokens = ['[CLS]']
	tokens.extend(tokenizer.tokenize(sentence))
	if len(tokens) > max_seq_len-1:
		tokens = tokens[:max_seq_len-1]
	tokens.append('[SEP]')
	
	segment_ids = [0] * len(tokens)
	input_ids = tokenizer.convert_tokens_to_ids(tokens)
	input_mask = [1] * len(input_ids)

	#Zero Mask till seq_length
	zero_mask = [0] * (max_seq_len-len(tokens))
	input_ids.extend(zero_mask)
	input_mask.extend(zero_mask)
	segment_ids.extend(zero_mask)
	
	return input_ids, input_mask, segment_ids

def convert_sentences_to_features(sentences, tokenizer, max_seq_len=20):
	all_input_ids = []
	all_input_mask = []
	all_segment_ids = []
	
	for sentence in sentences:
		input_ids, input_mask, segment_ids = convert_sentence_to_features(sentence, tokenizer, max_seq_len)
		all_input_ids.append(input_ids) #Avoiding CLS token in each sentence.
		all_input_mask.append(input_mask)
		all_segment_ids.append(segment_ids) 
	
	#CLS token should be added every input example
	##all_input_ids.insert(0, tokenizer.convert_tokens_to_ids(['CLS']))
	#all_input_mask.insert(0, 1)
	#all_segment_ids.insert(0, 0)

	all_input_ids = np.array(all_input_ids).flatten()
	all_input_mask = np.array(all_input_mask).flatten()
	all_segment_ids = np.array(all_segment_ids).flatten()

	

	if(len(all_input_ids) < max_bert_tokens):
		all_input_ids = np.pad(all_input_ids, pad_width=(0, max_bert_tokens - len(all_input_ids)))

	if(len(all_input_mask) < max_bert_tokens):
		all_input_mask = np.pad(all_input_mask, pad_width=(0, max_bert_tokens - len(all_input_mask)))

	if(len(all_segment_ids) < max_bert_tokens):
		all_segment_ids = np.pad(all_segment_ids, pad_width=(0, max_bert_tokens - len(all_segment_ids)))

	return all_input_ids, all_input_mask, all_segment_ids

def convert_essays_to_features(essays, tokenizer, max_seq_len=20):
	all_input_ids = []
	all_input_mask = []
	all_segment_ids = []
	
	for essay in essays:
		#print(len(essay))
		input_ids, input_mask, segment_ids = convert_sentences_to_features(essay, tokenizer, max_seq_len)
		all_input_ids.append(input_ids[:max_bert_tokens])
		all_input_mask.append(input_mask[:max_bert_tokens])
		all_segment_ids.append(segment_ids[:max_bert_tokens])
	
	return all_input_ids, all_input_mask, all_segment_ids

def findMaximumSeqLen(essays):
	max_seq_len=0
	for essay in essays:
		for sentence in essay:
			tokens = ['[CLS]']
			tokens.extend(tokenizer.tokenize(sentence))
			tokens.append('[SEP]')
			max_seq_len = max(max_seq_len, len(tokens))
	return max_seq_len

# Building the model
def create_model(score_weight,input_shape, sentenceFeatures_shape, essayFeatures_shape,linguistic_shape,readability_shape):
	input_word_ids = tf.keras.layers.Input(shape=input_shape, dtype=tf.int32,
										name="input_word_ids")
	input_mask = tf.keras.layers.Input(shape=input_shape, dtype=tf.int32,
									name="input_mask")
	segment_ids = tf.keras.layers.Input(shape=input_shape, dtype=tf.int32,
										name="segment_ids")

	features_input = tf.keras.layers.Input(shape=linguistic_shape, dtype='float32', name='features_input')
	readability_features_input = tf.keras.layers.Input(shape=readability_shape, dtype='float32', name='readability_features_input')
	sentence_feat_input = tf.keras.layers.Input(shape=sentenceFeatures_shape, dtype='float32', name='sentence_feat_input')
	essay_feat_input = tf.keras.layers.Input(shape=essayFeatures_shape, dtype='float32', name='essay_feat_input')
	
	pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
	flat = tf.squeeze(pooled_output[:, 0:1, :], axis=1)
	#bert_output = bert_layer2([input_word_ids])
	#flat = tf.keras.layers.Flatten()(bert_output.last_hidden_state)

	drop = tf.keras.layers.Dropout(0.5)(flat)
	#dense_1 = tf.keras.layers.Dense(512, activation='relu')(drop)
	#dense_2 = tf.keras.layers.Dense(1024, activation='relu')(dense_1)
	#dense_3 = tf.keras.layers.Dense(1024, activation='relu')(dense_2)
	#drop = tf.keras.layers.Dropout(0.4)(dense_3)
	output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(drop)

	model = tf.keras.Model(
		inputs=[{
		'input_word_ids': input_word_ids,
		'input_mask': input_mask,
		'input_type_ids': segment_ids
		}, sentence_feat_input, essay_feat_input, features_input, readability_features_input],
		outputs=output
  	)

	model.summary()

	return model