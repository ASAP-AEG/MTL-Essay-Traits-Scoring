# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-01-05 20:15:33
# @Last Modified by:   rokeer
# @Last Modified time: 2018-11-11 16:26:13

import random
import codecs
import sys
import nltk
import os
# import logging
import re
import numpy as np
import pickle as pk
import utils
import keras.backend as K

url_replacer = '<url>'
logger = utils.get_logger("Loading data...")
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'

MAX_SENTLEN = 50
MAX_SENTNUM = 100

#BERT
USE_BERT = False
if(USE_BERT):
	from transformers import BertTokenizer
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#Sentence-Transformers
#from sentence_transformers import SentenceTransformer

asap_ranges = {
	0: (0, 60),
	1: (2, 12),
	2: (1, 6),
	3: (0, 3),
	4: (0, 3),
	5: (0, 4),
	6: (0, 4),
	7: (0, 30),
	8: (0, 60),
	9: (0, 1)   
}

trait_ranges = {
	0: (0, 12),
	1: (1, 6),
	2: (1, 6),
	3: (0, 3),
	4: (0, 3),
	5: (0, 4),
	6: (0, 4),
	7: (0, 6),
	8: (0, 12)
}

num_traits = {
	0: 6,
	1: 5,
	2: 5,
	3: 4,
	4: 4,
	5: 4,
	6: 4,
	7: 4,
	8: 6
}


#Used for multi-task loss while compiling the multi-task learning model.
def multitask_loss(y_true, y_pred):
	# Avoid divide by 0
	y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
	# Multi-task loss
	return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))


def get_ref_dtype():
	return ref_scores_dtype

#Tokenize the strings
def tokenize(string):
	tokens = nltk.word_tokenize(string)
	for index, token in enumerate(tokens):
		if token == '@' and (index+1) < len(tokens):
			tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
			tokens.pop(index)
	return tokens


def get_score_range(prompt_id):
	return asap_ranges[prompt_id]

def get_trait_score_range(prompt_id):
	return trait_ranges[prompt_id]

def get_model_friendly_scores(scores_array, prompt_id):
	arg_type = type(prompt_id)
	assert arg_type in {int, np.ndarray}
	if arg_type is int:
		low, high = asap_ranges[prompt_id]
		scores_array = (scores_array - low) / (high - low)
	return scores_array

def get_trait_friendly_scores(scores_array, prompt_id_array):
	arg_type = type(prompt_id_array)
	assert arg_type in {int, np.ndarray}
	if arg_type is int:
		low, high = trait_ranges[prompt_id_array]
		scores_array = (scores_array - low) / (high - low)
	return scores_array

def get_model_and_trait_friendly_scores(scores_array, prompt_id, overall_score_column, attr_score_columns, original_score_index):
	norm_score= []
	#We are interested to use any traits as primary and as auxiliary task.
	#For that original_score_index = 6, overall_score_column = whichever trait we want to make primary
	#This is for overall score column if it is equal to 6 then we normalize the value according to the asap ranges otherwise trait ranges.
	all_col = [overall_score_column]+attr_score_columns
	all_col = [int(col) for col in all_col]
	#Similarily for other attributes columns but now this columns might contain col 6 so normalize it according to the asap ranges.
	for i, score_array in zip(all_col, scores_array):
		if i == original_score_index:
			norm_score.append(get_model_friendly_scores(score_array, prompt_id))
		else:
			norm_score.append(get_trait_friendly_scores(score_array, prompt_id))
	
	return norm_score


def unify_training_scores(score, prompt_id, essay_set, score_index, overall_score_column):
    if essay_set == 9:
        if score_index==overall_score_column: low, high = asap_ranges[prompt_id] #this is when score column is considered as STL
        else: low, high = trait_ranges[prompt_id] ##this is when trait columns considered as STL
        new_score = (score * (high - low)) + low
        return np.round(new_score)
    return score

def unify_trait_scores(score, prompt_id, essay_set):
    if essay_set == 9:
        min, max = trait_ranges[prompt_id]
        new_score = (score * (max - min)) + min
        return new_score
    return score



def convert_to_dataset_friendly_scores(scores_array, prompt_id_array):
	arg_type = type(prompt_id_array)
	assert arg_type in {int, np.ndarray}
	if arg_type is int:
		low, high = asap_ranges[prompt_id_array]
		scores_array = scores_array * (high - low) + low
		assert np.all(scores_array >= low) and np.all(scores_array <= high)
	else:
		assert scores_array.shape[0] == prompt_id_array.shape[0]
		dim = scores_array.shape[0]
		low = np.zeros(dim)
		high = np.zeros(dim)
		for ii in range(dim):
			low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
		scores_array = scores_array * (high - low) + low
	return scores_array

def is_number(token):
	return bool(num_regex.match(token))

def load_vocab(vocab_path):
	logger.info('Loading vocabulary from: ' + vocab_path)
	with open(vocab_path, 'rb') as vocab_file:
		vocab = pk.load(vocab_file)
	return vocab

def create_vocab(file_path,prompt_filePath, prompt_id, vocab_size, tokenize_text, to_lower):
	logger.info('Creating vocabulary from: ' + file_path)
	total_words, unique_words = 0, 0
	word_freqs = {}
	with codecs.open(file_path, mode='r', encoding='utf-8') as input_file:
		next(input_file)
		for line in input_file:
			tokens = line.strip().split('\t')
			essay_id = int(tokens[0])
			essay_set = int(tokens[1])
			content = tokens[2].strip()
			score = float(tokens[6])
			if essay_set == prompt_id or prompt_id <= 0 or essay_set==9:
				if tokenize_text:
					content = text_tokenizer(content, True, True, True)
				if to_lower:
					content = [w.lower() for w in content]
				for word in content:
					try:
						word_freqs[word] += 1
					except KeyError:
						unique_words += 1
						word_freqs[word] = 1
					total_words += 1

	
	logger.info('  %i total words, %i unique words' % (total_words, unique_words))
	import operator
	sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
	if vocab_size <= 0:
		# Choose vocab size automatically by removing all singletons
		vocab_size = 0
		for word, freq in sorted_word_freqs:
			if freq > 1:
				vocab_size += 1
	vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
	vcb_len = len(vocab)
	index = vcb_len
	for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
		vocab[word] = index
		index += 1

	if(essay_set == 9):
		index = len(vocab)
		cross_dom_vocab_freqs={}
		unique_words=0
		total_words=0
		logger.info('Creating other vocabulary for cross-domain from: ' + prompt_filePath)
		with codecs.open(prompt_filePath, mode='r', encoding='utf-8') as input_file:
			next(input_file)
			for line in input_file:
				tokens = line.strip().split('\t')
				word = tokens[0]
				if to_lower:
					word = tokens[0].lower()
				if word in cross_dom_vocab_freqs:
					cross_dom_vocab_freqs[word] += 1
				else:
					unique_words += 1
					cross_dom_vocab_freqs[word] = 1
				total_words += 1
		for word, _ in cross_dom_vocab_freqs.items():
			if word not in vocab.keys():
				vocab[word] = index
				index += 1

	return vocab

def create_char_vocab(file_path, prompt_id, tokenize_text, to_lower):
	logger.info("Create char vocabulary from: %s" % file_path)
	total_chars, unique_chars = 0, 0
	char_vocab = {}
	start_index = 1
	char_vocab['<unk>'] = start_index
	next_index = start_index + 1
	with codecs.open(file_path, 'r', encoding='utf-8') as input_file:
		next(input_file)
		for line in input_file:
			tokens = line.strip().split('\t')
			essay_id = int(tokens[0])
			essay_set = int(tokens[1])
			content = tokens[2].strip()
			score = float(tokens[6])
			if essay_set == prompt_id or prompt_id <= 0:
				if tokenize_text:
					content = text_tokenizer(content, True, True, True)
				if to_lower:
					content = [w.lower() for w in content]
				for word in content:
					for char in list(word):
						if not char in char_vocab:
							char_vocab[char] = next_index
							next_index += 1
							unique_chars += 1
						total_chars += 1
	logger.info('  %i total chars, %i unique chars' % (total_chars, unique_chars))
	return char_vocab

def read_essays(file_path, prompt_id):
	logger.info('Reading tsv from: ' + file_path)
	essays_list = []
	essays_ids = []
	with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
		input_file.next()
		for line in input_file:
			tokens = line.strip().split('\t')
			if int(tokens[1]) == prompt_id or prompt_id <= 0:
				essays_list.append(tokens[2].strip())
				essays_ids.append(int(tokens[0]))
	return essays_list, essays_ids

def replace_url(text):
	replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', url_replacer, text)
	return replaced_text

def text_tokenizer(text, replace_url_flag=True, tokenize_sent_flag=True, create_vocab_flag=False):
	text = replace_url(text)
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

	# TODO here
	tokens = tokenize(text)
	if tokenize_sent_flag:
		text = " ".join(tokens)
		sent_tokens = tokenize_to_sentences(text, MAX_SENTLEN, create_vocab_flag)
		# print sent_tokens
		# sys.exit(0)
		# if not create_vocab_flag:
		#     print "After processed and tokenized, sentence num = %s " % len(sent_tokens)
		return sent_tokens
	else:
		raise NotImplementedError

def tokenize_to_sentences(text, max_sentlength, create_vocab_flag=False):

	# tokenize a long text to a list of sentences
	sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)

	# Note
	# add special preprocessing for abnormal sentence splitting
	# for example, sentence1 entangled with sentence2 because of period "." connect the end of sentence1 and the begin of sentence2
	# see example: "He is running.He likes the sky". This will be treated as one sentence, needs to be specially processed.
	processed_sents = []
	for sent in sents:
		if re.search(r'(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent):
			s = re.split(r'(?=.{2,})(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent)
			# print sent
			# print s
			ss = " ".join(s)
			ssL = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', ss)

			processed_sents.extend(ssL)
		else:
			processed_sents.append(sent)

	if create_vocab_flag:
		sent_tokens = [tokenize(sent) for sent in processed_sents]
		tokens = [w for sent in sent_tokens for w in sent]
		# print tokens
		return tokens

	# TODO here
	sent_tokens = []
	for sent in processed_sents:
		shorten_sents_tokens = shorten_sentence(sent, max_sentlength)
		sent_tokens.extend(shorten_sents_tokens)
	# if len(sent_tokens) > 90:
	#     print len(sent_tokens), sent_tokens
	return sent_tokens

def shorten_sentence(sent, max_sentlen):
	# handling extra long sentence, truncate to no more extra max_sentlen
	new_tokens = []
	sent = sent.strip()
	tokens = nltk.word_tokenize(sent)
	if len(tokens) > max_sentlen:
		# print len(tokens)
		# Step 1: split sentence based on keywords
		# split_keywords = ['because', 'but', 'so', 'then', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
		split_keywords = ['because', 'but', 'so', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
		k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
		processed_tokens = []
		if not k_indexes:
			num = (int) (len(tokens) / max_sentlen)
			k_indexes = [(i+1)*max_sentlen for i in range(num)]

		processed_tokens.append(tokens[0:k_indexes[0]])
		len_k = len(k_indexes)
		for j in range(len_k-1):
			processed_tokens.append(tokens[k_indexes[j]:k_indexes[j+1]])
		processed_tokens.append(tokens[k_indexes[-1]:])

		# Step 2: split sentence to no more than max_sentlen
		# if there are still sentences whose length exceeds max_sentlen
		for token in processed_tokens:
			if len(token) > max_sentlen:
				num = (int) (len(token) / max_sentlen)
				s_indexes = [(i+1)*max_sentlen for i in range(num)]

				len_s = len(s_indexes)
				new_tokens.append(token[0:s_indexes[0]])
				for j in range(len_s-1):
					new_tokens.append(token[s_indexes[j]:s_indexes[j+1]])
				new_tokens.append(token[s_indexes[-1]:])

			else:
				new_tokens.append(token)
	else:
			return [tokens]

	# print "Before processed sentences length = %d, after processed sentences num = %d " % (len(tokens), len(new_tokens))
	return new_tokens

def read_dataset(file_path,attr_score_columns, overall_score_column, EssayID_wordFeaturesDict, EssayID_sentenceFeaturesDict, EssayID_essayFeaturesDict, prompt_id, vocab, to_lower, char_level=False):
	logger.info('Reading dataset from: ' + file_path)

	data_x, data_y, prompt_ids, data_traits = [], [], [], []
	num_hit, unk_hit, total = 0., 0., 0.
	max_sentnum = -1
	max_sentlen = -1
	true_score = []
	norm_score = []

	wordFeaturesDict = {}
	sentenceFeatures = []
	sent_essayIDList, essayFeaturesIDList, essayIDList=[], [], []
	#create lists for attribute scores
	for _ in range(len(attr_score_columns) + 1):
		true_score.append([])
		norm_score.append([])

	with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
		next(input_file)
		logger.info('Reading the traits of an essay: ')
		count=0
		for line in input_file:
			tokens = line.strip().split('\t')
			essay_id = int(tokens[0])
			essay_set = int(tokens[1])
			content = tokens[2].strip()
			score = float(tokens[overall_score_column])

			# if essay_set == prompt_id or prompt_id <= 0 or essay_set == 9:
			# 	score = unify_training_scores(score, prompt_id, essay_set, score_index, overall_score_column)


			essayWordList = []
			if essay_set == prompt_id or prompt_id <= 0 or essay_set == 9:
				# tokenize text into sentences

				sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
				if to_lower:
					sent_tokens = [[w.lower() for w in s] for s in sent_tokens]
				if char_level:
					raise NotImplementedError

				sent_indices = []
				indices = []
				if char_level:
					raise NotImplementedError
				else:
					for sent in sent_tokens:
						length = len(sent)
						if max_sentlen < length:
							max_sentlen = length

						for word in sent:
							if is_number(word):
								indices.append(vocab['<num>'])
								num_hit += 1
							elif word in vocab:
								indices.append(vocab[word])
							else:
								indices.append(vocab['<unk>'])
								unk_hit += 1
							total += 1
							#handCraftedWordeatures
							essayWordList.append(word)

						sent_indices.append(indices)
						indices = []

				# if essay_id in EssayID_wordFeaturesDict:
				# 	##print("len(wordFeaturesDict[essay_id]): ", len(wordFeaturesDict[essay_id]))
				# 	#print("word_count_inEssay: ", word_count_inEssay)
				# 	if len(EssayID_wordFeaturesDict[essay_id]) == len(essayWordList):
				# 		for word, features in zip(essayWordList, EssayID_wordFeaturesDict[essay_id]):
				# 			features = features.split(',')
				# 			wordFeaturesDict[word] = [np.float(feat) for feat in features]
				# 	else:
				# 		for word in essayWordList:
				# 			features = EssayID_wordFeaturesDict[essay_id][0]
				# 			feat_len = len(features)
				# 			wordFeaturesDict[word] = [0.0 for _ in range(feat_len)]

				
				if essay_id in EssayID_sentenceFeaturesDict:
					sent_essayIDList.append(essay_id)
				if essay_id in EssayID_essayFeaturesDict:
					essayFeaturesIDList.append(essay_id)
				
				essayIDList.append(essay_id)
			
				data_x.append(sent_indices)
				data_y.append(score)
				prompt_ids.append(prompt_id)
				# data_traits.append(traits)

				 # print ("tokens[overall_score_column]: ", tokens[overall_score_column])
				true_score_v = score
				true_score[0].append(true_score_v)
				#overall_min_score, overall_max_score = asap_ranges[prompt_id]
				#norm_score_v = (true_score_v - overall_min_score) / (overall_max_score - overall_min_score)
				#norm_score[0].append(norm_score_v)
				
				count = count+1
				#  read attribute scores
				attr_min_score, attr_max_score = trait_ranges[prompt_id]
				for i, col in enumerate(attr_score_columns,0):
					if essay_set == 9:
						true_score_v = unify_trait_scores(float(tokens[int(col)]), prompt_id, essay_set)
					else: 
						true_score_v = float(tokens[int(col)])
					true_score[i + 1].append(true_score_v)
					#norm_score_v = (true_score_v - attr_min_score) / (attr_max_score - attr_min_score)
					#norm_score[i + 1].append(norm_score_v)

				if max_sentnum < len(sent_indices):
					max_sentnum = len(sent_indices)
		
	
	logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))
	return data_x, data_y,wordFeaturesDict, sent_essayIDList, essayIDList, essayFeaturesIDList, prompt_ids, data_traits, max_sentlen, max_sentnum, np.array(true_score)


def get_handCrafted_sentence_features(essayIDList, EssayID_sentenceFeaturesDict, overall_maxsentnum):
	
	sentenceFeatures = [] 
	for essay_id in essayIDList:
		total_sent_feat = []
		for sent_feat in EssayID_sentenceFeaturesDict[essay_id]:
			sent_feat = sent_feat.split(',')
			sent_features = [float(s) for s in sent_feat]
			total_sent_feat.append(sent_features)
			feat_len = len(sent_features)
		if len(total_sent_feat) < overall_maxsentnum:
			for _ in range(overall_maxsentnum - len(total_sent_feat)):
				zeros = [0.0 for _ in range(feat_len)]
				total_sent_feat.append(zeros)
		sentenceFeatures.append(total_sent_feat)

	return sentenceFeatures

def get_handCrafted_essay_features(essayIDList, EssayID_essayFeaturesDict):
	
	essayFeatures = [] 
	#print("essayIDList: ", essayIDList)
	#print("overall_maxsentnum: ", overall_maxsentnum)
	for essay_id in essayIDList:
		essay_feat = EssayID_essayFeaturesDict[essay_id]
		essayFeatures.append(essay_feat)
		feat_len = len(essay_feat)

	return essayFeatures

def get_data(paths, prompt_filePath, attr_score_columns, overall_score_column, EssayID_wordFeaturesDict, EssayID_sentenceFeaturesDict, EssayID_essayFeaturesDict, prompt_id, vocab_size, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=None):
	train_path, dev_path, test_path = paths[0], paths[1], paths[2]

	logger.info("Prompt id is %s" % prompt_id)
	if not vocab_path:
		vocab = create_vocab(train_path, prompt_filePath, prompt_id, vocab_size, tokenize_text, to_lower)
		#print("Word_freqs: ", word_freqs)
		if len(vocab) < vocab_size:
			logger.warning('The vocabulary includes only %i words (less than %i)' % (len(vocab), vocab_size))
		
	else:
		vocab = load_vocab(vocab_path)
		if len(vocab) != vocab_size:
			logger.warning('The vocabulary includes %i words which is different from given: %i' % (len(vocab), vocab_size))
	logger.info('  Vocab size: %i' % (len(vocab)))

	train_x, train_y,train_wordFeaturesDict, train_sent_essayIDList, train_essayIDList, train_essayFeaturesIDList, train_prompts, train_traits, train_maxsentlen, train_maxsentnum, train_true_score = read_dataset(train_path, attr_score_columns, overall_score_column,EssayID_wordFeaturesDict, EssayID_sentenceFeaturesDict, EssayID_essayFeaturesDict, prompt_id, vocab, to_lower)
	dev_x, dev_y,dev_wordFeaturesDict, dev_sent_essayIDList, dev_essayIDList, dev_essayFeaturesIDList, dev_prompts, dev_traits, dev_maxsentlen, dev_maxsentnum, dev_true_score = read_dataset(dev_path, attr_score_columns, overall_score_column, EssayID_wordFeaturesDict, EssayID_sentenceFeaturesDict, EssayID_essayFeaturesDict, prompt_id, vocab, to_lower)
	test_x, test_y,test_wordFeaturesDict,test_sent_essayIDList, test_essayIDList, test_essayFeaturesIDList, test_prompts, test_traits, test_maxsentlen, test_maxsentnum, test_true_score = read_dataset(test_path, attr_score_columns, overall_score_column, EssayID_wordFeaturesDict, EssayID_sentenceFeaturesDict, EssayID_essayFeaturesDict, prompt_id, vocab,  to_lower)

	overal_maxlen = max(train_maxsentlen, dev_maxsentlen, test_maxsentlen)
	overal_maxnum = max(train_maxsentnum, dev_maxsentnum, test_maxsentnum)

	#Sentence Features HandCrafted
	train_sentenceFeatures = [] #get_handCrafted_sentence_features(train_sent_essayIDList, EssayID_sentenceFeaturesDict, overal_maxnum)
	dev_sentenceFeatures = [] #get_handCrafted_sentence_features(dev_sent_essayIDList, EssayID_sentenceFeaturesDict, overal_maxnum)
	test_sentenceFeatures = [] #get_handCrafted_sentence_features(test_sent_essayIDList, EssayID_sentenceFeaturesDict, overal_maxnum)

	train_essayFeatures = [] #get_handCrafted_essay_features(train_essayFeaturesIDList, EssayID_essayFeaturesDict)
	dev_essayFeatures = [] #get_handCrafted_essay_features(dev_essayFeaturesIDList, EssayID_essayFeaturesDict)
	test_essayFeatures = [] #get_handCrafted_essay_features(test_essayFeaturesIDList, EssayID_essayFeaturesDict)
	

	logger.info("Training data max sentence num = %s, max sentence length = %s" % (train_maxsentnum, train_maxsentlen))
	logger.info("Dev data max sentence num = %s, max sentence length = %s" % (dev_maxsentnum, dev_maxsentlen))
	logger.info("Test data max sentence num = %s, max sentence length = %s" % (test_maxsentnum, test_maxsentlen))
	logger.info("Overall max sentence num = %s, max sentence length = %s" % (overal_maxnum, overal_maxlen))

	return (train_x, train_y, train_wordFeaturesDict,  train_sentenceFeatures, train_essayFeatures, train_essayIDList, train_prompts,train_true_score), (dev_x, dev_y, dev_wordFeaturesDict, dev_sentenceFeatures, dev_essayFeatures, dev_essayIDList, dev_prompts,dev_true_score), (test_x, test_y, test_wordFeaturesDict, test_sentenceFeatures, test_essayFeatures, test_essayIDList, test_prompts,test_true_score), vocab, overal_maxlen, overal_maxnum

def get_essay_data(paths, max_sentnum, prompt_id, embedd_dim=50):

	
	model = SentenceTransformer('distilbert-base-nli-mean-tokens')

	train_x, dev_x, test_x = [], [], []
	embedd_dict, essay_sent_token = {}, {}
	sent_num = 1

	for path in paths:
		with codecs.open(path, mode='r', encoding='UTF8') as input_file:
			next(input_file)
			logger.info('Reading dataset from: ' + path)
			basename = os.path.splitext(os.path.basename(path))[0]	

			for line in input_file:
				tokens = line.strip().split('\t')
				essay_id = int(tokens[0])
				essay_set = int(tokens[1])
				content = tokens[2].strip()

				if essay_set == prompt_id or prompt_id <= 0:

					text = replace_url(content)
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
					
					sent_list = nltk.tokenize.sent_tokenize(text)
					
					sentence_embeddings = model.encode(sent_list)

					##Padding Essays for equal no. of sentences in each essays
					if(sentence_embeddings.shape[0] < max_sentnum):
						zero_padding = np.zeros((max_sentnum - sentence_embeddings.shape[0], sentence_embeddings.shape[1]))
						sentence_embeddings = np.concatenate((sentence_embeddings, zero_padding))
					
					sent_tokens = []

					for sent, embedd in zip(sent_list, sentence_embeddings):
							embedd_dict[sent] = embedd[:embedd_dim]
							essay_sent_token[sent] = sent_num
							sent_tokens.append(sent_num)
							sent_num+=1
					
					if basename == 'train': 
						train_x.append(sent_tokens)
											
					elif basename == 'dev': dev_x.append(sent_tokens)
					elif basename == 'test': test_x.append(sent_tokens)
	
	return np.array(train_x), np.array(dev_x), np.array(test_x), embedd_dict, essay_sent_token
