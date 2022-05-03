import codecs
import pickle
import pandas as pd
import numpy as np
import gzip

from sklearn import preprocessing

def read_features(path, prompt_id, normalized_features_df, readability_features):

	out_data = {
		'readability_x': [],
		'features_x': []
	}
	#print(normalized_features_df.head())

	with codecs.open(path, mode='r', encoding='UTF8') as input_file:
		next(input_file)
		for line in input_file:
			tokens = line.strip().split('\t')
			essay_id = int(tokens[0])
			essay_set = int(tokens[1])
			#print("essay_set: ", essay_set)
			#print("essay_id: ", essay_id)
		

			feats_df = normalized_features_df[lambda df: (df['item_id'] == essay_id) & ((df['prompt_id'] == prompt_id) | (essay_set == 9))]
			#print(feats_df.head())
			if len(feats_df) != 0:
				feats_list = feats_df.values.tolist()[0][2:]
				out_data['features_x'].append(feats_list)

				item_index = np.where(readability_features[:, :1] == essay_id)
				item_row_index = item_index[0][0]
				item_features = readability_features[item_row_index][1:]
				out_data['readability_x'].append(item_features)
	
	out_data['features_x'] = np.array(out_data['features_x'])
	out_data['readability_x'] = np.array(out_data['readability_x'])
	#print(out_data['features_x'])
	#print(out_data['readability_x'])
	return out_data

def read_hand_crafted_features(datapaths,feat_path,readability_feat_path, prompt_id):
	train_path, dev_path, test_path = datapaths[0], datapaths[1], datapaths[2]
	features_df = pd.read_csv(feat_path)
	
	column_names_not_to_normalize = ['item_id', 'prompt_id', 'score']
	column_names_to_normalize = list(features_df.columns.values)
	for col in column_names_not_to_normalize:
		column_names_to_normalize.remove(col)
	final_columns = ['item_id'] + column_names_to_normalize
	normalized_features_df = None

	is_prompt_id = features_df['prompt_id'] == prompt_id
	#prompt_id_df = features_df[is_prompt_id]
	prompt_id_df = features_df
	x = prompt_id_df[column_names_to_normalize].values
	min_max_scaler = preprocessing.MinMaxScaler()
	normalized_pd1 = min_max_scaler.fit_transform(x)
	df_temp = pd.DataFrame(normalized_pd1, columns=column_names_to_normalize, index = prompt_id_df.index)
	prompt_id_df[column_names_to_normalize] = df_temp
	final_columns = ['prompt_id'] + final_columns 
	final_df = prompt_id_df[final_columns]
	if normalized_features_df is not None:
		normalized_features_df = pd.concat([normalized_features_df,final_df],ignore_index=True)
	else:
		normalized_features_df = final_df

	with open(readability_feat_path, 'rb') as fp:
		readability_features = pickle.load(fp)
	
	#print("readability_features: ", readability_features)
	#print("linguistic_features: ", normalized_features_df)
	
	train_feat = read_features(train_path,prompt_id, normalized_features_df, readability_features)
	dev_feat = read_features(dev_path,prompt_id, normalized_features_df, readability_features)
	test_feat = read_features(test_path,prompt_id, normalized_features_df, readability_features)


	return train_feat, dev_feat, test_feat

def read_word_feat(path):
	word_features_list={}
	with open(path, encoding='UTF8') as input_file:
		Lines = input_file.readlines()
		for line in Lines:
			tokens = line.strip().split('\t')
			if tokens[0] == 'essayID': continue
			essay_id = int(tokens[0])
			features = tokens[1].split(' ')
			#features = [int(s.replace(" ", "")) for s in features]
			#features = [float(s.replace(" ", "")) for s in features] 
			#features = np.array(features)
			#print(features)
			#print(len(features))
			word_features_list[essay_id] = features
	return word_features_list
				

def read_sent_feat(path):
	sentence_features_list={}
	with open(path, encoding='UTF8') as input_file:
		Lines = input_file.readlines()
		for line in Lines:
			tokens = line.strip().split('\t')
			if tokens[0] == 'essayID': continue
			essay_id = int(tokens[0])
			features = tokens[1].split(' ')
			sentence_features_list[essay_id] = features
	return sentence_features_list

def read_essay_feat(path):
	essay_features_list={}
	with open(path, encoding='UTF8') as input_file:
		Lines = input_file.readlines()
		for line in Lines:
			tokens = line.strip().split('\t')
			if tokens[0] == 'essayID': continue
			essay_id = int(tokens[0])
			features = [float(num) for num in tokens[1:]]
			essay_features_list[essay_id] = features
	return essay_features_list
	
