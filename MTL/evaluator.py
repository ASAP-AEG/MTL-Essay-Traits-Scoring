# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-02-10 14:56:57
# @Last Modified by:   rokeer
# @Last Modified time: 2018-11-11 16:26:13
from utils import rescale_tointscore, get_logger, rescale_tointscore_for_attr
from metrics import *
import numpy as np
import sys
from keras.models import load_model

logger = get_logger("Evaluate stats")


class Evaluator():

	def __init__(self, prompt_id, use_char,out_dir,new_out_dir,tr_model_dir, modelname, tr_modelname,
				train_x, dev_x, test_x, 
				train_y, dev_y, test_y,
				train_norm_score, test_norm_score, dev_norm_score,
				overall_score_column, attr_score_columns, original_score_index):
		# self.dataset = dataset
		self.prompt_id = prompt_id
		self.train_x, self.dev_x, self.test_x = train_x, dev_x, test_x
		self.train_y, self.dev_y, self.test_y = train_y, dev_y, test_y

		#Hand Crafted Features
		# self.train_feats, self.dev_feats, self.test_feats = train_feats, dev_feats, test_feats
		# self.train_read_feat, self.dev_read_feat, self.test_read_feat = train_read_feat, dev_read_feat, test_read_feat
		# self.train_sentenceFeatures, self.dev_sentenceFeatures, self.test_sentenceFeatures = train_sentenceFeatures, dev_sentenceFeatures, test_sentenceFeatures
		# self.train_essayFeatures, self.dev_essayFeatures, self.test_essayFeatures = train_essayFeatures, dev_essayFeatures, test_essayFeatures
		# self.train_y_org = rescale_tointscore(train_y, self.prompt_id)
		# self.dev_y_org = rescale_tointscore(dev_y, self.prompt_id)
		# self.test_y_org = rescale_tointscore(test_y, self.prompt_id)
		self.overall_score_column, self.attr_score_columns, self.original_score_index = overall_score_column, attr_score_columns, original_score_index
		self.train_norm_score, self.test_norm_score, self.dev_norm_score = train_norm_score, test_norm_score, dev_norm_score
		
		#Muti-task learning for Attributes 
		self.total_traits = self.train_norm_score.shape[0]
		self.train_norm_shape = self.train_norm_score.shape
		self.dev_norm_shape = self.dev_norm_score.shape
		self.test_norm_shape = self.test_norm_score.shape
		self.count=0
		self.train_y_org = rescale_tointscore_for_attr(train_norm_score,self.count, self.train_norm_shape,self.prompt_id, self.overall_score_column, self.attr_score_columns, self.original_score_index)
		self.dev_y_org = rescale_tointscore_for_attr(dev_norm_score,self.count, self.dev_norm_shape,self.prompt_id, self.overall_score_column, self.attr_score_columns, self.original_score_index)
		self.test_y_org = rescale_tointscore_for_attr(test_norm_score,self.count, self.test_norm_shape,self.prompt_id, self.overall_score_column, self.attr_score_columns, self.original_score_index)
		self.train_tr_org = np.delete(self.train_y_org,0,axis=0)
		self.dev_tr_org = np.delete(self.dev_y_org,0,axis=0)
		self.test_tr_org = np.delete(self.test_y_org,0,axis=0)

		
		self.for_attribute=0
		self.out_dir = out_dir
		self.new_out_dir = new_out_dir
		self.tr_model_dir = tr_model_dir
		self.modelname = modelname
		self.tr_modelname = tr_modelname
		self.best_dev = [-1, -1, -1, -1]
		self.best_test = [-1, -1, -1, -1]
		# self.best_dev_tr = [-1]
		# self.best_test_tr = [-1]
		# self.dump_ref_scores()

	def dump_ref_scores(self):
		np.savetxt(self.new_out_dir + '/preds/dev_ref.txt', self.dev_y_org, fmt='%i')
		np.savetxt(self.new_out_dir + '/preds/test_ref.txt', self.test_y_org, fmt='%i')
	
	def dump_predictions(self, dev_pred, test_pred, epoch):
		dev_pred = np.array(dev_pred).squeeze()
		test_pred = np.array(test_pred).squeeze()
		np.savetxt(self.new_out_dir + '/preds/dev_pred_' + str(epoch) + '.txt', dev_pred, fmt='%.8f')
		np.savetxt(self.new_out_dir + '/preds/test_pred_' + str(epoch) + '.txt', test_pred, fmt='%.8f')

	def calc_correl(self, train_pred, dev_pred, test_pred):
		self.train_pr = pearson(self.train_y_org, train_pred)
		self.dev_pr = pearson(self.dev_y_org, dev_pred)
		self.test_pr = pearson(self.test_y_org, test_pred)

		self.train_spr = spearman(self.train_y_org, train_pred)
		self.dev_spr = spearman(self.dev_y_org, dev_pred)
		self.test_spr = spearman(self.test_y_org, test_pred)

	def calc_kappa(self, train_pred, dev_pred, test_pred, weight='quadratic'):
		train_pred_int = np.rint(train_pred).astype('int32')
		dev_pred_int = np.rint(dev_pred).astype('int32')
		test_pred_int = np.rint(test_pred).astype('int32')
		#for final score
		#self.train_qwk = kappa(self.train_y_org[0], train_pred[0],weight)
		self.dev_qwk = kappa(self.dev_y_org[0], dev_pred[0],weight)
		#self.test_qwk = kappa(self.test_y_org[0], test_pred[0], weight)

		# np.savetxt('train_norm_score.csv', self.train_y_org.T, delimiter=',',header="trait 0, trait 1, trait 2, trait 3, trait 4",fmt='%d')
		# np.savetxt('dev_norm_score.csv', self.dev_y_org.T, delimiter=',',header="trait 0, trait 1, trait 2, trait 3, trait 4",fmt='%d')
		# np.savetxt('test_norm_score.csv', self.test_y_org.T, delimiter=',',header="trait 0, trait 1, trait 2, trait 3, trait 4",fmt='%d')

		#Attributes
		train_tr_pred = np.delete(train_pred,0,axis=0)
		dev_tr_pred = np.delete(dev_pred,0,axis=0)
		test_tr_pred = np.delete(test_pred,0,axis=0)
		#self.train_tr_qwk = kappa_for_traits(self.train_tr_org, train_tr_pred,self.prompt_id, weight)
		self.dev_tr_qwk = kappa_for_traits(self.dev_tr_org, dev_tr_pred,self.prompt_id, weight)
		#self.test_tr_qwk = kappa_for_traits(self.test_tr_org, test_tr_pred,self.prompt_id, weight)



	def calc_rmse(self, train_pred, dev_pred, test_pred):
		self.train_rmse = root_mean_square_error(self.train_y_org, train_pred)
		self.dev_rmse = root_mean_square_error(self.dev_y_org, dev_pred)
		self.test_rmse = root_mean_square_error(self.test_y_org, test_pred)

	def evaluate(self, model, epoch, print_info=False):
		count=1

		#Attributes
		train_pred1=np.array(model.predict(self.train_x, batch_size=32)).squeeze()
		dev_pred1=np.array(model.predict(self.dev_x, batch_size=32)).squeeze()
		test_pred1=np.array(model.predict(self.test_x, batch_size=32)).squeeze()

		#train_pred1 = np.transpose(train_pred1)
		#dev_pred1 = np.transpose(dev_pred1)
		#test_pred1 = np.transpose(test_pred1)

		# print(test_pred1)
		# print(dev_pred1)

		# self.dump_predictions(dev_pred1, test_pred1, epoch)

		# train_pred_int = rescale_tointscore(train_pred, self.prompt_id)
		# dev_pred_int = rescale_tointscore(dev_pred, self.prompt_id)
		# test_pred_int = rescale_tointscore(test_pred, self.prompt_id)
		# self.calc_correl(train_pred_int, dev_pred_int, test_pred_int)
		# self.calc_kappa(train_pred_int, dev_pred_int, test_pred_int)
		# self.calc_rmse(train_pred_int, dev_pred_int, test_pred_int)
		#print("train_pred1: ", train_pred1)

		#Attributes
		train_pred_int = rescale_tointscore_for_attr(train_pred1,count,self.train_norm_shape, self.prompt_id, self.overall_score_column, self.attr_score_columns, self.original_score_index)
		dev_pred_int = rescale_tointscore_for_attr(dev_pred1,count,self.dev_norm_shape, self.prompt_id, self.overall_score_column, self.attr_score_columns, self.original_score_index)
		test_pred_int = rescale_tointscore_for_attr(test_pred1,count,self.test_norm_shape, self.prompt_id, self.overall_score_column, self.attr_score_columns, self.original_score_index)
		self.calc_kappa(train_pred_int, dev_pred_int, test_pred_int)

		# print("pred_dev_score: ", dev_pred_int)
		# print("dev_y_score: ", self.dev_y_org)

		# print("pred_test_score: ", test_pred_int)
		# print("test_y_score: ", self.test_y_org)

		# if self.dev_qwk > self.best_dev[0]:
			# self.best_dev = [self.dev_qwk, self.dev_pr, self.dev_spr, self.dev_rmse]
			# self.best_test = [self.test_qwk, self.test_pr, self.test_spr, self.test_rmse]
			# self.best_dev_epoch = epoch
			# model.save_weights(self.out_dir + '/' + self.modelname, overwrite=True)

		

		#Attributes
		if self.dev_qwk > self.best_dev[0]:
			self.best_dev = [self.dev_qwk]
			self.best_test = [kappa(self.test_y_org[0], test_pred_int[0], 'quadratic')] #[self.test_wk]
			self.best_dev_epoch = epoch
			self.best_dev_tr = self.dev_tr_qwk
			test_tr_pred = np.delete(test_pred_int,0,axis=0)
			self.best_test_tr = kappa_for_traits(self.test_tr_org, test_tr_pred,self.prompt_id, 'quadratic') #self.test_tr_qwk
			# model.save_weights(self.tr_model_dir + '/Prompt-' + str(self.prompt_id) + '/' + 'model_'+str(0)+'.hdf5', overwrite=True)
			#with open(self.out_dir + '/' + 'output_test_score.txt', 'w') as file:
				#for x in test_pred_int[0]:
					#file.write(str(x) + "\n")
			
		
		for i in range(len(self.tr_modelname)):
			if(self.dev_tr_qwk[i] > self.best_dev_tr[i]):
				self.best_dev_tr[i] = self.dev_tr_qwk[i]
				model.save_weights(self.tr_model_dir + '/Prompt-' + str(self.prompt_id) + '/' + 'model_'+str(i+1)+'.hdf5', overwrite=True)
				with open(self.out_dir + '/' + 'trait_'+str(i+1)+'.txt', 'w') as file:
					for x in test_pred_int[i+1]:
						file.write(str(x) + "\n")
			

		# if self.dev_tr_qwk > self.best_dev_tr[0]:
		#     self.best_dev_tr = [self.dev_tr_qwk]
		#     self.best_test_tr = [self.test_tr_qwk]
		#     self.best_dev_tr_epoch = epoch
		
		if print_info:
			self.print_info()


	def print_info(self):
		# logger.info('[DEV]   QWK:  %.3f, PRS: %.3f, SPR: %.3f, RMSE: %.3f, (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f)' % (
		#     self.dev_qwk, self.dev_pr, self.dev_spr, self.dev_rmse, self.best_dev_epoch,
		#     self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3]))
		# logger.info('[TEST]  QWK:  %.3f, PRS: %.3f, SPR: %.3f, RMSE: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f)' % (
		#     self.test_qwk, self.test_pr, self.test_spr, self.test_rmse, self.best_dev_epoch,
		#     self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3]))
		
		#Attributes
		if(self.best_dev_epoch != -1):

			logger.info('[DEV]   QWK:  %.3f, (Best QWK@ %i:   {{%.3f}})' % (
				self.dev_qwk, self.best_dev_epoch,self.best_dev[0]))
			logger.info('[TEST] (Best QWK@ %i:   {{%.3f}})' % (
				self.best_dev_epoch,self.best_test[0]))
		
			print("[DEV] TRAITS_QWK", self.dev_tr_qwk)
			#print("[TEST] TRAITS_QWK", self.test_tr_qwk)

		logger.info('--------------------------------------------------------------------------------------------------------------------------')
	
	def print_final_info(self):
		logger.info('--------------------------------------------------------------------------------------------------------------------------')
		# logger.info('Missed @ Epoch %i:' % self.best_test_missed_epoch)
		# logger.info('  [TEST] QWK: %.3f' % self.best_test_missed)
		#Attributes
		logger.info('Best QWK @ Epoch %i:' % self.best_dev_epoch)
		logger.info('  [DEV]  QWK: %.3f ' % (self.best_dev[0]))
		logger.info('  [TEST] QWK: %.3f ' % (self.best_test[0]))
		# logger.info('Best TRAITS_QWK @ Epoch %i:' % self.best_dev_tr_epoch)
		# logger.info('  [DEV]  QWK: %.3f ' % (self.best_dev_tr[0]))
		# logger.info('  [TEST] QWK: %.3f ' % (self.best_test_tr[0]))
		# logger.info('Best TRAITS_QWK @ Epoch %i:' % self.best_dev_epoch)
		#print("[DEV] TRAITS_QWK", self.best_dev_tr)
		#print("[TEST] TRAITS_QWK", self.best_test_tr)

		# logger.info('Best @ Epoch %i:' % self.best_dev_epoch)
		# logger.info('  [DEV]  QWK: %.3f,  PRS: %.3f, SPR: %.3f, RMSE: %.3f' % (self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3]))
		# logger.info('  [TEST] QWK: %.3f,  PRS: %.3f, SPR: %.3f, RMSE: %.3f' % (self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3]))

	def predict_final_score(self,model):
		for i in range(len(self.tr_modelname)):
			trait_weight = str(self.tr_model_dir+'/Prompt-'+str(self.prompt_id)+'/'+'model_'+str(i+1)+'.hdf5')
			model.load_weights(trait_weight)

			dev_pred1=np.array(model.predict(self.dev_x, batch_size=32)).squeeze()
			test_pred1=np.array(model.predict(self.test_x, batch_size=32)).squeeze()

			# dev_pred1 = np.transpose(dev_pred1)
			# test_pred1 = np.transpose(test_pred1)

			dev_pred_int = rescale_tointscore_for_attr(dev_pred1,1,self.dev_norm_shape, self.prompt_id, self.overall_score_column, self.attr_score_columns, self.original_score_index)
			test_pred_int = rescale_tointscore_for_attr(test_pred1,1,self.test_norm_shape, self.prompt_id, self.overall_score_column, self.attr_score_columns, self.original_score_index)

			dev_tr_pred = np.delete(dev_pred_int,0,axis=0)
			test_tr_pred = np.delete(test_pred_int,0,axis=0)

			dev_trait = kappa_for_traits(self.dev_tr_org, dev_tr_pred,self.prompt_id, 'quadratic')
			test_trait = kappa_for_traits(self.test_tr_org, test_tr_pred,self.prompt_id, 'quadratic')

			print ("For trait:", i),
			print ("DEV Trait: ", dev_trait[i])
			print ("TEST Trait: ", test_trait[i])
			




