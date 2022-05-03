# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2017-02-10 14:56:57
# @Last Modified by:   rokeer
# @Last Modified time: 2018-11-11 16:26:13
from utils import rescale_tointscore, get_logger, rescale_tointscore_for_attr
from metrics_STL import *
import numpy as np
import sys
from keras.models import load_model

logger = get_logger("Evaluate stats")


class Evaluator():

	def __init__(self, prompt_id,score_index,overall_score_column,out_dir, new_out_dir, modelname,
				train_x, dev_x, test_x, train_feats, dev_feats, test_feats,
				train_read_feat, dev_read_feat, test_read_feat,
				train_sentenceFeatures, dev_sentenceFeatures, test_sentenceFeatures,
				train_essayFeatures, dev_essayFeatures, test_essayFeatures,
				train_norm_score, test_norm_score, dev_norm_score, char_only=False):
		# self.dataset = dataset
		self.char_only = char_only
		self.prompt_id = prompt_id
		self.score_index = score_index
		self.overall_score_column = overall_score_column
		self.train_x, self.dev_x, self.test_x = train_x, dev_x, test_x

		#Hand Crafted Features
		self.train_feats, self.dev_feats, self.test_feats = train_feats, dev_feats, test_feats
		self.train_read_feat, self.dev_read_feat, self.test_read_feat = train_read_feat, dev_read_feat, test_read_feat
		self.train_sentenceFeatures, self.dev_sentenceFeatures, self.test_sentenceFeatures = train_sentenceFeatures, dev_sentenceFeatures, test_sentenceFeatures
		self.train_essayFeatures, self.dev_essayFeatures, self.test_essayFeatures = train_essayFeatures, dev_essayFeatures, test_essayFeatures


		# self.train_y_org = rescale_tointscore(train_y, self.prompt_id)
		# self.dev_y_org = rescale_tointscore(dev_y, self.prompt_id)
		# self.test_y_org = rescale_tointscore(test_y, self.prompt_id)

		self.train_y_org = rescale_tointscore(train_norm_score, self.prompt_id, self.score_index, self.overall_score_column)
		self.dev_y_org = rescale_tointscore(dev_norm_score, self.prompt_id, self.score_index, self.overall_score_column)
		self.test_y_org = rescale_tointscore(test_norm_score, self.prompt_id, self.score_index, self.overall_score_column)


		self.out_dir = out_dir
		self.new_out_dir = new_out_dir
		self.modelname = modelname
		self.best_dev = [-1]
		self.best_test = [-1]
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
		#for final score
		#self.train_qwk = kappa(self.train_y_org, train_pred,weight)
		self.dev_qwk = kappa(self.dev_y_org, dev_pred,weight)
		#self.test_qwk = kappa(self.test_y_org, test_pred, weight)


	def calc_rmse(self, train_pred, dev_pred, test_pred):
		self.train_rmse = root_mean_square_error(self.train_y_org, train_pred)
		self.dev_rmse = root_mean_square_error(self.dev_y_org, dev_pred)
		self.test_rmse = root_mean_square_error(self.test_y_org, test_pred)

	def evaluate(self, model, epoch, print_info=False):
		train_pred = np.array(model.predict([self.train_x,self.train_sentenceFeatures,self.train_essayFeatures,self.train_feats,self.train_read_feat], batch_size=32)).squeeze()
		dev_pred = np.array(model.predict([self.dev_x, self.dev_sentenceFeatures,self.dev_essayFeatures,self.dev_feats, self.dev_read_feat], batch_size=32)).squeeze()
		test_pred = np.array(model.predict([self.test_x, self.test_sentenceFeatures,self.test_essayFeatures,self.test_feats, self.test_read_feat], batch_size=32)).squeeze()
			


		# self.dump_predictions(dev_pred1, test_pred1, epoch)

		train_pred_int = rescale_tointscore(train_pred, self.prompt_id, self.score_index, self.overall_score_column)
		dev_pred_int = rescale_tointscore(dev_pred, self.prompt_id, self.score_index, self.overall_score_column)
		test_pred_int = rescale_tointscore(test_pred, self.prompt_id, self.score_index, self.overall_score_column)
		#print("Test_y_org: ", self.test_y_org)
		#print("Test_pred_int: ", test_pred_int)

		# self.calc_correl(train_pred_int, dev_pred_int, test_pred_int)
		self.calc_kappa(train_pred_int, dev_pred_int, test_pred_int)
		# self.calc_rmse(train_pred_int, dev_pred_int, test_pred_int)

		#if self.score_index != self.overall_score_column:
		if self.dev_qwk > self.best_dev[0]:
			self.best_dev = [self.dev_qwk]
			self.best_test = [kappa(self.test_y_org, test_pred_int, 'quadratic')]
			self.best_dev_epoch = epoch
			#model.save_weights(self.out_dir + '/Prompt-' + str(self.prompt_id) + '/' + 'model_'+str(self.score_index)+'.hdf5', overwrite=True)
			#with open(self.out_dir + '/' + 'trait_'+str(self.score_index)+'.txt', 'w') as file:
				#for x in test_pred_int:
					#file.write(str(x) + "\n")
		# else:
		#     if self.dev_qwk > self.best_dev[0]:
		#         self.best_dev = [self.dev_qwk]
		#         self.best_test = [self.test_qwk]
		#         self.best_dev_epoch = epoch
		#         # model.save_weights(self.out_dir + '/' + self.modelname, overwrite=True)
		#         #with open(self.out_dir + '/' + 'output_test_score.txt', 'w') as file:
		#             #for x in test_pred_int:
		#                 #file.write(str(x) + "\n")


		
		if print_info:
			self.print_info()


	def print_info(self):
		logger.info('[DEV]   QWK:  %.3f, (Best @ %i: {{%.3f}})' % (
			self.dev_qwk, self.best_dev_epoch,
			self.best_dev[0]))
		logger.info('[TEST] (Best @ %i: {{%.3f}})' % (
			self.best_dev_epoch,
			self.best_test[0]))
		

		logger.info('--------------------------------------------------------------------------------------------------------------------------')
	
	def print_final_info(self):
		logger.info('--------------------------------------------------------------------------------------------------------------------------')
		# logger.info('Missed @ Epoch %i:' % self.best_test_missed_epoch)
		# logger.info('  [TEST] QWK: %.3f' % self.best_test_missed)
		logger.info('Best QWK @ Epoch %i:' % self.best_dev_epoch)
		logger.info('  [DEV]  QWK: %.3f ' % (self.best_dev[0]))
		logger.info('  [TEST] QWK: %.3f ' % (self.best_test[0]))

		logger.info('Best @ Epoch %i:' % self.best_dev_epoch)
		logger.info('  [DEV]  QWK: %.3f' % (self.best_dev[0]))
		logger.info('  [TEST] QWK: %.3f' % (self.best_test[0]))

	
	def predict_final_score(self,model):
		trait_weight = str(self.out_dir+'/Prompt-'+str(self.prompt_id)+'/'+'model_'+str(self.score_index)+'.hdf5')
		model.load_weights(trait_weight)

		dev_pred1=np.array(model.predict(self.dev_x, batch_size=32)).squeeze()
		test_pred1=np.array(model.predict(self.test_x, batch_size=32)).squeeze()

		dev_pred_int = rescale_tointscore(dev_pred1, self.prompt_id)
		test_pred_int = rescale_tointscore(test_pred1, self.prompt_id)

		dev_trait = kappa(self.dev_y_org, dev_pred_int, 'quadratic')
		test_trait = kappa(self.test_y_org, test_pred_int, 'quadratic')

		print ("For trait:", self.score_index),
		print ("DEV Trait: ", round(dev_trait,3))
		print ("TEST Trait: ", round(test_trait,3))
			

			




