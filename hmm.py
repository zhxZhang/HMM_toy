#coding: utf-8
import numpy as np
import pandas as pd
import time 
from functools import wraps
from copy import deepcopy
from numba import jit

def TimeCount(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		start = time.clock()
		func(*args, **kwargs)
		end = time.clock()
		print ('funtion {0}() spends {1} s.'.format(func.__name__, (end - start)))
	return wrapper


class TrainError(Exception):
   '''
   训练异常
   '''
	def __init__(self, message):
		Exception.__init__(self)
		print('TrainError:', message)

class Hmm():
	'''
	Input:观测序列obs_seq/隐状态数量hidden_status/观测状态数量obs_status/序列相应行情his_data
	Output:下期预测值/0跌/1震/2涨
	'''
	pi_thre = 0.00001
	hidden_thre = 0.01
	obs_thre = 0.01

	def __init__(self, obs_seq, hidden_status, obs_status, hist_data=None):
		self.hidden_num = hidden_status
		self.obs_num = obs_status

		self.obs_seq = obs_seq
		self.hist_data = hist_data
		self.obs_length = len(obs_seq)

		self.trans_obs = None
		self.trans_hidden = None
		self.pi = None

		self.alpha = None
		self.beta = None
		self.gamma = None
		self.xi = None

		self.predict = []

	
	def ParaInit(self):
      '''初始化'''
		self.trans_obs = np.random.random((self.hidden_num, self.obs_num))
		self.trans_hidden = np.random.random((self.hidden_num, self.hidden_num))
		self.pi = np.random.random((self.hidden_num))

		for i in range(self.hidden_num):
			self.trans_obs[i] = self.trans_obs[i] / np.sum(self.trans_obs, axis=1)[i]
		for i in range(self.hidden_num):
			self.trans_hidden[i] = self.trans_hidden[i]  / np.sum(self.trans_hidden, axis=1)[i]
		self.pi = self.pi / np.sum(self.pi)

		self.gamma = np.zeros((self.obs_length, self.hidden_num))
		self.xi = np.zeros((self.obs_length-1, self.hidden_num, self.hidden_num))

	def Forward(self):
      '''前向算法'''
		self.alpha = np.zeros((self.obs_length, self.hidden_num))

		self.alpha[0] = self.pi * self.trans_obs[:, self.obs_seq[0]]
		for t in range(1, self.obs_length):
			self.alpha[t] = [np.sum(self.alpha[t-1] * self.trans_hidden[:,i]) for i in range(self.hidden_num)] * self.trans_obs[:, self.obs_seq[t]]
		# prob = sum(self.alpha[self.obs_length-1])

	def Backward(self):
      '''后向算法'''
		self.beta = np.zeros((self.obs_length, self.hidden_num))
		self.beta[self.obs_length-1] = 1

		for t in range(self.obs_length-1)[::-1]:
			for i in range(self.hidden_num):
				for j in range(self.hidden_num):
					self.beta[t,i] += self.trans_hidden[i,j] * self.trans_obs[j,self.obs_seq[t+1]] * self.beta[t+1,j]
		# prob = np.sum([self.pi[i] * self.trans_obs[i,self.obs_seq[0]] * self.beta[0,i] for i in range(self.hidden_num)])
	
	def CountXi(self):
      '''克西值更新'''
		for t in range(self.obs_length-1):
			for i in range(self.hidden_num):
				for j in range(self.hidden_num):
					tmp = sum([self.alpha[t,n] * self.trans_hidden[n,m] * self.trans_obs[m, self.obs_seq[t+1]] * self.beta[t+1,m]
								for n in range(self.hidden_num)
								for m in range(self.hidden_num)])
					self.xi[t][i][j] = self.alpha[t,i] * self.trans_hidden[i,j] * self.trans_obs[j, self.obs_seq[t+1]] * self.beta[t+1,j] / tmp
	
	def CountGamma(self):
      '''gamma值更新'''
      
		for t in range(self.obs_length):
			for i in range(self.hidden_num):
				tmp = sum([self.alpha[t,j] * self.beta[t,j] for j in range(self.hidden_num)])
				self.gamma[t][i] = self.alpha[t][i] * self.beta[t][i] / tmp
	
	def BawmWelch(self):
      '''BW迭代寻优算法'''
      
		for i in range(1000):
			self.Forward()
			self.Backward()
			self.CountXi()
			self.CountGamma()
			if self.ParaUpdate(): break
        
	def Predict(self):
      '''外推预测'''
		self.ParaInit()
		self.BawmWelch()
		for i in range(1000):
			try:
				predict =  self.Viterbi()
				return predict
			except TrainError:
				continue

	
	def ParaUpdate(self):
      '''迭代过程'''
		self.pre_pi = deepcopy(self.pi)
		self.pre_trans_hidden = deepcopy(self.trans_hidden)
		self.pre_trans_obs = deepcopy(self.trans_obs)

		self.pi = self.gamma[0]
		for i in range(self.hidden_num):
			for j in range(self.hidden_num):
				self.trans_hidden[i,j] = sum(self.xi[:,i,j][:-2]) / sum(self.gamma[:,i][:-2])

		for j in range(self.hidden_num):
			for k in range(self.obs_num):
				self.trans_obs[j,k] = sum([self.gamma[t][j] for t in range(self.obs_length) if k == self.obs_seq[t]]) / sum(self.gamma[:,j])

		diff_pi = sum(map(abs, self.pi - self.pre_pi))
		diff_hid = sum(sum(map(abs, self.trans_hidden - self.pre_trans_hidden)))
		diff_obs = sum(sum(map(abs, self.trans_obs - self.pre_trans_obs)))

		return True if (diff_pi < Hmm.pi_thre and diff_hid < Hmm.hidden_thre and diff_obs < Hmm.obs_thre) else False

	
	def Viterbi(self):
      '''维特比预测算法'''
		store_prob = np.zeros((self.obs_length, self.hidden_num))
		store_id = np.zeros((self.obs_length, self.hidden_num))
		store_prob[0] = self.pi * self.trans_obs[:, self.obs_seq[0]]  

		for t in range(1, self.obs_length):
			for i in range(1, self.hidden_num):
				tmp = store_prob[t-1] * self.trans_hidden[:,i]
				if t == 1 and i == 1:
					if sum(tmp) < 0.01:
						raise TrainError('sum == 0')
				store_id[t,i] = np.argmax(tmp)
				store_prob[t,i] = np.max(tmp) * self.trans_obs[i, self.obs_seq[t]]
			# print('store_prob',store_prob[t])
			# print('store_id',store_id[t])

		ind = np.argmax(store_prob[self.obs_length-1])

		predict_1step = np.argmax([self.trans_hidden[ind]])
		# print(predict_1step)
		if store_prob[self.obs_length-1, ind] < 0.0000000001:
		# 	print(store_prob)
			self.print()
			raise TrainError('The observe seq Prob is 0! Train again!')


		for i in reversed(range(self.obs_length)):
			self.predict.append(ind)
			ind = int(store_id[i, ind])

		best, worst = self.Confirm()
		# print(type(best),best)
		if predict_1step == best: return 2
		if predict_1step == worst: 
			return 0
		else:
			return predict_1step

	
	def Confirm(self):
      '''确认预测状态在历史上的表示含义'''
		tmp_dict = dict.fromkeys(range(self.hidden_num),0)

		# if self.hist_data == None: return 
		for i in range(self.obs_length):
			tmp_dict[self.predict[i]] += self.hist_data[i]
		result = sorted(tmp_dict.items(), key=lambda x: x[1], reverse=True)
		# print(result[0][0],result[-1][0])
		return result[0][0], result[-1][0]

	def print(self):
		print('self.trans_hidden = ',self.trans_hidden)
		print('self.trans_obs = ',self.trans_obs)
		print('self.pi = ',self.pi)


def test():
	data = [0,1,2,1,1,1,2,0,1,2,1,2,0]
	data_ = [0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.2,0.1,0.3]
	tes = Hmm(data, 3, 3,data_)
	print(tes.Predict())

if __name__ == '__main__':
	test()