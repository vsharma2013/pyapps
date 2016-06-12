import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run():
	print 'ADALineGD executed'
	df = pd.read_csv('./data/iris.csv', header=None)
	y = df.iloc[0:100, 4].values
	y = np.where(y == 'Iris-setosa', -1, 1)
	X = df.iloc[0:100, [0,2]].values
	X_std = np.copy(X);
	X_std[:, 0] = X[:,0] - X[:, 0].mean() / X[:, 0].std() 
	X_std[:, 1] = X[:,1] - X[:, 1].mean() / X[:, 1].std()
	ada = AdalineGD(eta=0.01, n_iter=15)
	ada.fit(X_std, y)
	print ada.cost_



class AdalineGD(object):
	"""docstring for AdalineGD"""
	def __init__(self, eta=0.01, n_iter=10):
		super(AdalineGD, self).__init__()
		self.eta = eta
		self.n_iter = n_iter

	def fit(self, X, y):
		self.w_ = np.zeros(1 + X.shape[1]);
		self.cost_ = []

		for i in range(self.n_iter):
			output = self.net_input(X)
			errors = (y - output)
			self.w_[1:] = self.eta * X.T.dot(errors)
			self.w_[0] = self.eta * errors.sum()
			cost = (errors ** 2).sum()/2
			self.cost_.append(cost)
		return self
	

	def net_input(self, X):
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def activation(self, X):
		return self.net_input(X)

	def predict(self, X):
		ret = np.where(self.activation(X) >= 0.0, 1, -1)
		return ret