import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.linear_model import Perceptron2

def run():
	print 'intro executed2'
	df = pd.read_csv('./data/iris.csv', header=None)
	y = df.iloc[0:100, 4].values
	y = np.where(y == 'Iris-setosa', -1, 1)
	X = df.iloc[0:100, [0,2]].values
	ppn = Perceptron(eta=0.01, iter=10)
	ppn.fit(X,y)
	plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Missclassifications#')
	plt.show()

class Perceptron(object):
	"""docstring for Perceptron"""
	def __init__(self, eta=0.01, iter=10):
		super(Perceptron, self).__init__()
		self.eta = eta
		self.n_iter = iter

	def fit(self, X, y):
		self.w_ = np.zeros(1 + X.shape[1])
		self.errors_ = []
		for _ in range(self.n_iter):
			errors = 0
			for xi, target in zip(X, y):
				update = self.eta * (target - self.predict(xi))
				self.w_[1:] += update * xi
				self.w_[0] += update
				errors += int(update != 0.0)
			self.errors_.append(errors)
		return self

	def net_input(self, X):
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def predict(self, X):
		ret = np.where(self.net_input(X) >= 0.0, 1, -1)
		return ret




		