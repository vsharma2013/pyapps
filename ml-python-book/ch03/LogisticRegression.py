from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run():
	iris = datasets.load_iris()
	X = iris.data[:,[2,3]]
	y = iris.target
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
	
	sc = StandardScaler();
	sc.fit(X_train)
	
	X_train_std = sc.transform(X_train)
	X_test_std  = sc.transform(X_test)

	lr = LogisticRegression(C=1000.0, random_state=0)
	lr.fit(X_train_std, y_train)

	y_pred = lr.predict(X_test_std)

	print ('Missclassified samples = %d' % (y_test != y_pred).sum())
	print('Accuracy = %.2f' % accuracy_score(y_test, y_pred))

	print(y_pred)