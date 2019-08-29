from scipy.sparse import csr_matrix, load_npz
from numpy import genfromtxt
from pyod.models.knn import KNN   
from sklearn.svm import SVC
import numpy as np
import sys
from sklearn.ensemble import IsolationForest
class outlier_detectionx:
	def __init__(self):
		self.my_estimators=100

	def isolation_forest(self,feat_matrix, imgs):

		to_model_columns=feat_matrix.columns
		clf=IsolationForest(n_estimators=self.my_estimators, max_samples='auto', contamination=float(.12), \
		                        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
		clf.fit(feat_matrix[to_model_columns])
		pred = clf.predict(feat_matrix[to_model_columns])
		feat_matrix['anomaly']=pred
		outliers=feat_matrix.loc[feat_matrix['anomaly']==-1]
		outlier_index=list(outliers.index)
		#print(outlier_index)
		#Find the number of anomalies and normal points here points classified -1 are anomalous
		print(feat_matrix['anomaly'].value_counts())
		# print(feat_matrix.columns.get_loc(feat_matrix['anomaly'].eq(-1)))
		# print(feat_matrix.index[feat_matrix['anomaly'] == -1].tolist())
		return feat_matrix.index[feat_matrix['anomaly'] == -1].tolist()



	def some_random_test():
		np.set_printoptions(threshold=sys.maxsize)

		X = load_npz("X.npz").toarray()
		Y = genfromtxt('Y.csv', delimiter=',')

		# train kNN detector
		clf_name = 'KNN'
		clf = KNN()

		# find outliers per class
		# print(Y.shape)
		# print(X[Y == 1.].shape)
		# print(X[Y == 0.].shape)
		# print(X[Y == 7.].shape)

		# collect the outliers in a per class manner 
		classList = [1.0, 0.0, 7.0]
		y_train_pred_total = []
		for clas in classList:
			clf.fit(X[Y == clas])
			y_train_pred_total.append(clf.labels_)

		# -------------------------RESULT---------------------
		# 0:inlier, 1: outlier
		np.array(y_train_pred_total).tofile('outliers.csv', sep=',', format='%10.5f')