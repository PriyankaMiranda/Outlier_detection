from scipy.sparse import csr_matrix, load_npz
from sklearn.svm import SVC
import numpy as np
import feature_matcher
import random 
import feature_matcher 
class classificationx:
	def __init__(self):
		self.my_estimators=100
		self.fx=feature_matcher.feature_matcherx()
		# self.class_feat_matrix=self.fx.match(dirx,tot_feat_vector)
	def SVM_classify(self,db,classes):
		m_list=list()
		for x in range(len(db)):
			m_list.append(x)
		mapping=dict(zip(m_list, classes))
		self.generate_test(db,mapping)

	def generate_test(self,db,mapping):
		test_class=db[random.randint(0,len(db)-1)]
		test_tuple=test_class[random.randint(0,len(test_class)-1)]
		print(test_tuple)
		print(len(test_tuple))
		input()
		for x in db:
			print("--")
			#distance from the class
		input()
		print("SVM Classification")
		#For each class, make a feature dist matrix from the obj
		













# # ---------------------------original training------------------------
# original_clf = SVC(gamma = 'auto')
# original_clf.fit(X, Y)
# print('Original case accuracy: ', original_clf.score(X, Y))

# # ----------------------------reduced training--------------------
# reduced_clf = SVC(gamma = 'auto')
# # # mask the outliers
# reduced_clf.fit(X[outliers == 0.0], Y[outliers == 0.0])
# print('Outlier case accuracy: ', reduced_clf.score(X[outliers == 0.0], Y[outliers == 0.0]))