import json
import nltk
import sklearn
import pickle
import numpy as np
from scipy.sparse import csr_matrix, save_npz

fileNameTrain = "newelecTrain.json"
fileNameTest = "newelecTest.json"

# =================== Load data ===================
def loadData(fileName,numOfDataPoints,mode = 1,seed=10):

	np.random.seed(seed)
	with open(fileName) as json_file:
		data = json.load(json_file)

	trainingData = []
	trainingTarget = []

	for key in data :
		dataSelection = np.asarray(data[key])
		tempSize = len(trainingData)
		
		# =================== For training data ===================
		if mode == 1 and int(key) in classList:

			if dataSelection.shape[0] > numOfDataPoints:
				trainingData.extend(np.random.choice(dataSelection, size = numOfDataPoints, replace = False))
			else :
				trainingData.extend(dataSelection)
			trainingTarget.extend([int(key)] * (len(trainingData) - tempSize))


		# =================== For test data ===================
		elif mode == 2 and int(key) in classList:

			numOfDataPoint = numOfDataPoints * 100000
			if dataSelection.shape[0] > numOfDataPoint:
				print(numOfDataPoint)
				trainingData.extend(np.random.choice(dataSelection, size = numOfDataPoint, replace = False))
			else :
				trainingData.extend(dataSelection)
			trainingTarget.extend([int(key)] * (len(trainingData) - tempSize))
	if mode == 1:
		print('=================== Training data reading completed ===================')
	elif mode == 2:
		print('=================== Test data reading completed ===================')

	return trainingData, trainingTarget

# =================== Vectorising data ===================
def documentprep(trainingData,fileName,trainingTarget,mode=1):

	if mode == 1:
		data = []
		row = []
		col = []

		vocabulary = {}
		numDoc = len(trainingTarget)
		
		for documentNumber,dataFile in enumerate(trainingData):
			for indexWord,word in enumerate(nltk.word_tokenize(dataFile)):
				word = word.encode('utf-8').strip()
				index = vocabulary.setdefault(word, len(vocabulary))
				row.append(documentNumber)
				col.append(index)
				data.append(1.0)
		
		count_vect = sklearn.feature_extraction.text.CountVectorizer(stop_words=None,lowercase=False,vocabulary=vocabulary,ngram_range =(1,1))
		csrMatrix = csr_matrix((data, (row,col)),shape=(len(trainingData),len(vocabulary)))

		with open(fileName+".pk", 'wb') as fin:
			pickle.dump(count_vect, fin)

	# ============== Using Sklearn count vect ==============
	else : 
		data = []
		vocabulary = {}
		count_vect = sklearn.feature_extraction.text.CountVectorizer(stop_words=None,lowercase=False,ngram_range =(1,2))
		csrMatrix = count_vect.fit_transform(trainingData)

		with open(fileName+".pk", 'wb') as fin:
			pickle.dump(count_vect, fin)
	print('=================== Vectorisation completed ===================')
	return csrMatrix

# ===============================EXECUTION===========================
classList = [0,1,7]
# classList = [0,1,6,7]
numOfDataPoint = 1000

trainingData, trainingTarget = loadData(fileNameTrain, numOfDataPoint, mode = 1)
testData, testTarget =  loadData(fileNameTrain, numOfDataPoint, mode = 2, seed=10)

fileName = 'electNode'
csrMatrix = documentprep(trainingData, fileName, trainingTarget, mode=1)

# X - csrMatrix
# Y - trainingTarget

np.array(trainingTarget).tofile('Y.csv', sep=',', format='%10.5f')
save_npz("X.npz", csrMatrix)


fileName = 'electNodeTest'
csrMatrix_test = documentprep(testData, fileName, testTarget, mode=1)

# X_test - csrMatrix_test
# Y_test - testTarget

np.array(testTarget).tofile('Y_test.csv', sep=',', format='%10.5f')
save_npz("X_test.npz", csrMatrix_test)
