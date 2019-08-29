import scipy.spatial
import numpy as np
class feature_matcherx:
	def __init__(self):
		self.distance_measure='cosine'
		self.dist_list=list()
		self.final_feature_vector=list(np.array([]))
		self.dist_matrix=list(np.array([]))

	def dist(self, vector,matrix,start_loc):
		self.dist_list=list()
		# getting distance between the images
		v = vector.reshape(1, -1)
		for x in range(start_loc+1):
			self.dist_list.append(0)
		for x in matrix[start_loc+1:]:
			try:
				# print(len(x))
				# print(len(v[0]))
				# input()
				x=x.reshape(1,-1)		
				temp_dist=scipy.spatial.distance.cdist(x, v, self.distance_measure)
				self.dist_list.append(temp_dist[0][0])
			except:
				self.dist_list.append(1)
		return self.dist_list
	# def match(self, image_path, topn=5):
	# 	features = extract_features(image_path)
	# 	img_distances = self.cos_cdist(features)
	# 	# getting top 5 records
	# 	nearest_ids = np.argsort(img_distances)[:topn].tolist()
	# 	nearest_img_paths = self.names[nearest_ids].tolist()
	# 	return nearest_img_paths, img_distances[nearest_ids].tolist()

	def match(self,directory_name,total_feat_vector):
		print(directory_name)
		for x in range(np.array(total_feat_vector).shape[1]):
			column=np.array(total_feat_vector)[:,x]		
			featn=[]
			for i,elem in enumerate(column):
				featn.append(self.dist(elem,column,i))
			dist_arr=np.asarray(featn)
			i_lower = np.tril_indices(dist_arr.shape[0], -1)
			dist_arr[i_lower] = dist_arr.T[i_lower]
			try:
				prev_arr=np.concatenate((prev_arr, dist_arr), axis=1)
			except:
				prev_arr=dist_arr
		return prev_arr
			# input()
			# print(np.append(self.dist_matrix, list(dist_arr), axis=1))
			# input()

			# self.final_feature_vector.append(distances)

		# for feat_vector in np.asarray(total_feat_vector).T:# for each feature
		# 	print(feat_vector)# all features of a single image is in this array
		# 	input()
		# 	for i,imagex in enumerate(feat_vector):# for each image
		# 		classx_matrix=self.dist(imagex,feat_vector,i)# calculating distance 


		# x=np.asarray(self.final_feature_vector)
