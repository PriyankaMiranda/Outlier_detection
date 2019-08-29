import numpy as np 
import cv2
import os
import shutil
import sys
import pandas as pd
import time
from functools import reduce
#_____________________Python files required_________________________________________
import display_module 
import feature_matcher
import outlier_detection
# import clustering
import classification 
# import vector_quantization
#___________________________________________________________________________________

class feature_extractorx:
	def __init__(self,vector_size):
		self.vector_size=vector_size
		# how many feature keypoints you would like to extract	
		self.selected_features=[0,0,0,1,0,0,0,0]
		#Edit this list on the basis of the features you want to extract
		self.features=[self.harris_corner,self.shi_tomasi,self.sift,self.surf,self.orb,self.brisk,self.brief,self.fast]
		#Feature list. These are functions in the list not strings. Specify the functions accordingly.
		self.feat_db=list()
		self.classes=list()
		self.feat_db_reduce=list()
		self.check_methods=[os.path.isdir,os.path.isfile]
		self.final_df=pd.DataFrame()
		#Final dataframe
		self.final_reduced_df=pd.DataFrame()
		#Final reduced dataframe
		self.Y_df=pd.DataFrame()
		#Final reduced dataframe
	def test_img_extractor(self,test_location):#Use this while testing on a single image 
		print("Test Image Extractor")
		features=self.select_features(test_location)

	def extractor(self,ext_dir,int_dir):#Use this when directories consisting of images for each class is available  
		print("Extracting Images from directory......")
		self.img_operations(ext_dir,int_dir)

	def select_features(self,img_loc):
		# print("Extracting Features......")
		# ____________________________________________________
		# 1. Harris Corner detection 
		# 2. Shi Tomasi Corner detection
		# 3. SIFT
		# 4. SURF
		# 5. ORB
		# 6. BRISK
		# 7. BRIEF
		# 8. FAST
		# _____________________________________________________
		# result_dest=img_loc.rsplit('/', 1)[0]+"/Results"
		# #Path to destination folder
		# if not os.path.exists(result_dest):
		# 	os.makedirs(result_dest)
		# #Creating directory if not present
		image_desc=[]
		img=cv2.imread(img_loc)
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)		
		gray = np.float32(gray)	
		images=[gray,gray,img,img,img,img,img,img]
		#Paramaeters list.There are objects of the image, not the location. 
		for index , x in enumerate(self.selected_features):
			if(x==1):
				kp,dsc=self.features[index](images[index])
				# feature_dest=result_dest+"/"+img_loc.rsplit('/', 1)[-1].rsplit('.', 1)[0]+"_"+str(features[index]).split(" ")[1]
				#Feature x results destination(Here, we concatenate the name of each function with the name of the original file)
				#Ensure that the names for the feature extraction functions make sense.
				# display_module.display_keypoints(images[index],kp, feature_dest)
				# Use the above line only when you want to view the feature descriptors
				# print("Result destination:"+feature_dest)
				if (dsc is None):
					print("No descriptors extracted! Discarding image.")
					raise
				else:
					dsc = dsc.flatten()
					needed_size = (self.vector_size * 80)
					# Descriptor vector size is set to 100 
					if dsc.size < needed_size:
						dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])	
				image_desc.append(dsc)
		return image_desc

	#_________________________READ FUNCTION__________________________________
	#Ensure that all the classes have separate folders
	#The external folder location is to be passed which includes all the classes
	#File structure:
	#__main_dir__
	#            |__Class1__ 
	#            |          |__img1
	#            |          |__img2
	#            |          |__img3
	#            |          |.....
	#            |__Class2__ ......
	#            |.........
	def img_operations(self,location,dirx):
		try:
			self.check(0,location+"/"+dirx)
		except: 
			return
		self.classes.append(dirx)
		print(location+"/"+dirx)
		tot_feat_vector=list()
		imgs=list()
		experiment_start = time.time()
		#Required to map the images to the features extracted
		for file in os.listdir(location+"/"+dirx):
			# print(file)
			try:
				self.check(1,location+"/"+dirx+"/"+file)
				n_feat=self.select_features(location+"/"+dirx+"/"+file)
			except:
				continue
			tot_feat_vector.append(n_feat)
			imgs.append(file)
		experiment_end = time.time()
		elapsed_time = self.humanize_time(experiment_end - experiment_start)
		print("Feature extraction:"+":"+str(dirx)+":"+(elapsed_time))
		experiment_start = time.time()
		try:
			print("In-class feature matching")
			tot_feat_vector_update=tot_feat_vector.copy()
			#Feature matrix generation for n features 
			fx=feature_matcher.feature_matcherx()
			class_feat_matrix=fx.match(dirx,tot_feat_vector)
		except:
			self.method_exception("In-class feature matching")
		experiment_end = time.time()
		elapsed_time = self.humanize_time(experiment_end - experiment_start)
		print("In-class feature matching:"+":"+str(dirx)+":"+(elapsed_time))
		experiment_start = time.time()

		try:
			print("Outlier detection")
			#convert to readable pandas dataframe format
			df=pd.DataFrame(data=class_feat_matrix)
			#Next we look into outlier detection for the given feature matrix 
			od=outlier_detection.outlier_detectionx()

			outlier_rows=od.isolation_forest(df, imgs)
			# for x in outlier_rows:
			# 	del tot_feat_vector_update[x-1]
		except:
			self.method_exception("Outlier detection")	

		experiment_end = time.time()
		elapsed_time = self.humanize_time(experiment_end - experiment_start)
		print("Outlier detection:"+":"+str(dirx)+":"+(elapsed_time))
		experiment_start = time.time()


		try:
		#Saving image		
			count=0
			for x in outlier_rows:
				# print(imgs[x-count])
				count+=1
				del imgs[x-count]
			if not os.path.exists(location+"_od"+"/"+dirx):
				os.makedirs(location+"_od"+"/"+dirx)
			self.save_imgs(imgs,location,dirx)				
		except: 
			self.method_exception("Saving images")	

		experiment_end = time.time()
		elapsed_time = self.humanize_time(experiment_end - experiment_start)
		print("Saving images:"+":"+str(dirx)+":"+(elapsed_time))

	def humanize_time(self,secs):
	    mins, secs = divmod(secs, 60)
	    hours, mins = divmod(mins, 60)
	    return '%02d:%02d:%02f' % (hours, mins, secs)
	#_________________________SAVING IMAGES__________________________________
	def save_imgs(self, img_list, path,dirx):
		print("Saving images")
		for root, dirs, files in os.walk(path+"/"+dirx):
			for file in files:
				try:
					if str(file) in img_list:
						shutil.copy(path+"/"+dirx+"/"+file,path+"_od/"+dirx+"/"+file)
				except:
					pass

	#_________________________HARRIS CORNER DETECTION__________________________________
	def harris_corner(self,gray):
		try:
			# print("Harris corner detection")
			kp = cv2.cornerHarris(gray,2,3,0.04)
			print(len(kp))
			return kp
		except:
			error_instructions("Harris Corner")
	#_________________________SHI TOMASI FEATURES__________________________________
	def shi_tomasi(self,gray):
		try:
			# print("Shi Tomsai corner detection")
			kp=cv2.goodFeaturesToTrack(gray,25,0.01,10)
			kp=np.int0(kp)
			return kp
		except:
			error_instructions("Shi Tomsai")
	#_________________________SIFT AND SURF MODULE__________________________________
	# def sift_surf(self,img,selected): Do this when you decide on optimizing the code
	# 	try:
	# 		# print("SIFT features")
	# 		sift=cv2.xfeatures2d.sift_or_surf[1][x]()
	# 		kp=sift_or_surf[2].detect(img, None)
	# 		kp=sorted(kp, key = lambda x:-x.response)[:self.vector_size]
	# 		# print("Number of keypoints considered: "+str(len(kp)))
	# 		kp,dsc=sift_or_surf[2].compute(img, kp)
	# 		return kp,dsc     #returning the feature descriptors
	# 	except:
	# 		self.error_instructions("")



	#_________________________SIFT FEATURES__________________________________
	def sift(self,img):
		try:
			# print("SIFT features")
			sift=cv2.xfeatures2d.SIFT_create()
			kp=sift.detect(img, None)
			kp=sorted(kp, key = lambda x:-x.response)[:self.vector_size]
			# print("Number of keypoints considered: "+str(len(kp)))
			kp,dsc=sift.compute(img, kp)
			return kp,dsc     #returning the feature descriptors
		except:
			self.error_instructions("SIFT")
	#_________________________SURF FEATURES__________________________________
	def surf(self,img):
		try:
			# print("Surf features")
			surf=cv2.xfeatures2d.SURF_create()
			kp=surf.detect(img, None)
			kp=sorted(kp, key = lambda x:-x.response)[:self.vector_size]
			# print("Number of keypoints considered: "+str(len(kp)))
			kp,dsc=surf.compute(img, kp)
			return kp,dsc
		except:
			self.error_instructions("SURF")
	#_________________________ORB FEATURES__________________________________
	def orb(self,img):
		try:
			# print("Orb features")
			orb = cv2.ORB_create()
			kp = orb.detect(img,None)
			kp=sorted(kp, key = lambda x:-x.response)[:self.vector_size]
			kp, desc = orb.compute(img, kp)
			return kp,desc
		except:
			self.error_instructions("ORB")
	#_________________________BRISK FEATURES__________________________________
	def brisk(self,img):
		try:
			# print("Brisk features")
			brisk=cv2.BRISK_create()
			kp = brisk.detect(img,None)
			kp=sorted(kp, key = lambda x:-x.response)[:self.vector_size]
			kp, desc = brisk.compute(img, kp)			
			return kp,desc
		except:
			error_instructions("BRISK")
	#_________________________BRIEF FEATURES__________________________________
	def brief(self,img):
		try:
			star=cv2.xfeatures2d.StarDetector_create()
			brief=cv2.xfeatures2d.BriefDescriptorExtractor_create()
			kp=star.detect(img,None)
			kp=sorted(kp, key = lambda x:-x.response)[:self.vector_size]
			kp,desc=brief.compute(img,kp)
			return kp,desc
		except:
			self.error_instructions("BRIEF")
	#_________________________FAST FEATURES__________________________________
	def fast(self,img):
		try:
			# print("Fast features")
			fast=cv2.FastFeatureDetector_create()
			kp = fast.detect(img,None)
			kp=sorted(kp, key = lambda x:-x.response)[:self.vector_size]
			kp,desc=fast.compute(img,kp)
			return kp,desc
		except:
			self.error_instructions("FAST")

	def error_instructions(self,methodx):
		print("Try- pip3 install --user opencv-python==3.4.2.16")
		print("And- pip3 install opencv-contrib-python==3.4.2.16")
		print("Error during "+str(methodx)+" feature extraction")	

	def method_exception(self, cause):
		print("Error during "+str(cause)+".")

	def check(self,choice,loc):
		type= ['Directory','File']
		if not(self.check_methods[choice](loc)):
			raise ValueError(type[choice]+" "+loc+" does not exist.")
# if __name__=="__main__":
# 	folder_loc = sys.argv[1]
# 	extract_x=feature_extractor(20)# define the number of keypoints detected 
# 	extract_x.extractor(str(folder_loc))
# 	# x.test_img_extractor("/home/sumedh/Documents/Priyanka/image_classification/elephant.jpeg")