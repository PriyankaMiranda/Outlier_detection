import feature_extractor 
import time
import multiprocessing 
import sys 
import os

def multiprocessing_pipeline(location):
	starttime = time.time()
	processes = []
	print("Reading Images.......")
	classes=next(os.walk(location))[1]
	print("Number of classes: "+str(len(classes)))
	for dirx in os.listdir(location):
		fx=feature_extractor.feature_extractorx(20)
		p = multiprocessing.Process(target=fx.extractor, args=(str(folder_loc),dirx))
		processes.append(p)
		p.start()
	for process in processes:
		process.join()
		print('Time taken is {} seconds'.format(time.time() - starttime))


if __name__=="__main__":
	folder_loc = sys.argv[1]

	multiprocessing_pipeline(str(folder_loc))