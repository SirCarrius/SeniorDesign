import json, os, sys, pickle
import numpy as np

training_image = open(os.path.join(sys.path[0], "train_word_absolue_paths.tsv"),'r')
testing_image = open(os.path.join(sys.path[0], "val_word_absolute_paths.tsv"), 'r')

traing_dict = dict()
testing_dict = dict()

count = 0
for line in testing_image:
	l = line.rstrip()
	word_path = l.split("\t")
	word = word_path[0]
	path = word_path[1]
	image_dict = dict()
	print ("path "+path)
	for i in xrange(1, 101):
		image_path_jpg = path + "/0" + str(i) + ".jpg.pkl"
		image_path_gif = path + "/0" + str(i) + ".gif.pkl"
		image_path_png = path + "/0" + str(i) + ".png.pkl"
		image_path_jpeg = path + "/0" + str(i) + ".jpeg.pkl"
		try:
			image_arr = pickle.load(open(image_path_jpg, 'r'))
		except IOError:
			try:
				image_arr = pickle.load(open(image_path_gif, 'r'))
			except IOError:
				try:
					image_arr = pickle.load(open(image_path_png, 'r'))
				except IOError:
					try:
						image_arr = pickle.load(open(image_path_jpeg, 'r'))
					except IOError:
						pass
		# check the array is not none
		if image_arr != None and (not np.isnan(image_arr).any()):
			image_dict[i] = image_arr
	print("count " + str(count) + "is done!")
	rowNum = len(image_dict.keys())
	colNum = len(image_dict.values()[0])
	stackedMat =  np.hstack(image_dict.values())
	stackedMat.shape = (rowNum, colNum)
	mean_vec = np.nanmean(stackedMat, axis=0)
	var_vec = np.nanvar(stackedMat, axis=0)
	final_vec = np.hstack([mean_vec, var_vec])
	testing_dict[word] = final_vec
	#print (word)
	#print (final_vec)
	count = count + 1
	if (count % 200 == 0):
		with open('test_pic_update.pickle', 'wb') as handle:
			pickle.dump(testing_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)	

with open('test_pic_update.pickle', 'wb') as handle:
	pickle.dump(testing_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	
	
