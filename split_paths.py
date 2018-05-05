import json, time, socket
import os, sys, requests


image_path = open(os.path.join(sys.path[0], "train_rating"), 'r')

path_list = list()

count = 0
for line in image_path:
	allpaths = json.loads(line)
	for word, score in allpaths:
		count += 1
		curr_list = [word, path]
		path_list.append(curr_list)
		if count % 100 == 0:
			with open('train_'+str(count/20)+'.txt', 'w') as outfile:
				json.dump(path_list, outfile)
				path_list = list()
