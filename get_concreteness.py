import os, pickle, json, sys

concretenss = open(os.path.join(sys.path[0], 'concreteness_ratings.txt'), 'r')
concreteness_dict = dict()

for line in concretenss:
	words = line.split('\t')
	print words[0]
	print words[2]
	concreteness_dict[words[0]]  = words[2]

with open('concreteness_ratings.pickle', 'wb') as handle:
	pickle.dump(concreteness_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)