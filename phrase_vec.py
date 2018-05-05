import json, os, sys, pickle

# with open('lexvec.pickle', 'rb') as handle:
#     lexvec = pickle.load(handle)
# rid of single words that don't exist in lexvec
# train_phrases = open(os.path.join(sys.path[0], "train_phrases.txt"), 'r')
# test_phrases = open(os.path.join(sys.path[0], "test_phrases.txt"), 'r')
# train_rating = open(os.path.join(sys.path[0], "train_rating.txt"), 'r')
# test_rating = open(os.path.join(sys.path[0], "test_rating.txt"), 'r')

train_words = open(os.path.join(sys.path[0], "test_vec_first.txt"), 'r')

# lexvecdict = dict()
# turn lexvec into a dict - lexvec split by space
# for line in lexvec:
# 	l = line.rstrip()
# 	word_vec = l.split(" ")
# 	w = word_vec[0].lower()
# 	vec = [float(i) for i in word_vec[1:len(word_vec)]]
# 	lexvecdict[w] = vec

# with open('lexvec.pickle', 'wb') as handle:
# 	pickle.dump(lexvecdict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# test_rating_dict = dict()
# # turn rating into a dict
# for line in train_rating:
# 	l = line.rstrip()
# 	word_rating = l.split("\t")
# 	word = word_rating[0].replace('"','')
# 	rating = float(word_rating[1])
# 	# only store phrases
# 	if word.split(" ") > 1:
# 		test_rating_dict[word] = rating



test_embedding = dict()
count = 0
for line in train_words:
	print count
	count = count + 1
	l = line.rstrip().split('\t')
	word = l[0]
	vec = l[1:len(l)]
	word = word.replace('"', '')
	test_embedding[word] = vec
# # test_concreteness = dict()
# for line in train_words:
# 	l = line.rstrip()
# 	l = l.replace("[","").replace("]","").replace('"','')
# 	strlist = l.split(",")
# 	for item in strlist:
# 		w = item.lstrip().rstrip()
# 		###### FOR RATING
# 		rating = test_rating_dict[w]

# 		###### FOR VEC
# 		p = w.split(" ")
# 		count = 0
# 		tempdict = dict()
# 		for i in p:
# 			try:
# 				tempdict[count] = lexvec[i]
# 				count += 1
# 			except:
# 				pass
# 		if count == len(p):
# 			avg_vec = [sum(x)/float(count) for x in zip(*tempdict.values())]
# 			test_embedding[w] = avg_vec
			# test_concreteness[w] = rating
print(len(test_embedding.keys()))
# print(len(test_concreteness.keys()))


# with open('train_phrases_rating.pickle', 'wb') as handle:
#  	pickle.dump(test_concreteness, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('test_word_vec.pickle', 'wb') as handle:
 	pickle.dump(test_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)


train_words.close()
# train_phrases.close()
# test_phrases.close()
# train_rating.close()
# test_rating.close()