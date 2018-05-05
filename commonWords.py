import json, os, sys

# each line split by space
# lexvec = open(os.path.join(sys.path[0], "lexvec"), 'r')
# each line split by tab
train_data_first = open(os.path.join(sys.path[0], "train_vec_first.txt"), 'r')
test_data_first = open(os.path.join(sys.path[0], "test_vec_first.txt"), 'r')
full_train_rating = open(os.path.join(sys.path[0], "train_rating.txt"), 'r')
full_test_rating = open(os.path.join(sys.path[0], "test_rating.txt"), 'r')
train_rating_first = open(os.path.join(sys.path[0], "train_rating_first.txt"), 'r')
test_rating_first = open(os.path.join(sys.path[0], "test_rating_first.txt"), 'r')

# lexvecset = [line.rstrip() for line in lexvec]
full_train = [line.rstrip().split("\t")[0].replace('"', '') for line in full_train_rating]
train_first = [line.rstrip().split("\t")[0].replace('"', '') for line in train_rating_first]
train_phrases = list(set(full_train) - set(train_first))

full_test = [line.rstrip().split("\t")[0].replace('"', '') for line in full_test_rating]
test_first = [line.rstrip().split("\t")[0].replace('"', '') for line in test_rating_first]
test_phrases = list(set(full_test) - set(test_first))

print(len(train_phrases))
print(len(test_phrases))

# get rid of the single words that don't exist in lexvec
train_phrases = [ word for word in train_phrases if len(word.split(" ")) > 1]
test_phrases = [ word for word in test_phrases if len(word.split(" ")) > 1]

print(len(train_phrases))
print(len(test_phrases))


# lexvec.close()
train_data_first.close()
full_train_rating.close()
full_test_rating.close()
train_rating_first.close()
test_rating_first.close()

with open('train_phrases.txt', 'w') as fp:
    json.dump(train_phrases, fp)

with open('test_phrases.txt', 'w') as fp:
    json.dump(test_phrases, fp)

# idea 1: similarity
# idea 2: combine words to see if they exist
# for trainline in trainset:
# 	word = trainline.rstrip().split("\t")[0].rstrip().lstrip().lower().replace('"', '')
# 	if " " in word:
# 		phrases = word.split(" ")
# 		count = 0
# 		tempdict = dict()
# 		for p in phrases:
# 			for line in lexvecset:
# 				splitline = line.rstrip().split(" ")
# 				w = splitline[0].rstrip().lstrip().lower()
# 				vec = [float(i) for i in splitline[1:len(splitline)]]
# 				if p == w:
# 					count += 1
# 					tempdict[p] = vec
# 					# all parts are found in the dataset
# 					if count == len(phrases):
# 						# print(tempdict.values())
# 						avg_vec = [sum(x)/float(count) for x in zip(*tempdict.values())]
# 						trainvec[word] = avg_vec
# 						break
# 			#print ("if case "+word + " "+ avg_vec)
# 	else:
# 		for line in lexvecset:
# 			splitline = line.rstrip().split(" ")
# 			w = splitline[0].rstrip().lstrip().lower()
# 			vec = [float(i) for i in splitline[1:len(splitline)]]
# 			#print("word "+w + " " + "vec "+vec)
# 			if word == w:
# 				trainvec[word] = vec
# 				break

# lexvec.close()
# train.close()
# with open(sys.argv[1]+'.json', 'w') as fp:
#     json.dump(trainvec, fp)