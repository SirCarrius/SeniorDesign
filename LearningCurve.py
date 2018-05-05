import os, json, sys, pickle, logging
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import neighbors, svm

# tab separated
train_vec_first = open(os.path.join(sys.path[0], "train_vec_first.txt"), 'r')
test_vec_first = open(os.path.join(sys.path[0], "test_vec_first.txt"), 'r')
train_rating_first = open(os.path.join(sys.path[0], "train_rating_first.txt"), 'r')
test_rating_first = open(os.path.join(sys.path[0], "test_rating_first.txt"), 'r')

with open('train_phrases_rating.pickle', 'rb') as handle:
    train_phrases_rating = pickle.load(handle)
with open('train_phrases_vec.pickle', 'rb') as handle:
    train_phrases_vec = pickle.load(handle)

with open('test_phrases_rating.pickle', 'rb') as handle:
    test_phrases_rating = pickle.load(handle)
with open('test_phrases_vec.pickle', 'rb') as handle:
    test_phrases_vec = pickle.load(handle)


train_vec = [(line.rstrip().split("\t")[0].replace('"', ''), line.rstrip().split("\t")[1:len(line.rstrip().split("\t"))]) for line in train_vec_first]
train_vec = [(word, [float(num) for num in vec])for word, vec in train_vec]
test_vec = [(line.rstrip().split("\t")[0].replace('"', ''), line.rstrip().split("\t")[1:len(line.rstrip().split("\t"))]) for line in test_vec_first]
test_vec = [(word, [float(num) for num in vec])for word, vec in test_vec]
train_rating = [(line.rstrip().split("\t")[0].replace('"', ''), float(line.rstrip().split("\t")[1])) for line in train_rating_first]
test_rating = [(line.rstrip().split("\t")[0].replace('"', ''), float(line.rstrip().split("\t")[1])) for line in test_rating_first]

train_rating_first.close()
train_vec_first.close()
test_rating_first.close()
test_vec_first.close()


test_err = list()
percent = [10,20,30,40,50,60,70,80,90,100]

full_train_x = [vec for word, vec in train_vec]
# add the vec for phrases
full_train_x = full_train_x + train_phrases_vec.values()

full_train_y = [rating for word, rating in train_rating]
full_train_y = full_train_y + train_phrases_rating.values()

test_x = [vec for word, vec in test_vec]
test_x = test_x + test_phrases_vec.values()

test_y = [rating for word, rating in test_rating]
test_y = test_y + test_phrases_rating.values()

reg = linear_model.LassoCV()
for i in xrange(1, 11):
	percent = 0.1 * i
	train_x = full_train_x[0:percent*len(full_train_x)]
	train_y = full_train_y[0:percent*len(full_train_y)]
	reg.fit(train_x, train_y)
	# y_pred_train = reg.predict(train_x)
	y_pred_test = reg.predict(test_x)
	test_sq_err = mean_squared_error(test_y, y_pred_test)
	test_err.append(test_sq_err)
print (test_err)

plt.scatter(percent,test_err, s=10)
plt.title("Learning Curve")
plt.xlabel('Percent', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.show()
