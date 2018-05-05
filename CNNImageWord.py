import os, json, sys, pickle, math
import numpy as np
from sklearn import linear_model
from sklearn import neighbors
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neural_network import MLPRegressor

with open('train_phrases_vec.pickle', 'rb') as handle:
    train_phrases_vec = pickle.load(handle)

with open('test_phrases_vec.pickle', 'rb') as handle:
    test_phrases_vec = pickle.load(handle)

with open('train_word_vec.pickle', 'rb') as handle:
    train_word_vec = pickle.load(handle)

with open('test_word_vec.pickle', 'rb') as handle:
    test_word_vec = pickle.load(handle)

with open('train_pic_update.pickle', 'rb') as handle:
	train_image = pickle.load(handle)

with open('test_pic_update.pickle', 'rb') as handle:
	test_image = pickle.load(handle)

with open('concreteness_ratings.pickle', 'rb') as handle:
	concreteness_rating  = pickle.load(handle)

train_x = list()
train_y = list()

test_x = list()
test_y = list()

for key in train_word_vec.keys():
	word_vec = train_word_vec[key]
	image_vec = train_image[key]
	final_vec = np.concatenate([np.asarray(word_vec), np.asarray(image_vec)])
	try:
		final_vec = np.array(final_vec).astype(np.float)
		train_x.append(final_vec)
		train_y.append(concreteness_rating[key])
	except ValueError:
		print(key)

for key in train_phrases_vec.keys():
	word_vec = train_phrases_vec[key]
	image_vec = train_image[key]
	final_vec = np.concatenate([np.asarray(word_vec), np.asarray(image_vec)])
	try:
		final_vec = np.array(final_vec).astype(np.float)
		train_x.append(final_vec)
		train_y.append(concreteness_rating[key])
	except ValueError:
		print(key)

train_y = np.array(train_y).astype(np.float)
print "train done"
print len(train_x)
print len(train_y)
print len(train_x[0])
print type(train_x[0])
print type(train_y)

for key in test_word_vec.keys():
	word_vec = test_word_vec[key]
	image_vec = test_image[key]
	final_vec = np.concatenate([np.asarray(word_vec), np.asarray(image_vec)])
	try:
		final_vec = np.array(final_vec).astype(np.float)
		test_x.append(final_vec)
		test_y.append(concreteness_rating[key])
	except ValueError:
		print (key)
for key in test_phrases_vec.keys():
	word_vec = test_phrases_vec[key]
	image_vec = test_image[key]
	final_vec = np.concatenate([np.asarray(word_vec), np.asarray(image_vec)])
	try:		
		final_vec = np.array(final_vec).astype(np.float)
		test_x.append(final_vec)
		test_y.append(concreteness_rating[key])
	except ValueError:
		print(key)
test_y = np.array(test_y).astype(np.float)
print "test done"
print len(test_x)
print len(test_y)
print len(test_x[0])
print type(test_x[0])
print type(test_y[0])

reg = MLPRegressor(hidden_layer_sizes=(200,200,200),activation='logistic', solver='adam', alpha=0.001,learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
reg.fit(train_x, train_y)

y_pred_test = reg.predict(test_x)
corr = stats.spearmanr(test_y, y_pred_test)
plt.scatter(test_y,y_pred_test, s=10)
plt.title("Word Embeddings and Image Features\n (corr = " +str(round(corr[0],4))+')')
plt.xlabel('Ground Truth Concreteness', fontsize=12)
plt.ylabel('Predicted Concreteness', fontsize=12)
plt.savefig('LogisticNNImageWordThree200.png')
