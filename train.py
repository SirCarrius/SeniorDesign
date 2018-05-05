import os, json, sys, compago, pickle, math
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import neighbors, svm

app = compago.Application()

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

# train_x = {k:v for v, k in train_vec}
# train_y = {k:v for v, k in train_rating}
# full_data = [(train_x[k], train_y[k])  for k in train_x.keys() & train_y.keys()]

@app.command
def LinearRegression():
	train_x = [vec for word, vec in train_vec]
	# add the vec for phrases
	train_x = train_x + train_phrases_vec.values()

	train_y = [rating for word, rating in train_rating]
	train_y = train_y + train_phrases_rating.values()

	test_x = [vec for word, vec in test_vec]
	test_x = test_x + test_phrases_vec.values()

	test_y = [rating for word, rating in test_rating]
	test_y = test_y + test_phrases_rating.values()

	reg = linear_model.LinearRegression()
	reg.fit(train_x, train_y)
	# 1x300
	coef = reg.coef_
	intercept = reg.intercept_
	# 1*300 X 300*645, where 645 is the number of observations
	# y_pred_train = coef.dot(np.asarray(train_x).transpose())
	# y_pred_train = y_pred_train+np.repeat(intercept, train_x_num)

	# y_pred_test = coef.dot(np.asarray(test_x).transpose())
	# y_pred_test = y_pred_test+np.repeat(intercept, test_x_num)
	y_pred_train = reg.predict(train_x)
	y_pred_test = reg.predict(test_x)

	train_sq_err = mean_squared_error(train_y, y_pred_train)
	test_sq_err = mean_squared_error(test_y, y_pred_test)
	corr = stats.spearmanr(test_y, y_pred_test)

	print("train error "+str(train_sq_err) + " test error "+str(test_sq_err))
	print("corr "+str(corr))

	plt.scatter(test_y,y_pred_test, s=10)
	plt.title("Only Word Embedding\n (corr = " +str(round(corr[0],4))+')')
	plt.xlabel('Ground Truth Concreteness', fontsize=12)
	plt.ylabel('Predicted Concreteness', fontsize=12)
	plt.show()

@app.command
def LinearRidge():
	train_x = [vec for word, vec in train_vec]
	# add the vec for phrases
	train_x = train_x + train_phrases_vec.values()
	train_x_num = len(train_x)

	train_y = [rating for word, rating in train_rating]
	train_y = train_y + train_phrases_rating.values()

	test_x = [vec for word, vec in test_vec]
	test_x = test_x + test_phrases_vec.values()

	test_y = [rating for word, rating in test_rating]
	test_y = test_y + test_phrases_rating.values()

	reg = linear_model.RidgeCV()
	reg.fit(train_x, train_y)
	# 1x300
	coef = reg.coef_
	intercept = reg.intercept_
	# 1*300 X 300*645, where 645 is the number of observations
	y_pred_train = reg.predict(train_x)

	y_pred_test = reg.predict(test_x)

	train_sq_err = mean_squared_error(train_y, y_pred_train)
	test_sq_err = mean_squared_error(test_y, y_pred_test)
	corr = stats.spearmanr(test_y, y_pred_test)

	print("train error "+str(train_sq_err) + " test error "+str(test_sq_err))
	print("corr "+str(corr))

	plt.scatter(test_y,y_pred_test, s=10)
	plt.title("Only Word Embedding\n (corr = " +str(round(corr[0],4))+')')
	plt.xlabel('Ground Truth Concreteness', fontsize=12)
	plt.ylabel('Predicted Concreteness', fontsize=12)
	plt.show()

@app.command
def LinearLasso():
	train_x = [vec for word, vec in train_vec]
	# add the vec for phrases
	train_x = train_x + train_phrases_vec.values()

	train_y = [rating for word, rating in train_rating]
	train_y = train_y + train_phrases_rating.values()

	test_x = [vec for word, vec in test_vec]
	test_x = test_x + test_phrases_vec.values()

	test_y = [rating for word, rating in test_rating]
	test_y = test_y + test_phrases_rating.values()

	reg = linear_model.LassoCV()
	reg.fit(train_x, train_y)
	# 1x300
	coef = reg.coef_
	intercept = reg.intercept_
	# 1*300 X 300*645, where 645 is the number of observations
	y_pred_train = reg.predict(train_x)

	y_pred_test = reg.predict(test_x)

	train_sq_err = mean_squared_error(train_y, y_pred_train)
	test_sq_err = mean_squared_error(test_y, y_pred_test)
	corr = stats.spearmanr(test_y, y_pred_test)

	print("train error "+str(train_sq_err) + " test error "+str(test_sq_err))
	print("corr "+str(corr))

	plt.scatter(test_y,y_pred_test, s=10)
	plt.title("Only Word Embedding\n (corr = " +str(round(corr[0],4))+')')
	plt.xlabel('Ground Truth Concreteness', fontsize=12)
	plt.ylabel('Predicted Concreteness', fontsize=12)
	plt.show()
	

@app.command
def KNN():
	train_x = [vec for word, vec in train_vec]
	# add the vec for phrases
	train_x = train_x + train_phrases_vec.values()

	train_y = [rating for word, rating in train_rating]
	train_y = train_y + train_phrases_rating.values()

	test_x = [vec for word, vec in test_vec]
	test_x = test_x + test_phrases_vec.values()

	test_y = [rating for word, rating in test_rating]
	test_y = test_y + test_phrases_rating.values()

	reg = neighbors.KNeighborsRegressor(5, "uniform")
	reg.fit(train_x, train_y)
	# 1x300
	# coef = reg.coef_
	# intercept = reg.intercept_
	# 1*300 X 300*645, where 645 is the number of observations
	# y_pred_train = reg.predict(train_x)

	y_pred_test = reg.predict(test_x)

	# train_sq_err = mean_squared_error(train_y, y_pred_train)
	test_sq_err = mean_squared_error(test_y, y_pred_test)
	corr = stats.spearmanr(test_y, y_pred_test)

	print(" test error "+str(test_sq_err))
	print("corr "+str(corr))

	plt.scatter(test_y,y_pred_test, s=10)
	plt.title("Only Word Embedding\n (corr = " +str(round(corr[0],4))+')')
	plt.xlabel('Ground Truth Concreteness', fontsize=12)
	plt.ylabel('Predicted Concreteness', fontsize=12)
	plt.show()

@app.command
def SVR():
	train_x = [vec for word, vec in train_vec]
	# add the vec for phrases
	train_x = train_x + train_phrases_vec.values()

	train_y = [rating for word, rating in train_rating]
	train_y = train_y + train_phrases_rating.values()

	test_x = [vec for word, vec in test_vec]
	test_x = test_x + test_phrases_vec.values()

	test_y = [rating for word, rating in test_rating]
	test_y = test_y + test_phrases_rating.values()

	reg = svm.SVR()
	reg.fit(train_x, train_y)
	# 1x300
	# coef = reg.coef_
	# intercept = reg.intercept_
	# 1*300 X 300*645, where 645 is the number of observations
	# y_pred_train = reg.predict(train_x)

	y_pred_test = reg.predict(test_x)

	# train_sq_err = mean_squared_error(train_y, y_pred_train)
	test_sq_err = mean_squared_error(test_y, y_pred_test)
	corr = stats.spearmanr(test_y, y_pred_test)

	print(" test error "+str(test_sq_err))
	print("corr "+str(corr))

	plt.scatter(test_y,y_pred_test, s=10)
	plt.title("Only Word Embedding\n (corr = " +str(round(corr[0],4))+')')
	plt.xlabel('Ground Truth Concreteness', fontsize=12)
	plt.ylabel('Predicted Concreteness', fontsize=12)
	plt.show()

@app.command
def LinearLassoLearningCurve():
	test_err = list()
	perc = [10,20,30,40,50,60,70,80,90,100]

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
		print(percent)
		print(math.ceil(percent*len(full_train_x)))
		train_x = full_train_x[0:int(math.ceil(percent*len(full_train_x)))]
		train_y = full_train_y[0:int(math.ceil(percent*len(full_train_y)))]
		reg.fit(train_x, train_y)
		# y_pred_train = reg.predict(train_x)
		y_pred_test = reg.predict(test_x)
		test_sq_err = mean_squared_error(test_y, y_pred_test)
		test_err.append(test_sq_err)
	print (test_err)

	plt.plot(perc,test_err)
	plt.title("Learning Curve")
	plt.xlabel('Percent', fontsize=12)
	plt.ylabel('Error', fontsize=12)
	plt.show()

if __name__ == "__main__":
    app.run()
