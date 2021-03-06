import os, json, sys, pickle, math
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor


with open('train_pic_update.pickle', 'rb') as handle:
	train_words = pickle.load(handle)

with open('test_pic_update.pickle', 'rb') as handle:
	test_words = pickle.load(handle)

with open('concreteness_ratings.pickle', 'rb') as handle:
	concreteness_rating = pickle.load(handle)

train_x = list()
train_y = list()

test_x = list()
test_y = list()
count  = 0
for key in train_words.keys():
	vec = train_words[key]
	#if not np.isnan(vec).any():
	#print key
	#print vec
	#count = count + 1
	train_x.append(vec)
	train_y.append(concreteness_rating[key])

for key in test_words.keys():
	vec = test_words[key]
	#if not np.isnan(vec).any():
	test_x.append(test_words[key])
	test_y.append(concreteness_rating[key])

print count
print(len(train_x))
print(len(train_x[0]))
print(len(test_x))
print(len(test_x[0]))

train_y = np.array(train_y).astype(np.float)
test_y = np.array(test_y).astype(np.float)

scaler = StandardScaler()
scaler.fit(train_x)
train_x_s = scaler.transform(train_x)

scaler2 = StandardScaler()
scaler2.fit(test_x)
test_x_s = scaler2.transform(test_x)

reg = linear_model.SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, shuffle=True,n_iter=1000)
reg2 = AdaBoostRegressor(reg, n_estimators=100, loss='square')
reg2.fit(train_x_s, train_y)

y_pred_test = reg2.predict(test_x_s)

corr = stats.spearmanr(test_y, y_pred_test)
plt.scatter(test_y,y_pred_test, s=10)
plt.title("Only Image Features\n (corr = " +str(round(corr[0],4))+')')
plt.xlabel('Ground Truth Concreteness', fontsize=12)
plt.ylabel('Predicted Concreteness', fontsize=12)
plt.savefig('LinearRegressionBatchN1000StandardAdaboost.png')	
