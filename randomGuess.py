import json, pickle, os, math, random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error

with open('test_pic.pickle', 'rb') as handle:
	test_picture = pickle.load(handle)

with open('concreteness_ratings.pickle', 'rb') as handle:
	test_concreteness = pickle.load(handle)

true_y = list()
test_y = list()

for item in test_picture.keys():
	print item
	test = round(random.uniform(1, 5.000000000001), 3)
	true_label = test_concreteness[item]
	print test
	print true_label
	true_y.append(true_label)
	test_y.append(test)

print(len(true_y))
print(len(test_y))
#test_sq_err = mean_squared_error(true_y, test_y)
corr = stats.spearmanr(true_y, test_y)

#print(" test error "+str(test_sq_err))
print("corr "+str(corr))

plt.scatter(true_y, test_y, s=10)
plt.title("Random Guessing Images\n (corr = " +str(round(corr[0],4))+')')
plt.xlabel('Ground Truth Concreteness', fontsize=12)
plt.ylabel('Predicted Concreteness', fontsize=12)
#fig.savefig('randomGuess.png')
#plt.close(fig)
plt.savefig("randomGuess.png")
