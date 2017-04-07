from __future__ import absolute_import
from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def getData(num_features = 100):
  # read data
  with open('../word2vec/ag_word2vec_n_'+str(num_features)+'.dataset','r',encoding='utf8') as f:
    print('Read Lines')
    lines = f.readlines()
    # data
    x = np.asarray([l.split('\\C')[1].split(',') for l in lines])
    print('X dimension',x.shape)
    y = np.asarray([l.split('\\C')[0] for l in lines])
    # classes
    classes = sorted(list(set(y)))
    print(classes)
    y = [classes.index(item) for item in y]
    
    # shuffle
    x, y = shuffle(x, y, random_state=0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    # should be shuffled
    print(y_train[0:10])
    return x_train, x_test, y_train, y_test, len(classes)

# # returns a compiled model
# # identical to the previous one
model = load_model('test.h5')
# Load data
dimension = 100


print('Start Reading Data')
x_train, x_test, y_train, y_test, nb_classes  = getData(dimension)
# Convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

score, acc = model.evaluate(x_train, y_train, verbose=0)
print('Train accuracy:', acc)
score, acc = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', acc)


# import operator

# # matrix[actual][prediction]
# matrix = {}
# matrix[0] = {}
# matrix[1] = {}
# matrix[2] = {}
# matrix[3] = {}
# matrix[4] = {}
# print(x_test.shape)
# y_predict = model.predict(x_test)
# y_predict = [max(enumerate(item), key=operator.itemgetter(1))[0] for item in y_predict]
# y_test = [max(enumerate(item), key=operator.itemgetter(1))[0] for item in y_test]

# for i in range(0, len(y_test)):
#   p = y_predict[i]
#   a = y_test[i]
#   if p in (matrix[a]).keys():
#     matrix[a][p] += 1
#   else:
#     matrix[a][p] = 1

# for i in range(0,5):
#   print(matrix[i])
#   acc = matrix[i][i] / (matrix[i][0]+matrix[i][1]+matrix[i][2]+matrix[i][3]+matrix[i][4])
#   print(acc)



