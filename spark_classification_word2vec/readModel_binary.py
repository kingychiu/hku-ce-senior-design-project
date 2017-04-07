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

all_classes = ['Business', 'Entertainment', 'Sci/Tech', 'Sports', 'World']

def getData(num_features = 100, class_to_be_predict = 0):
  # read data
  with open('../word2vec/ag_word2vec_n_'+str(num_features)+'.dataset','r',encoding='utf8') as f:
    # print('Read Lines')
    lines = f.readlines()
    # data
    x = np.asarray([l.split('\\C')[1].split(',') for l in lines])
    # print('X dimension',x.shape)
    y = np.asarray([l.split('\\C')[0] for l in lines])
    # classes
    classes = sorted(list(set(y)))
    # print(classes)
    y = [1 if item == all_classes[class_to_be_predict] else 0 for item in y]
    # shuffle
    x, y = shuffle(x, y, random_state=0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    # should be shuffled
    print(y_train[0:20])
    return x_train, x_test, y_train, y_test, 2


# # returns a compiled model
# # identical to the previous one
# Load data
dimension = 100
number_layers = 1

for i in range(0, len(all_classes)):
  print(all_classes[i])
  model = load_model('mlp_'+all_classes[i].replace('/', '_')+str(dimension)+'n_'+str(number_layers)+'l.h5')
  # print('Start Reading Data')
  x_train, x_test, y_train, y_test, nb_classes  = getData(dimension, i)
  # Convert class vectors to binary class matrices
  y_train = np_utils.to_categorical(y_train, nb_classes)
  print(y_train[0:5])
  y_test = np_utils.to_categorical(y_test, nb_classes)

  score, acc = model.evaluate(x_train, y_train, verbose=0)
  print('Train accuracy:', acc)
  score, acc = model.evaluate(x_test, y_test, verbose=0)
  print('Test accuracy:', acc)
