from __future__ import absolute_import
from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from elephas import optimizers as elephas_optimizers
from pyspark import SparkContext, SparkConf
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

all_classes = ['Business', 'Entertainment', 'Sci/Tech', 'Sports', 'World']
class_to_be_predict = 4

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
    y = [1 if item == all_classes[class_to_be_predict] else 0 for item in y]
    # shuffle
    x, y = shuffle(x, y, random_state=0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    # should be shuffled
    print(y_train[0:10])
    return x_train, x_test, y_train, y_test, 2



# Define basic parameters
batch_size = 64
nb_epoch = 100

# Create Spark context
conf = SparkConf().setAppName('MLP').setMaster('spark://cep16001s1:7077').set('spark.eventLog.enabled', True).set('spark.akka.frameSize', 500)
sc = SparkContext(conf=conf)

# Load data
dimension = 100
number_layers = 1
x_train, x_test, y_train, y_test, nb_classes  = getData(dimension)

# Convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Dense(dimension, input_dim=dimension))
model.add(Activation('relu'))
model.add(Dropout(0.2))

for i in range(0,number_layers-1):
  model.add(Dense(dimension))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))


model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer=SGD())

sgd = SGD(lr=0.1)

# Build RDD from numpy features and labels
rdd = to_simple_rdd(sc, x_train, y_train)

# Initialize SparkModel from Keras model and Spark context
adagrad = elephas_optimizers.Adagrad()
spark_model = SparkModel(sc,
                         model,
                         optimizer=adagrad,
                         frequency='epoch',
                         mode='asynchronous',
                         num_workers=7,master_optimizer=sgd)

# Train Spark model
spark_model.train(rdd, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2, validation_split=0.1)
model.save('mlp_'+all_classes[class_to_be_predict].replace('/', '_')+str(dimension)+'n_'+str(number_layers)+'l.h5')  # creates a HDF5 file 'my_model.h5'

print('original data')
score, acc = model.evaluate(x_train, y_train, verbose=0)
print('Train accuracy:', acc)
score, acc = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', acc)

sc.stop()