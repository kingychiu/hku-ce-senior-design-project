"""
07-03-2017, Anthony Chiu
"""

# keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils

# spark, elephas
from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from elephas import optimizers as elephas_optimizers
from pyspark import SparkContext, SparkConf
from keras.optimizers import SGD, Adam

# classes
from preprocess import PreProcess
from file_io import FileIO

## MODEL ##

p = PreProcess('./datasets/ag_dataset.txt')
x_train, x_test, y_train, y_test, num_classes = p.run()

# Convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
# Reshape
dimension = x_train.shape[1]
x_train = x_train.reshape(x_train.shape[0], dimension, 1)
x_test = x_test.reshape(x_test.shape[0], dimension, 1)
print('# Training Data', x_train.shape, y_train.shape)
print('# Testing Data', x_test.shape, y_test.shape)

# model config
pool_size = (1, 2)
model = Sequential()

# Convolution Layer(s)
model.add(Convolution1D(nb_filter=16,
                        filter_length=3,
                        border_mode="valid",
                        activation='relu',
                        subsample_length=1,
                        input_shape=(dimension, 1)))
print(model.output_shape)
model.add(MaxPooling1D(2))
print(model.output_shape)

model.add(Convolution1D(nb_filter=32,
                        filter_length=3,
                        border_mode="valid",
                        activation='relu',
                        subsample_length=1))
print(model.output_shape)
model.add(MaxPooling1D(2))
print(model.output_shape)

model.add(Convolution1D(nb_filter=64,
                        filter_length=3,
                        border_mode="valid",
                        activation='relu',
                        subsample_length=1))
print(model.output_shape)
model.add(MaxPooling1D(2))
print(model.output_shape)

# Fully Connected Layer
model.add(Flatten())
print(model.output_shape)

model.add(Dense(model.output_shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

## END OF MODEL ##

## SPARK ##
# Create Spark context
conf = SparkConf().setAppName('CNN_6') \
    .setMaster('spark://cep16001s1:7077') \
    .set('spark.eventLog.enabled', True) \
    .set('spark.rpc.message.maxSize', 1000)
sc = SparkContext(conf=conf)
# Build RDD from numpy features and labels
rdd = to_simple_rdd(sc, x_train, y_train)
# Epoch Before Check Point
num_epoch_in_one_step = 10
batch_size = 1000
# Accuracy records
stat_lines = []
adam = elephas_optimizers.Adam()
adagrad = elephas_optimizers.Adagrad()
spark_model = SparkModel(sc, model,
                         mode='asynchronous',
                         frequency='epoch',
                         num_workers=7,
                         optimizer=adagrad,
                         master_optimizer=SGD(),
                         master_loss='categorical_crossentropy',
                         master_metrics=['accuracy'])
for i in range(0, 200):
    # Train Spark model
    # Initialize SparkModel from Keras model and Spark context
    spark_model.train(rdd,
                      nb_epoch=num_epoch_in_one_step,
                      batch_size=batch_size,
                      verbose=2,
                      validation_split=0.1)
    score1 = model.evaluate(x_train, y_train, verbose=0)
    score2 = model.evaluate(x_test, y_test, verbose=0)
    print('#############################')
    print('Finished epochs', (i + 1) * num_epoch_in_one_step)
    print('Train accuracy:', score1[1])
    print('Test accuracy:', score2[1])
    print('#############################')
    stat_lines.append(str((i + 1) * 10) + ', ' + str(score1[1]) + ', ' + str(score2[1]))
    FileIO.write_lines_to_file('./cnn_6.log', stat_lines)
    if (i + 1) % 10 == 0 and i != 0:
        model.save('./models/cnn_6_' + str((i + 1) * 10) + 'ep.h5')
# sc.stop()
## END OF SPARK ##
