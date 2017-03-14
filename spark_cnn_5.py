"""
07-03-2017, Anthony Chiu
"""

# numpy
import numpy as np
# keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# spark, elephas
from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from elephas import optimizers as elephas_optimizers
from pyspark import SparkContext, SparkConf
from keras.optimizers import SGD, Adam
# other
from file_io import FileIO

def get_data():
    with open('./datasets/ag_dataset_10000_each_one_hot.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()
        tensor = []
        labels = []
        print(len(lines))

        for line in lines:
            matrix = []
            labels.append(line.split('|l|')[0])
            char_strs = line.split('|l|')[1].split('|c|')
            for char_str in char_strs:
                vector = char_str.split(',')[:40]
                matrix.append(vector)
            tensor.append(matrix)

        x = np.asarray(tensor)
        classes = sorted(list(set(labels)))
        y = np.asarray([classes.index(item) for item in labels])
        print('Labels', classes)

        # shuffle
        x, y = shuffle(x, y, random_state=0)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        f.close()
        return x_train, x_test, y_train, y_test, len(classes)


x_train, x_test, y_train, y_test, num_classes = get_data()

# Convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
# Reshape
x_train = x_train.reshape(x_train.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print('# Training Data', x_train.shape, y_train.shape)
print('# Testing Data', x_test.shape, y_test.shape)

# model config
model = Sequential()
input_shape = (x_test.shape[1], x_test.shape[2], x_test.shape[3])
# Convolution Layer(s)
model.add(Convolution2D(8, 3, 3,
                        border_mode="same",
                        input_shape=input_shape))
model.add(Convolution2D(8, 3, 3,
                        border_mode="same"))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)

model.add(Convolution2D(16, 3, 3,
                        border_mode="same"))
model.add(Convolution2D(16, 3, 3,
                        border_mode="same"))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2)))
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
conf = SparkConf().setAppName('CNN_5') \
    .setMaster('spark://cep16001s1:7077') \
    .set('spark.eventLog.enabled', True) \
    .set('spark.rpc.message.maxSize', 1000)
sc = SparkContext(conf=conf)
# Build RDD from numpy features and labels
rdd = to_simple_rdd(sc, x_train, y_train)
# Epoch Before Check Point
num_epoch_in_one_step = 10
batch_size = 128
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
    FileIO.write_lines_to_file('./cnn_5.log', stat_lines)
    if (i + 1) % 10 == 0 and i != 0:
        model.save('./models/cnn_5_' + str((i + 1) * 10) + 'ep.h5')
# sc.stop()
## END OF SPARK ##
