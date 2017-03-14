"""
06-03-2017, Anthony Chiu
This CNN run very slow on local machine. This file is used for ensure the model is correct only.
The actual training will be done on SPARK.
"""
import os

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
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


def get_data():
    with open('./datasets/ag_dataset_10000_each_one_hot.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()
        tensor = []
        labels = []
        for line in lines:
            matrix = []
            labels.append(line.split('|l|')[0])
            char_strs = line.split('|l|')[1].split('|c|')
            for char_str in char_strs:
                vector = char_str.split(',')
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
pool_size = (1, 2)
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

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, nb_epoch=100,
          verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_train, y_train, verbose=0)
print('Train accuracy:', score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])
