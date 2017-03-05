# keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

# classes
from preprocess import PreProcess

p = PreProcess('./datasets/ag_dataset.txt')
x_train, x_test, y_train, y_test, num_classes = p.run()

# Convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
# Reshape
dimension = x_train.shape[1]
x_train = x_train.reshape(x_train.shape[0], 1, dimension, 1)
x_test = x_test.reshape(x_test.shape[0], 1, dimension, 1)
print('# Training Data', x_train.shape, y_train.shape)
print('# Testing Data', x_test.shape, y_test.shape)

# model config
pool_size = (1, 2)
model = Sequential()
# CNN
model.add(Convolution2D(32, 3, 1,
                        border_mode="same",
                        # (channel, row, col)
                        input_shape=(1, dimension, 1)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 1, border_mode='same'))
model.add(Activation('relu'))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=pool_size))
print(model.output_shape)

# FC
model.add(Flatten())
print(model.output_shape)

model.add(Dense(model.output_shape[1] // 2))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, nb_epoch=1,
          verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_train, y_train, verbose=0)
print('Train accuracy:', score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])

