# keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

# spark, elephas
from keras.optimizers import SGD, Adam

# classes
from preprocess import PreProcess
from file_io import FileIO
import datetime

## MODEL ##

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
epoch = 1000
pool_size = (1, 2)
num_conv_block = 3
model = Sequential()
# Convolution Layer(s)
model.add(Convolution2D(2 ** 6, 3, 1,
                        border_mode="same",
                        input_shape=(1, dimension, 1)))
model.add(Activation('relu'))
model.add(Convolution2D(2 ** 6, 3, 1, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
print(model.output_shape)

for i in range(num_conv_block - 1):
    num_filters = 2 ** (7 + i)
    print(num_filters)
    model.add(Convolution2D(num_filters, 3, 1, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(num_filters, 3, 1, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    print(model.output_shape)

# Fully Connected Layer
model.add(Flatten())
print(model.output_shape)

model.add(Dense(model.output_shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(model.output_shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
## END OF MODEL ##

start_time = datetime.datetime.now()
print(start_time)
history = model.fit(x_train, y_train, 128, epoch,
                    verbose=1, validation_data=(x_test, y_test))
end_time = datetime.datetime.now()
print(str(end_time - start_time))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

## SAVE
lines = []
lines.append(str(end_time - start_time))
lines.append(','.join([str(a) for a in history.history['loss']]))
lines.append(','.join([str(a) for a in history.history['acc']]))
lines.append(','.join([str(a) for a in history.history['val_acc']]))
FileIO.write_lines_to_file('./gpu_cnn.log', lines)
model.save('./models/gpu_cnn_epoch_' + str(epoch) + 'ep_' + str(num_conv_block) + '_convB.h5')
