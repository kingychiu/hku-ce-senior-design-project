# numpy
import numpy as np
# keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# spark, elephas
from keras.optimizers import SGD, Adam
# other
from file_io import FileIO
import datetime


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
                vector = char_str.split(',')[:50]
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
x_train = x_train.reshape(x_train.shape[0], x_test.shape[1], x_test.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])
print('# Training Data', x_train.shape, y_train.shape)
print('# Testing Data', x_test.shape, y_test.shape)

# model config
input_shape = (x_test.shape[1], x_test.shape[2])
epoch_step = 10
num_conv_block = 3
model = Sequential()
# Convolution Layer(s)
print(input_shape)
model.add(Convolution1D(2 ** 6, 3,
                        border_mode="same",
                        input_shape=input_shape))
model.add(Activation('relu'))
print(model.output_shape)
model.add(MaxPooling1D(2))
print(model.output_shape)
for i in range(num_conv_block - 1):
    num_filters = 2 ** (7 + i)
    print(num_filters)
    model.add(Convolution1D(num_filters, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    print(model.output_shape)

# Fully Connected Layer
model.add(Flatten())
print(model.output_shape)

model.add(Dense(model.output_shape[1] // 2))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(model.output_shape[1] // 2))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
## END OF MODEL ##
loss = []
acc = []
val_acc = []
start_time = datetime.datetime.now()
for i in range(0, 20):
    history = model.fit(x_train, y_train, 128, epoch_step,
                        verbose=1, validation_data=(x_test, y_test))
    end_time = datetime.datetime.now()
    print(str(end_time - start_time))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    ## SAVE
    loss = loss + history.history['loss']
    acc = acc + history.history['acc']
    val_acc = val_acc + history.history['val_acc']
    print(len(loss))
    lines = []
    lines.append(str(end_time - start_time))
    lines.append(','.join([str(a) for a in loss]))
    lines.append(','.join([str(a) for a in acc]))
    lines.append(','.join([str(a) for a in val_acc]))
    FileIO.write_lines_to_file('./gpu_one_hot_cnn_' + str(num_conv_block) + '_convB.log', lines)
    model.save(
        './models/gpu_one_hot_cnn_epoch_' + str((i + 1) * epoch_step) + 'ep_' + str(
            num_conv_block) + '_convB.h5')
