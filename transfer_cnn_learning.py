# numpy
import numpy as np
# keras
from keras.utils import np_utils
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, Adadelta
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# other
from file_io import FileIO
import datetime


### DATA ###
def get_data(path):
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        tensor = []
        labels = []
        for line in lines:
            matrix = []
            labels.append(line.split('|l|')[0])
            char_look_up_list = line.split('|l|')[1].split(',')
            for char_look_up in char_look_up_list:
                look_up_vector = []
                for digit_str in char_look_up:
                    if digit_str == '0' or digit_str == '1':
                        look_up_vector.append(int(digit_str))
                matrix.append(look_up_vector)
            tensor.append(matrix)
        print(tensor[0])
        x = np.array(tensor)
        del tensor
        print(x.shape)
        classes = sorted(list(set(labels)))
        y = np.asarray([classes.index(item) for item in labels])
        print('Labels', classes)
        # shuffle
        x, y = shuffle(x, y, random_state=0)
        ## cut
        x = x[:50000]
        y = y[:50000]
        print(len(x))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        f.close()
        return x_train, x_test, y_train, y_test, len(classes)


x_train = {}
x_test = {}
y_train = {}
y_test = {}
num_classes = {}
x_train['ag1'], x_test['ag1'], y_train['ag1'], y_test['ag1'], num_classes['ag1'] = get_data(
    './datasets/ag_7blkup_10000each.txt')
x_train['ag2'], x_test['ag2'], y_train['ag2'], y_test['ag2'], num_classes['ag2'] = get_data(
    './datasets/ag2_7blkup_10000each.txt')

# Convert class vectors to binary class matrices
y_train['ag1'] = np_utils.to_categorical(y_train['ag1'], num_classes['ag1'])
y_train['ag2'] = np_utils.to_categorical(y_train['ag2'], num_classes['ag2'])
y_test['ag1'] = np_utils.to_categorical(y_test['ag1'], num_classes['ag1'])
y_test['ag2'] = np_utils.to_categorical(y_test['ag2'], num_classes['ag2'])
# Reshape
x_train['ag1'] = x_train['ag1'].reshape(x_train['ag1'].shape[0], x_train['ag1'].shape[1],
                                        x_train['ag1'].shape[2], 1)
x_train['ag2'] = x_train['ag2'].reshape(x_train['ag2'].shape[0], x_train['ag2'].shape[1],
                                        x_train['ag2'].shape[2], 1)
x_test['ag1'] = x_test['ag1'].reshape(x_test['ag1'].shape[0], x_test['ag1'].shape[1],
                                      x_test['ag1'].shape[2], 1)
x_test['ag2'] = x_test['ag2'].reshape(x_test['ag2'].shape[0], x_test['ag2'].shape[1],
                                      x_test['ag2'].shape[2], 1)
### END OF DATA ###

### COMMON CNN LAYERS TEMPLATE###
input_shape = (x_train['ag1'].shape[1], x_train['ag1'].shape[2], x_train['ag1'].shape[3])
epoch_step = 1


def create_init_model(num_classes):
    init_model = Sequential()
    init_model.add(Convolution2D(2 ** 7, 3, 3,
                                 border_mode="same",
                                 input_shape=input_shape))
    init_model.add(Activation('relu'))
    init_model.add(Convolution2D(2 ** 7, 3, 3, border_mode='same'))
    init_model.add(Activation('relu'))
    init_model.add(MaxPooling2D(pool_size=(4, 2)))
    init_model.add(Dropout(0.25))

    init_model.add(Convolution2D(2 ** 8, 3, 3, border_mode='same'))
    init_model.add(Activation('relu'))
    init_model.add(Convolution2D(2 ** 8, 3, 3, border_mode='same'))
    init_model.add(Activation('relu'))
    init_model.add(MaxPooling2D(pool_size=(4, 2)))
    init_model.add(Dropout(0.25))
    init_model.add(Flatten())
    init_model.add(Dense(1536, name='d_cl_1'))
    init_model.add(Activation('relu', name='a_cl_1'))
    init_model.add(Dropout(0.25, name='dr_cl_1'))
    init_model.add(Dense(num_classes, name='d_cl_2'))
    init_model.add(Activation('softmax', name='a_cl_2'))
    init_model.compile(loss='categorical_crossentropy',
                       optimizer=Adam(),
                       metrics=['accuracy'])
    return init_model
### END OF COMMON CNN LAYERS TEMPLATE ###

### CLASSIFICATION MODELS ###
Models = {}
Models['ag1'] = create_init_model(num_classes['ag1'])
Models['ag1'].summary()
Models['ag2'] = create_init_model(num_classes['ag2'])
Models['ag2'].summary()
### END OF CLASSIFICATION MODELS ##


for i in range(0, 10):
    if i % 2 == 0:
        # train on ag1
        pass
    else:
        # train on ag1
        pass


#
# model_path = './models/ag1_ag2.h5'
# model = load_model(model_path)
# print('Read Model Done')
# print(len(model.layers))
# model.summary()
# model.layers.pop()
# model.layers.pop()
# model.layers.pop()
# model.layers.pop()
# model.layers.pop()
#
# model.add(Dense(1536, name='d_cl_1'))
# model.add(Activation('relu', name='a_cl_1'))
# model.add(Dropout(0.25, name='do_cl_1'))
# model.add(Dense(num_classes, name='d_cl_2'))
# model.add(Activation('softmax', name='a_cl_2'))
# model.summary()
# print(len(model.layers))
# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(),
#               metrics=['accuracy'])
#
# loss = []
# acc = []
# val_acc = []
# start_time = datetime.datetime.now()
# for i in range(0, 100):
#     model.fit(x_train, y_train, 128, epoch_step,
#               verbose=1, validation_data=(x_test, y_test))
#     end_time = datetime.datetime.now()
#     print(str(end_time - start_time))
#     score1 = model.evaluate(x_train, y_train, verbose=0)
#     score2 = model.evaluate(x_test, y_test, verbose=0)
#     print('Train accuracy:', score1[1])
#     print('Test accuracy:', score2[1])
#     ## SAVE
#     acc.append(score1[1])
#     val_acc.append(score2[1])
#     lines = []
#     lines.append(str(end_time - start_time))
#     lines.append(','.join([str(a) for a in acc]))
#     lines.append(','.join([str(a) for a in val_acc]))
#     FileIO.write_lines_to_file('./ag1_ag2.log', lines)
#     model.save('./models/ag1_ag2.h5')
