# numpy
import numpy as np
# keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, Adadelta
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# spark, elephas
from keras.optimizers import SGD, Adam
# other
from file_io import FileIO
import datetime


def get_data():
    with open('./datasets/ag_dataset_7bit_look_up.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()
        tensor = []
        labels = []
        print(len(lines))
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
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        f.close()
        return x_train, x_test, y_train, y_test, len(classes)


x_train, x_test, y_train, y_test, num_classes = get_data()
# Convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

from keras.models import load_model
model_path = './models/gpu_look_up_cnn_epoch_60ep_3_convB_4_layers.h5'
model = load_model(model_path)
score, acc = model.evaluate(x_train, y_train, verbose=0)
print('Train accuracy:', acc)
score, acc = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', acc)