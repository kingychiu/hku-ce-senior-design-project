# numpy
import numpy as np
# keras
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Model
from keras import backend as K
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from file_io import FileIO


def char2lookup(char):
    if ord(char) >= 32 and ord(char) <= 127:
        return ord(char) - 32
    else:
        return 95

def string27Bits(string):
    asciis = [char2lookup(char) for char in list(string)]
    if len(asciis) >= 100:
        asciis = asciis[:100]
    else:
        diff = 100 - len(asciis)
        asciis += (diff * [26])
    look_up_matrix = []
    for char in asciis:
        binary_look_up = '{0:07b}'.format(char)
        if (len(binary_look_up) != 7):
            print(len(binary_look_up))
        look_up_matrix.append(binary_look_up)
    return look_up_matrix




model_path = './models/switch_learning_ag12bbc.h5'
model = load_model(model_path)

print('num of layers', len(model.layers))
intermediate_layer_model = Model(input=model.input,
                                 output=model.layers[12].output)
intermediate_layer_model.summary()


def get_deep_features(string):
    input = string27Bits(string)
    intermediate_output = intermediate_layer_model.predict(input)
    return intermediate_output