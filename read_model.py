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

def get_data():
    with open('./datasets/gistnote_highlight_7blkup.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()
        tensor = []
        labels = []
        print(len(lines))
        for line in lines:
            matrix = []
            labels.append(line.split('|l|')[0])
            if not len(line.split('|l|')) > 1:
                continue
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
        # x, y = shuffle(x, y, random_state=0)
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        f.close()
        return x


model_path = './models/switch_learning_ag12bbc.h5'
model = load_model(model_path)


x = get_data()
print(x.shape)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
print(x.shape)

p = model.predict(x)
print(p[0:3])
labels = ['business', 'entertainment', 'politics', 'sport', 'tech']