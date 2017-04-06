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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import operator

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
        look_up_vector = []
        for digit_str in binary_look_up:
            if digit_str == '0' or digit_str == '1':
                look_up_vector.append(int(digit_str))
        look_up_matrix.append(look_up_vector)
    return look_up_matrix


print('loading model')
model_path = './models/switch_learning_ag12bbc.h5'
model = load_model(model_path)

intermediate_layer_model = Model(input=model.input,
                                 output=model.layers[12].output)
del model
intermediate_layer_model.summary()


def get_deep_features(string):
    input = string27Bits(string)
    input = np.array([input])
    # (135386, 100, 7, 1)
    print(input.shape)
    input = input.reshape(input.shape[0], input.shape[1], input.shape[2], 1)
    intermediate_output = intermediate_layer_model.predict(input)
    return intermediate_output


print('loading knn')

stat = {}

with open('./datasets/switch_ag12bbc.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()
    print(len(lines))
    labels = []
    features = []
    for line in lines:
        label = line.split('|sep|')[0]
        if label in stat.keys() and stat[label] < 5000:
            stat[label] += 1
            labels.append(label)
            features.append(line.split('|sep|')[1].split(','))
        elif label not in stat.keys():
            print(label)
            stat[label] = 1
            labels.append(label)
            features.append(line.split('|sep|')[1].split(','))
        else:
            pass
    f.close()

print(stat)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3,
                                                    random_state=42)
classes = sorted(list(set(y_train)))
print(classes)
print('train', len(x_train))
print('test', len(x_test))
del labels
del features

neigh = KNeighborsClassifier(n_neighbors=50)
neigh.fit(x_train, y_train)

word_vectors = []
with open('./datasets/word2vec_ag12bbc.txt', 'r', encoding='utf8') as original_data:
    lines = original_data.readlines()
    word_vectors = [np.array(l.split('|sep|')[1].split(',')) for l in lines]


def get_labels(string):
    sample = get_deep_features(string)
    distances, neighbors = neigh.kneighbors(sample)
    return neighbors[0]


print('ready')
with open('./fb_posts/Cristiano.txt', 'r', encoding='utf8') as fb_posts:
    lines = fb_posts.readlines()
    count = 0
    for line in lines:
        neighbors = get_labels(line)
        for n in neighbors:
            print(n, word_vectors[n])
        count += 1
        print(count)
        break
