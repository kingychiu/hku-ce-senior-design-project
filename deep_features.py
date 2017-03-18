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
    with open('./datasets/ag_7blkup_4_cl_gt_50.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()[:1000]
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
        f.close()
        return x, y, len(classes)


x, y, num_classes = get_data()
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

model_path = './models/7blkup_4classes.h5'
model = load_model(model_path)
model.summary()
print('num of layers', len(model.layers))
intermediate_layer_model = Model(input=model.input,
                                 output=model.layers[13].output)
lines = []
while len(x) != 0:
    batch_x = x[:128]
    batch_y = y[:128]
    x = x[128:]
    y = y[128:]
    print(x.shape)
    print(y.shape)
    intermediate_output = intermediate_layer_model.predict(batch_x)
    for i in range(len(intermediate_output)):
        output = intermediate_output[i]
        f = ','.join(str(output.tolist()))
        print(batch_y[i])
        print(type(batch_y[i]))
        lines.append(batch_y[i] + '|sep|' + f)
        print(lines)
    break
FileIO.write_lines_to_file('./datasets/7blkup_4classes_dfeatures.txt', lines)


#
# kmeans = KMeans(n_clusters=10, random_state=0).fit(intermediate_output)
# print(kmeans.labels_[:10])
