# numpy
import numpy as np
# keras
from keras.utils import np_utils
from keras.models import load_model
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def get_data():
    with open('./datasets/ag_7blkup_4_cl_gt_50.txt', 'r', encoding='utf8') as f:
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
        f.close()
        return x, y, len(classes)


x, y, num_classes = get_data()
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

print('Read Data Done')
# Convert class vectors to binary class matrices
y = np_utils.to_categorical(y, num_classes)


model_path = './models/7blkup_4classes.h5'
model = load_model(model_path)
print('Read Model Done')
model.summary()
score, acc = model.evaluate(x, y, verbose=0)
print('Whole accuracy:', acc)
