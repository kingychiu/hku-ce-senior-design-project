# numpy
import numpy as np
# keras
from keras.utils import np_utils
from keras.models import load_model,Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, Adadelta
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

model_path = './models/7blkup_4classes.h5'
model = load_model(model_path)
print('Read Model Done')
print(len(model.layers))
model.summary()
model.layers.pop()
model.layers.pop()
model.layers.pop()
model.layers.pop()
model.layers.pop()

model.add(Dense(1536))
model.add(Activation('relu', name='a_cl_1'))
model.add(Dropout(0.25, name='do_cl_1'))
model.add(Dense(5))
model.add(Activation('softmax', name='a_cl_2'))
model.summary()
print(len(model.layers))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
