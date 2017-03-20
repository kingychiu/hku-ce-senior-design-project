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
# remove and add the new classifier
model = Model(input=model.input,
                  output=model.layers[13].output)
model = Sequential(model.layers)
n = model.output_shape[1]
model.add(Dense(n))
model.add(Activation('relu', name='cls_act1'))
model.add(Dropout(0.25))
model.add(Dense(5))
model.add(Activation('softmax', name='cls_act2'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
