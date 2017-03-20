# numpy
import numpy as np
# keras
from keras.utils import np_utils
from keras.models import load_model
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

model_path = './models/7blkup_4classes.h5'
model = load_model(model_path)
print('Read Model Done')
model.summary()
print(len(model.layers))
# remove the classification layers
