# numpy
import numpy as np
# keras
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Model
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

model_path = './models/7blkup_4classes.h5'
model = load_model(model_path)
model.sumary()
# layer_name = 'my_layer'
# intermediate_layer_model = Model(inputs=model.input,
#                                  outputs=model.get_layer(layer_name).output)
# intermediate_output = intermediate_layer_model.predict(data)