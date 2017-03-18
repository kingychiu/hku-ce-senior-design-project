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

model_path = './models/7blkup_4classes.h5'
model = load_model(model_path)
model.summary()
print('num of layers', len(model.layers))

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])

# layer_output = get_3rd_layer_output([X])[0]