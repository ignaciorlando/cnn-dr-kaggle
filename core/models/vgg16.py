import sys
from os import path, makedirs
# Import from sibling directory ..\api
sys.path.append(path.dirname(path.abspath(__file__)) + "/..")
from core.metrics.custom_binary_metrics import precision, recall, f1

import keras.backend as K
from keras.applications import vgg16
from keras.models import Model
from keras.optimizers import Adam, SGD, rmsprop
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization, Activation
from keras import metrics


def build(image_size=None):
    image_size = image_size or (512, 512)
    if K.image_dim_ordering() == 'th':
        input_shape = (3,) + image_size
    else:
        input_shape = image_size + (3, )
    model = vgg16.VGG16(include_top=False, input_tensor=Input(input_shape), weights=None)

    x = model.input
    y = model.output
    y = Flatten()(y)
    y = BatchNormalization()(y)
    y = Dense(256, activation='relu')(y) #y = Dense(4096, activation='relu')(y)
    #y = Dropout(.5)(y)
    #y = Dense(256, activation='relu')(y) #y = Dense(4096, activation='relu')(y)
    #y = Dropout(.5)(y)
    y = Dense(1)(y)
    y = Activation('sigmoid')(y)

    model = Model(inputs=x, outputs=y)

    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    #rms = rmsprop(lr=0.05, decay=1e-6)
    #adam = Adam(lr=1e-2)

    model.compile(optimizer=sgd, loss = 'binary_crossentropy', metrics=[precision, recall, f1])
    return model