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


def build(image_size, config):

    # reorder according to backend
    if K.image_dim_ordering() == 'th':
        input_shape = (3,) + image_size
    else:
        input_shape = image_size + (3, )

    # get batch normalization from config
    batch_normalizations = config['batch_normalization'].split()
    # get dropout probability
    dropout_prob = float(config['dropout'])

    # initialize VGG-16
    model = vgg16.VGG16(include_top=False, input_tensor=Input(input_shape), weights=None)

    x = model.input
    y = model.output
    y = Flatten()(y)
    #y = BatchNormalization(axis=1)(y)

    # first fully connected module
    y = Dense(4096)(y)
    if 'fc' in batch_normalizations:
        y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)
    if dropout_prob > 0.0:
        y = Dropout(dropout_prob)(y)

    # second fully connected module
    y = Dense(4096)(y)
    if 'fc' in batch_normalizations:
        y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)
    if dropout_prob > 0.0:
        y = Dropout(dropout_prob)(y)
    
    # prediction
    y = Dense(1)(y)
    y = Activation(config['prediction_activation'])(y)

    model = Model(inputs=x, outputs=y)

    """
    print("Writing model into json file")
    text_file = open("vgg16.json", "w")
    text_file.write(model.to_json())
    text_file.close()
    """
    return model