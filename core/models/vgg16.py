# -*- coding: utf-8 -*-

import sys
from os import path, makedirs
# Import from sibling directory ..\api
sys.path.append(path.dirname(path.abspath(__file__)) + "/..")
from core.metrics.custom_binary_metrics import precision, recall, f1

import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam, SGD, rmsprop
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras import metrics
from keras import regularizers
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
#from ..utils.data_utils import get_file
from keras.applications.imagenet_utils import _obtain_input_shape

import warnings


"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""

def VGG16(input_tensor=None, input_shape=None,
          pooling=None,
          regularizer=None,
          weight_decay=5e-4):

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      include_top=False)


    if regularizer not in {'l1', 'l2', None}:
        raise ValueError('The `regularizer` argument should be either '
                         '`None` (no regularization), `l1` or `l2`.')                  

    if weight_decay >= 1:
            raise ValueError('The `weight_decay` argument should be a number in the [0, 1) interval.')                         

    if regularizer=='l1':
        reg = regularizers.l1(float(weight_decay))
    elif regularizer=='l2':
        reg = regularizers.l2(float(weight_decay))
    else:
        reg = None

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=reg)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=reg)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=reg)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=reg)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=reg)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=reg)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=reg)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=reg)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=reg)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=reg)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=reg)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=reg)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=reg)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    return model



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
    model = VGG16(input_tensor=Input(input_shape), 
                  regularizer=config['regularizer'],
                  weight_decay=float(config['weight_decay']))

    x = model.input
    y = model.output
    y = Flatten()(y)
    if 'last-conv' in batch_normalizations:
        y = BatchNormalization()(y)

    # first fully connected module
    y = Dense(4096)(y)
    if 'fc' in batch_normalizations:
        y = BatchNormalization()(y)
    y = Activation('relu')(y)
    if dropout_prob > 0.0:
        y = Dropout(dropout_prob)(y)

    # second fully connected module
    y = Dense(4096)(y)
    if 'fc' in batch_normalizations:
        y = BatchNormalization()(y)
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