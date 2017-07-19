import keras.backend as K
from keras.applications import vgg16
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization, Activation
from keras import metrics

def build(image_size=None):
    image_size = image_size or (512, 512)
    if K.image_dim_ordering() == 'th':
        input_shape = (3,) + image_size
    else:
        input_shape = image_size + (3, )
    model = vgg16.VGG16(include_top=False, input_tensor=Input(input_shape), weights=None)

    #bottleneck_model.trainable = False
    #for layer in bottleneck_model.layers:
        #layer.trainable = False

    #model.add(Flatten())
    #model.add(BatchNormalization())
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(.5))
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(.5))
    #model.add(Dense(2))

    x = model.input
    y = model.output
    y = Flatten()(y)
    #y = BatchNormalization()(y)
    y = Dense(4096, activation='relu')(y)
    y = Dropout(.5)(y)
    y = Dense(4096, activation='relu')(y)
    y = Dropout(.5)(y)
    y = Dense(2)(y)
    #y = Activation('softmax')(y)

    model = Model(input=x, output=y)

    model.compile(optimizer=Adam(lr=1e-4), loss = 'sparse_categorical_crossentropy', metrics=[metrics.binary_accuracy])
    return model