import sys
from os import path, makedirs
# Import from sibling directory ..\api
sys.path.append(path.dirname(path.abspath(__file__)) + "/..")

from core.models import vgg16
from core.augmentation import data_augmentation, no_augmentation
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from shutil import rmtree

import keras.backend as K

def identify_classes(input_data_path):
    # TODO: deduce this from input_data_path directory structure
    classes = ['0', '1']
    class_mode = 'binary'
    return classes, class_mode, None

def train(
    model,
    loss,
    metrics,
    optimizer,
    custom_callbacks,
    data_augmentation_policy,
    input_data_path,
    output_path,
    image_shape=(512, 512, 3),
    batch_size=32):
    """
    """

    # create output directory if it does not exist
    if not path.exists(output_path):
        makedirs(output_path)

    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # print the model
    model.summary()

    # identify classes
    classes, class_mode, class_weights = identify_classes(input_data_path)

    ### SET UP THE NETWORK ARCHITECTURE

    # SET UP THE TRAINING AND VALIDATION DATA GENERATION POLICIES
    train_data_generator = data_augmentation_policy.data_augmentation('training')
    validation_data_generator = data_augmentation_policy.data_augmentation('validation')

    train_generator = train_data_generator.flow_from_directory(
        path.join(input_data_path, 'training'),
        target_size=(image_shape[0], image_shape[1]),
        batch_size=batch_size,
        classes=classes,
        class_mode=class_mode)

    # this is a similar generator, for validation data
    validation_generator = validation_data_generator.flow_from_directory(
        path.join(input_data_path, 'validation'),
        target_size=(image_shape[0], image_shape[1]),
        batch_size=batch_size,
        classes=classes,
        class_mode=class_mode)

    # initialize callbacks
    # 1. tensorboard callback
    tensorboard_path = path.join(output_path)
    if path.exists(tensorboard_path):
        rmtree(tensorboard_path)
    makedirs(tensorboard_path)
    tensorboad_cb = TensorBoard(log_dir=tensorboard_path, write_images=False)
    # 2. checkpoint callback
    checkpoint_path = path.join(output_path)
    checkpointer_cb = ModelCheckpoint(filepath=path.join(checkpoint_path, 'weights.hdf5'), verbose=1, save_best_only=True)

    # TRAIN THE MODEL
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=150,
        validation_data=validation_generator,
        validation_steps= validation_generator.samples // batch_size,
        class_weight=class_weights,
        callbacks=custom_callbacks + [tensorboad_cb, checkpointer_cb])
