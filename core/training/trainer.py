import sys
from os import path, makedirs
# Import from sibling directory ..\api
sys.path.append(path.dirname(path.abspath(__file__)) + "/..")

from core.models import vgg16
from core.augmentation import data_augmentation, no_augmentation
from keras.callbacks import TensorBoard, CSVLogger
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

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    weights_filename = model.name + '.h5'

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
    tensorboard_path = output_path
    if path.exists(tensorboard_path):
        rmtree(tensorboard_path)
    makedirs(tensorboard_path)
    tensorboad_cb = TensorBoard(log_dir=tensorboard_path)


    # TRAIN THE MODEL
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=150,
        validation_data=validation_generator,
        validation_steps= validation_generator.samples // batch_size,
        class_weight=class_weights,
        callbacks=custom_callbacks + [tensorboad_cb])

    # SAVE THE WEIGHTS
    model.save_weights(path.join(output_path, weights_filename))

"""
def main(data_path, output_path, image_shape, batch_size):

    train_data_path = path.join(data_path, 'training')
    validation_data_path = path.join(data_path, 'validation')

    train(train_data_path, validation_data_path, output_path, image_shape, batch_size)

def usage():
    print('ERROR: Usage: vgg16.py <data_path> <output_path> [--image_shape] [--batch_size]')

import argparse
import sys

if __name__ == '__main__':

    if len(sys.argv) < 3:
        usage()
        exit()
    else:
        # create an argument parser to control the input parameters
        parser = argparse.ArgumentParser()
        parser.add_argument("data_path", help="directory with training/validation directories", type=str)
        parser.add_argument("output_path", help="directory to save the models", type=str)
        parser.add_argument("-ih", "--image_height", help="input image height", type=int, default=512)
        parser.add_argument("-iw", "--image_width", help="input image width", type=int, default=512)
        parser.add_argument("-ic", "--channels", help="input image channels", type=int, default=3)
        parser.add_argument("-b", "--batch_size", help="class threshold", type=int, default=32)
        args = parser.parse_args()

        # call the main function
        main(args.data_path, args.output_path, (args.image_height, args.image_width, args.channels), args.batch_size)
"""