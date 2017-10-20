#!/usr/bin/env python3

import sys
from os import path, makedirs

from core.metrics.custom_binary_metrics import tp, fp, fn
from core.metrics.Confusion_Matrix import Confusion_Matrix

from configparser import ConfigParser

from core.models import vgg16
from keras.optimizers import SGD
from core.augmentation import data_augmentation
from core.preprocess import load_pickle_subset
from keras.callbacks import TensorBoard, CSVLogger
from shutil import rmtree
from numpy import random
import ntpath

import keras.backend as K

REPRESENTATIVE_SAMPLES = 10000

def identify_classes(input_data_path, use_weight):
    # TODO: deduce this from input_data_path directory structure
    classes = ['0', '1']
    class_mode = 'binary'
    if use_weight:
        weights = []
    else:
        weights = None
    return classes, class_mode, weights


def main(input_data_path, output_path, config_file):
    
    # fix random seed for reproducibility
    seed = 7
    random.seed(seed)

    # append the name of the configuration file to the output path
    output_path = path.join(output_path, ntpath.basename(config_file)[:-4])

    # read the configuration file    
    config = ConfigParser()
    config.read(config_file)

    # get image shape
    image_shape = [int(i) for i in config['input']['image_shape'].split() ]

    # initialize the CNN architecture
    if config['architecture']['architecture']=='vgg16':
        model = vgg16.build((image_shape[0], image_shape[1]), config['architecture'])
    elif config['architecture']['architecture']=='inception-v4':
        pass
    elif config['architecture']['architecture']=='resnet':
        pass

    # initialize the optimizer
    if config['optimizer']['optimizer']=='SGD':
        optimizer = SGD(lr=float(config['optimizer']['lr']),
                        decay=float(config['optimizer']['decay']),
                        momentum=float(config['optimizer']['momentum']),
                        nesterov=(config['optimizer']['nesterov']=='True'))

    # assign evaluation metrics
    metrics = []
    if 'tp' in config['evaluation']['metrics'].split():
        metrics = metrics + [tp]
    if 'fp' in config['evaluation']['metrics'].split():
        metrics = metrics + [fp]
    if 'fn' in config['evaluation']['metrics'].split():
        metrics = metrics + [fn]            
    #TODO: if there is crap in the configuration file, explode

    # assign the loss function
    loss = config['loss']['loss']

    # assign the batch size
    training_batch_size = int(config['training']['batch_size'])
    validation_batch_size = int(config['validation']['batch_size'])

    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    # identify the classes
    classes, class_mode, class_weights = identify_classes(input_data_path, (config['loss']['weighted']=='True'))

    # initialize the image data generator
    # - for the training data
    train_data_generator = data_augmentation.get_image_data_generator('training', config['augmentation'])
    train_generator = train_data_generator.flow_from_directory(
        path.join(input_data_path, 'training'),
        target_size=(image_shape[0], image_shape[1]),
        batch_size=training_batch_size,
        classes=classes,
        class_mode=class_mode)
    # - for validation data
    validation_data_generator = data_augmentation.get_image_data_generator('validation', config['augmentation'])
    validation_generator = validation_data_generator.flow_from_directory(
        path.join(input_data_path, 'validation'),
        target_size=(image_shape[0], image_shape[1]),
        batch_size=validation_batch_size,
        classes=classes,
        class_mode=class_mode)

    # initialize callbacks
    tensorboard_path = output_path
    if path.exists(tensorboard_path):
        rmtree(tensorboard_path)
    makedirs(tensorboard_path)
    tensorboad_cb = TensorBoard(log_dir=tensorboard_path)

    # load pickles for computing statistics
    X_subset, y_labels = load_pickle_subset.load_pickle_subset(path.join(input_data_path, 'training'), 
                                                               REPRESENTATIVE_SAMPLES, 
                                                               image_shape[0])
    # compute statistics for normalization
    train_data_generator.fit(X_subset)
    validation_data_generator.mean = train_data_generator.mean
    validation_data_generator.std = train_data_generator.std

    # create output directory if it does not exist
    if not path.exists(output_path):
        makedirs(output_path)

    # write configuration file in the output folder
    with open(path.join(output_path, ntpath.basename(config_file)), 'w') as config_output_file:
        config.write(config_output_file)

    # TRAIN THE MODEL
    model.fit_generator(
        train_generator,
        steps_per_epoch=(train_generator.samples // training_batch_size) * float(config['training']['steps_per_epoch_coefficient']),
        epochs=int(config['training']['epochs']),
        validation_data=validation_generator,
        validation_steps= (validation_generator.samples // validation_batch_size),
        class_weight=class_weights,
        callbacks=[Confusion_Matrix()] + [tensorboad_cb])

    # SAVE THE WEIGHTS
    model.save_weights(path.join(output_path, model.name + '.h5'))


def usage():
    print('ERROR: Usage: test.py <data_path> <output_path> [--image_shape] [--batch_size]')

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
        parser.add_argument("config_file", help="configuration file", type=str)

        args = parser.parse_args()

        # call the main function
        main(args.data_path, args.output_path, args.config_file)
