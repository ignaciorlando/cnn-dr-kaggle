import csv
import numpy as np
from os import listdir, path
from scipy import ndimage
from six.moves import cPickle as pickle

def pickle_dataset(root_dir_path, csv_labels_path, image_extension='.jpeg', image_size=512):
    '''
    Given a root directory root_dir_path, and a csv file with labels csv_labels_path,
    save two files, dataset.pickle and labels.pickle, into root_dir_path.
    '''

    # Get number of images to pickle
    images_dir_path = path.join(root_dir_path, 'images')
    images_count = len(listdir(images_dir_path))

    dataset_pickle_path = path.join(root_dir_path, 'dataset.pickle')
    labels_pickle_path = path.join(root_dir_path, 'labels.pickle')

    # Allocate memory for all the images to pickle
    print('float32: ' +str(images_count*image_size*image_size*3*32/8/1024/1024/1024) + 'GBs')
    print('uint8: ' + str(images_count*image_size*image_size*3*8/8/1024/1024/1024) + 'GBs')
    dataset = np.ndarray(shape=(images_count, image_size, image_size, 3),
                         dtype=np.float32)
    labels = np.ndarray(shape=(images_count, 1), dtype=np.uint8)

    file_handler = csv.reader(csv_labels_path, delimeter=',')
    current_image = 0
    next(file_handler) # skip headers in csv file
    for row in file_handler:
        # CSV format: filename_without_extension, level
        image_filename = row[0] + image_extension

        # Only pickle images that were successfully preprocessed
        if path.exists(image_filename):
            dataset[current_image, :, :, :] = ndimage.imread(path.join(images_dir_path, image_filename)).astype(np.float)

            labels[current_image, 0] = row[1].astype(np.uint8)

            current_image = current_image + 1

    dump_pickle(dataset, dataset_pickle_path)
    dump_pickle(labels, labels_pickle_path)

def dump_pickle(np_array, dst_filename):
    '''
    Given a numpy array with data, pickle it in a file with name dst_filename.
    '''

    try:
        with open(dst_filename, 'wb') as f:
            pickle.dump(np_array, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', dst_filename, ':', e)

def usage():
    print('ERROR: Usage: pickling.py <root_dir_path> <csv_labels_path> [--extension] [--size]')

import argparse
import sys

if __name__ == '__main__':

    if len(sys.argv) < 3:
        usage()
        exit()
    else:
        # create an argument parser to control the input parameters
        parser = argparse.ArgumentParser()
        parser.add_argument("input_directory", help="folder with the input images", type=str)
        parser.add_argument("csv_file", help="csv file with the labels", type=str)
        parser.add_argument("-e", "--extension", help="image extension of the files in input_directory", type=str, default='.jpeg')
        parser.add_argument("-s", "--size", help="input image size (squared images)", type=int, default=512)
        args = parser.parse_args()

        # call the main function
        pickle_dataset(args.input_directory, args.csv_file, args.extension, args.size)