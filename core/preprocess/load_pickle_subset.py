import csv
import numpy as np
from os import listdir, path
from scipy import ndimage, misc
from six.moves import cPickle as pickle

def load_pickle_subset(root_dir_path, image_count=10000, image_size=512):
    '''
    Returns two pickle files, sub_dataset and sub_labels, containing a subset of
    image_count images from the dataset in root_dir_path. In case the files do not exist, it
    creates them.
    '''

    # Prepare filenames
    sub_dataset_pickle_filename = path.join(root_dir_path, 'sub_dataset.pickle')
    labels_pickle_filename = path.join(root_dir_path, 'sub_labels.pickle')

    try:

        # Load data set
        with open(sub_dataset_pickle_filename, 'rb') as f:
            dataset = pickle.load(f)
        # Load labels
        with open(labels_pickle_filename, 'rb') as f:
            labels = pickle.load(f)

    except FileNotFoundError:

        # Get the name of the classes in the training data
        dif_classes = listdir(root_dir_path)

        # Initialize an array with dif_classes positions (one per each class)
        number_of_training_images_per_grade = np.zeros(len(dif_classes))
        for i in range(0, len(number_of_training_images_per_grade)):
            # Count the number of images in the folder
            number_of_training_images_per_grade[i] = len(listdir(path.join(root_dir_path, dif_classes[i])))

        # Get the total number of images by summing up 
        total_number_of_images = np.sum(number_of_training_images_per_grade)
        
        # Get the fractions of images
        number_of_images_to_copy = (np.round(image_count * number_of_training_images_per_grade / total_number_of_images)).astype('int16')

        # Allocate memory for all the images to pickle
        dataset = np.ndarray(shape=(image_count, image_size, image_size, 3), dtype=np.uint8)
        labels = np.ndarray(shape=(image_count, 1), dtype=np.uint8)

        print('Pickling images...')

        # Add images to the dataset
        iterator_labels = 0
        current_image = 0
        for i in range(0, len(number_of_images_to_copy)):
            # Get image filenames
            current_folder = path.join(root_dir_path, dif_classes[i])
            image_filenames = listdir(current_folder)
            # Iterate for each of the images
            for j in range(0, number_of_images_to_copy[i]):
                # Read the image and copy it to the array
                dataset[current_image, :, :, :] = misc.imresize(ndimage.imread(path.join(current_folder, image_filenames[i])), (image_size, image_size, 3))
                current_image = current_image + 1
            # Assign the labels
            labels[iterator_labels:number_of_images_to_copy[i]-1] = int(dif_classes[i])
            # Move the iterator
            iterator_labels = number_of_images_to_copy[i]

        # Dump pickle files
        dump_pickle(dataset, sub_dataset_pickle_filename)
        dump_pickle(labels, labels_pickle_filename)

    return dataset, labels


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
    print('ERROR: Usage: load_pickle_subset.py <root_dir_path> [--image_count] [--image_size]')

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
        parser.add_argument("-n", "--imagecount", help="number of images to pickle", type=int, default=10000)
        parser.add_argument("-s", "--imagesize", help="input image size (squared images)", type=int, default=512)
        args = parser.parse_args()

        # call the main function
        load_pickle_subset(args.input_directory, args.imagecount, args.imagesize)