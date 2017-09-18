import shutil
import csv
from os import listdir, path, makedirs
import numpy as np
from math import floor

def encode_training_data(root_dir_path, csvfile, output_path, class_threshold=None, image_extension='.jpeg', training_proportion=0.9):
    '''
    Given a root directory root_dir_path, a csv file with labels csv_labels_path,
    and a class_threshold, create...
    '''

    # Prepare the image path
    images_dir_path = path.join(root_dir_path, 'images')
    # Get the number of images to save
    num_images = len(listdir(images_dir_path))

    # Prepare the output paths
    training_data_path = path.join(output_path, 'training')
    makedirs(training_data_path, exist_ok=True)
    validation_data_path = path.join(output_path, 'validation')
    makedirs(validation_data_path, exist_ok=True)

    # Prepare a dictionary from the CSV file
    filename_labels_dictionary = initialize_dictionary_from_file(csvfile)

    # Count the number of different labels to build a balanced training/validation split
    counter_labels = dict()
    for key, value in filename_labels_dictionary.items():
        if not (str(value) in counter_labels.keys()):
            counter_labels[value] = 1
        else:
            counter_labels[value] = counter_labels[value] + 1
    
    # Compute the size of each subset 
    number_of_training_images_per_grade = np.zeros(len(counter_labels))
    for key in counter_labels.keys():
        number_of_training_images_per_grade[int(key)] = floor(counter_labels[key] * training_proportion)

    # For each image in the input path
    image_filenames = listdir(images_dir_path)
    current_image_number = 1
    num_images = len(image_filenames)
    for current_image_filename in image_filenames:
        # Get only image name, without the extension, to look up for the label in the dictionary
        image_entry_name = current_image_filename[:current_image_filename.rfind('.')]
        # Get current label
        current_label = int(filename_labels_dictionary[image_entry_name])
        # Check if we still need to move images to the training set
        if number_of_training_images_per_grade[current_label] > 0:
            current_output_folder = training_data_path
            number_of_training_images_per_grade[current_label] = number_of_training_images_per_grade[current_label] - 1 
        else:
            current_output_folder = validation_data_path            
        # Prepare the true label
        if class_threshold != None:
            current_label = int(current_label >= class_threshold)
        # Create the corresponding folder
        current_output_folder = path.join(current_output_folder, str(int(current_label)))
        makedirs(current_output_folder, exist_ok=True)
        # Copy the image to the target folder
        shutil.copyfile(path.join(images_dir_path, current_image_filename), 
                        path.join(current_output_folder, current_image_filename))
        # Print a message
        if (current_image_number % 100) == 0:
            print('Processing image ' + str(current_image_number) + '/' + str(num_images) + '\n', end="", flush=True)
        current_image_number = current_image_number + 1


def initialize_dictionary_from_file(filename):
    with open(filename, mode='r', newline='\n') as infile:
        reader = csv.reader(infile, delimiter=',')
        next(reader) # skip first line
        mydict = {rows[0]:rows[1] for rows in reader}
        return mydict

def usage():
    print('ERROR: Usage: encode_training_data.py <root_dir_path> <csv_file> <output_path> <class_threshold> [--extension] [--proportion]')

import argparse
import sys

if __name__ == '__main__':

    if len(sys.argv) < 4:
        usage()
        exit()
    else:
        # create an argument parser to control the input parameters
        parser = argparse.ArgumentParser()
        parser.add_argument("input_directory", help="folder with the input images", type=str)
        parser.add_argument("csv_file", help="csv file with the labels", type=str)
        parser.add_argument("output_path", help="target folder in which files will be saved", type=str)
        parser.add_argument("threshold", help="class threshold", type=int, default=None)
        parser.add_argument("-e", "--extension", help="image extension of the files in input_directory", type=str, default='.jpeg')
        parser.add_argument("-p", "--proportion", help="proportion of images to use for trainig (between 0-1)", type=str, default=0.9)
        args = parser.parse_args()

        # call the main function
        encode_training_data(args.input_directory, args.csv_file, args.output_path, args.threshold, args.extension, args.proportion)
