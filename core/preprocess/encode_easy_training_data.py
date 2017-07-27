import shutil
import csv
from os import listdir, path, makedirs
import numpy as np
from math import floor

def encode_easy_training_data(root_dir_path, csvfile, output_path, num_images=500, image_extension='.jpeg', training_proportion=0.9):
    '''
    Given a root directory root_dir_path, a csv file with labels csv_labels_path,
    and a class_threshold, create...
    '''

    # Prepare the image path
    images_dir_path = path.join(root_dir_path, 'images')

    # Prepare the output paths
    training_data_path = path.join(output_path, 'training')
    makedirs(training_data_path, exist_ok=True)
    validation_data_path = path.join(output_path, 'validation')
    makedirs(validation_data_path, exist_ok=True)

    # Prepare a dictionary from the CSV file
    filename_labels_dictionary = initialize_dictionary_from_file(csvfile)
    
    # Compute the size of each subset 
    print(num_images)
    print(training_proportion)
    number_of_training_images_per_grade = np.multiply(np.ones(2), floor(num_images * training_proportion))
    number_of_validation_images_per_grade = np.multiply(np.ones(2), floor(num_images * (1-training_proportion)))

    # For each image in the input path
    image_filenames = listdir(images_dir_path)
    for current_image_filename in image_filenames:
        # Get only image name, without the extension, to look up for the label in the dictionary
        image_entry_name = current_image_filename[:current_image_filename.rfind('.')]  
        # Get current label
        current_label = int(filename_labels_dictionary[image_entry_name])
        # We will only copy those images with label = 0 or 4
        if (current_label==0) or (current_label==4):
            # If the label is 4, generate 1
            if current_label==4:
                current_label=1  
            valid_copy = False                
            # Check if we still need to move images to the training set
            if number_of_training_images_per_grade[current_label] > 0:
                current_output_folder = training_data_path
                number_of_training_images_per_grade[current_label] = number_of_training_images_per_grade[current_label] - 1 
                valid_copy = True
            else:
                # Check if we still need to move images to the validation set
                if number_of_validation_images_per_grade[current_label] > 0:
                    number_of_validation_images_per_grade[current_label] = number_of_validation_images_per_grade[current_label] - 1
                    current_output_folder = validation_data_path            
                    valid_copy = True
            # If we need to copy the file        
            if valid_copy:
                # Create the corresponding folder
                current_output_folder = path.join(current_output_folder, str(int(current_label)))
                makedirs(current_output_folder, exist_ok=True)
                # Copy the image to the target folder
                shutil.copyfile(path.join(images_dir_path, current_image_filename), 
                                path.join(current_output_folder, current_image_filename))


def initialize_dictionary_from_file(filename):
    with open(filename, mode='r', newline='\n') as infile:
        reader = csv.reader(infile, delimiter=',')
        next(reader) # skip first line
        mydict = {rows[0]:rows[1] for rows in reader}
        return mydict

def usage():
    print('ERROR: Usage: encode_training_data.py <root_dir_path> <csv_file> <output_path> [--number_of_images] [--extension] [--proportion]')

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
        parser.add_argument("-n", "--number_of_images", help="number of images", type=int, default=500)
        parser.add_argument("-e", "--extension", help="image extension of the files in input_directory", type=str, default='.jpeg')
        parser.add_argument("-p", "--proportion", help="proportion of images to use for trainig (between 0-1)", type=str, default=0.9)
        args = parser.parse_args()

        # call the main function
        encode_easy_training_data(args.input_directory, args.csv_file, args.output_path, args.number_of_images, args.extension, args.proportion)
