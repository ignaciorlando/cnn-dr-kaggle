import shutil
import csv
import os

def encode_training_data(root_dir_path, csv_labels_path, output_path, class_threshold=None, image_extension='.jpeg'):
    '''
    Given a root directory root_dir_path, a csv file with labels csv_labels_path,
    and a class_threshold, create...
    '''

    # Prepare the image path
    images_dir_path = os.path.join(root_dir_path, 'images')
    # Get the number of images to save
    num_images = len(os.listdir(images_dir_path))

    # Prepare the output path
    os.makedirs(output_path, exist_ok=True)

    # Open the CSV file with the labels
    with open(csv_labels_path, newline='\n') as csvfile:
        file_handler = csv.reader(csvfile, delimiter=',')
        current_image = 1
        next(file_handler) # skip headers in csv file
        # For each entry in the CSV file
        for row in file_handler:
            # CSV format: filename_without_extension, level
            image_filename = os.path.join(images_dir_path, row[0] + image_extension)
            # Only save images that were successfully preprocessed
            if os.path.exists(image_filename):

                # Assign image label
                if class_threshold==None:
                    image_label = row[1]
                else:
                    image_label = int(row[1]) >= class_threshold

                # Create the corresponding folder
                class_output_folder = os.path.join(output_path, str(int(image_label)))
                os.makedirs(class_output_folder, exist_ok=True)
                # Copy the image to the new output folder
                target_image_path = os.path.join(class_output_folder, row[0] + image_extension)
                shutil.copyfile(image_filename, target_image_path)

                # Print a message
                if (current_image % 100) == 0:
                    print('Processing image ' + str(current_image) + '/' + str(num_images) + '\n', end="", flush=True)
                current_image = current_image + 1


def usage():
    print('ERROR: Usage: encode_training_data.py <root_dir_path> <csv_labels_path> <output_path> <class_threshold> [--extension]')

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
        parser.add_argument("output_path", help="target folder in which files will be saved", type=str)
        parser.add_argument("threshold", help="class threshold", type=int, default=512)
        parser.add_argument("-e", "--extension", help="image extension of the files in input_directory", type=str, default='.jpeg')
        args = parser.parse_args()

        # call the main function
        encode_training_data(args.input_directory, args.csv_file, args.output_path, args.threshold, args.extension)
