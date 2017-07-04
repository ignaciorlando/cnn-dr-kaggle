from skimage import io, transform
from os import listdir, path, makedirs
from get_fov_mask import get_fov_mask
from crop_fov_mask import crop_fov_mask
from equalize_fundus_image_intensities import equalize_fundus_image_intensities
import numpy as np


def main(root_path, output_path, first_image=0, last_image=None, overwrite=False, mask_threshold=0.01):
    """
    This function runs the preprocessing filters to all images inside root_path/images
    and creates a separate folder in root_path/masks with the resulting FOVs.
    """

    # check if first_image is lower than last_image
    if not(last_image is None):
        assert (first_image < last_image), 'first_image (%d) is higher than last_image (%d)' % (first_image, last_image)

    # By default, we assume that the input folder has a folder inside namely 'images'
    input_directory = path.join(root_path, 'images')

    # The output folders will be:
    # - images: will contain the preprocessed images
    output_directory_images = path.join(output_path, 'images')
    # - masks: will contain the FOV masks
    output_directory_masks = path.join(output_path, 'masks')

    # if those folders do not exist, create them
    if not path.exists(output_directory_images):
        makedirs(output_directory_images)
    if not path.exists(output_directory_masks):
        makedirs(output_directory_masks)

    # get all the files in the folder
    all_filenames = listdir(input_directory)

    # check if the last image does not exceed the amount of available images
    num_images = len(all_filenames)
    if last_image is None:
        last_image = num_images - 1
    assert (last_image <= num_images), 'last_image (%d) is higher than the number of images (%d)' % (last_image, num_images)

    # get only the images given by first_image and last_image
    images_to_process = all_filenames[first_image:last_image+1]

    # for each image in the folder
    current_image = 0
    num_images = last_image - first_image + 1
    for file_i in images_to_process:

        # (if it is a file...) and ((overwrite) or (not overwrite and the image was processed before ))
        if path.isfile(path.join(input_directory, file_i)):

            # move the counter of images
            current_image = current_image + 1
            # prepare output file name for the mask
            mask_filename = file_i[:file_i.rfind('.')] + '.png'

            # if we have to overwrite or if we don't by the mask doesn't exist
            if overwrite or (not overwrite and not path.exists(path.join(output_directory_masks, mask_filename))):

                print(file_i)
                print('Processing image ' + str(current_image) + '/' + str(num_images), end="", flush=True)

                # read the input image
                image = io.imread(path.join(input_directory, file_i))

                # create the FOV mask associated to the image
                print('\tCreating mask...', end="", flush=True)
                # generate the fov mask
                mask = get_fov_mask(image, mask_threshold)

                # crop the images and the masks around the FOV
                print('\tCropping around the FOV...', end="", flush=True)
                preprocessed_image, mask = crop_fov_mask(image, mask)

                # downsize the image and the mask
                print('\Resizing the image...', end="", flush=True)
                preprocessed_image = transform.resize(preprocessed_image, (512, 512, 3), preserve_range=True).astype(np.uint8)
                mask = transform.resize(mask, (512, 512), order=0, preserve_range=True)

                # apply contrast equalization on the image
                print('\tPreprocessing the image...', end="", flush=True)
                preprocessed_image = equalize_fundus_image_intensities(preprocessed_image, mask)

                # save the preprocessed image
                io.imsave(path.join(output_directory_images, file_i), preprocessed_image)
                # save the cropped mask
                io.imsave(path.join(output_directory_masks, mask_filename), mask)

            #print('.', end="", flush=True)

    print()



def usage():
    print('ERROR: Usage: main.py <path-to-directory-of-image-to-be-processed> <output-path> <mask-threshold>')


import argparse
import sys

if __name__ == '__main__':


    if len(sys.argv) < 3:
        usage()
        exit()
    else:
        # lets create an argument parser to control the input parameters
        parser = argparse.ArgumentParser()
        parser.add_argument("input_directory", help="folder with the input images", type=str)
        parser.add_argument("output_directory", help="folder with the output images", type=str)
        parser.add_argument("-f", "--first_image", help="first image to process", type=int, default=0)
        parser.add_argument("-l", "--last_image", help="last image to process", type=int, default=None)
        parser.add_argument("-o", "--overwrite", help="indicate if we should overwrite the precomputed files", action="store_true")
        parser.add_argument("-t", "--mask_threshold", help="threshold to compute the masks", type=float, default=0.01)
        args = parser.parse_args()

        # call the main function
        main(args.input_directory, args.output_directory, args.first_image, args.last_image, args.overwrite, args.mask_threshold)
