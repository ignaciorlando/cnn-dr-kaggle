from skimage import io
from os import listdir, path, makedirs
from get_fov_mask import get_fov_mask
from crop_fov_mask import crop_fov_mask
from equalize_fundus_image_intensities import equalize_fundus_image_intensities


def main(root_path, output_path, mask_threshold=0.01):
    """
    This function runs the preprocessing filters to all images inside root_path/images
    and creates a separate folder in root_path/masks with the resulting FOVs.
    """

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

    # for each file in the folder
    num_images = len(listdir(input_directory))
    current_image = 0
    for file_i in listdir(input_directory):
        current_image = current_image + 1
        # if it is a file...
        if path.isfile(path.join(input_directory, file_i)):
            print(file_i)
            print('Processing image ' + str(current_image) + '/' + str(num_images), end="", flush=True)

            # read the input image
            image = io.imread(path.join(input_directory, file_i))
            # prepare output file name for the mask
            mask_filename = file_i[:file_i.rfind('.')] + '.png'

            # create the FOV mask associated to the image
            print('\tCreating mask...', end="", flush=True)
            # generate the fov mask
            mask = get_fov_mask(image, mask_threshold)

            # crop the images and the masks around the FOV
            print('\tCropping around the FOV...', end="", flush=True)
            image, mask = crop_fov_mask(image, mask)
            # save the cropped mask
            io.imsave(path.join(output_directory_masks, mask_filename), mask)

            # apply contrast equalization on the image
            print('\tPreprocessing the image...', end="", flush=True)
            preprocessed_image = equalize_fundus_image_intensities(image, mask)
            # save the preprocessed mask
            io.imsave(path.join(output_directory_images, file_i), preprocessed_image)

            #print('.', end="", flush=True)

    print()



def usage():
    print('ERROR: Usage: main.py <path-to-directory-of-image-to-be-processed> <output-path> <mask-threshold>')



import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
        exit()
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
