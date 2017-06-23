from skimage import io
from os import listdir, path, makedirs
from get_fov_mask import get_fov_mask

def create_mask(output_directory, input_directory, filename, threshold):

    image = io.imread(path.join(input_directory, filename))
    mask = get_fov_mask(image, threshold).astype(float)

    io.imsave(path.join(output_directory, filename), mask)

def main(root_path, mask_threshold):
    """
    This function runs the preprocessing filters to all images inside root_path/images
    and creates a separate folder in root_path/masks with the resulting FOVs.
    """

    input_directory = path.join(root_path, 'images')
    output_directory = path.join(root_path, 'masks')

    if not path.exists(output_directory):
        makedirs(output_directory)

    for file_i in listdir(input_directory):
        if path.isfile(path.join(input_directory, file_i)):
            create_mask(output_directory, input_directory, file_i, mask_threshold)
            print('.')

def usage():
    print('ERROR: Usage: main.py <path-to-directory-of-image-to-be-processed> <mask-threshold>')

import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        usage()
        exit()

    main(sys.argv[1], float(sys.argv[2]))