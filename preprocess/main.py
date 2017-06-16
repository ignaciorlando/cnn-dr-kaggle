from skimage import io
from os import walk, path
from get_fov_mask import get_fov_mask

def create_mask(dirpath, filename, threshold):
    mask_extension = ".png"

    image = io.imread(path.join(dirpath,filename))
    mask = get_fov_mask(image, threshold).astype(float)

    mask_fullpath = dirpath + filename.split('.')[0] + '_mask' + mask_extension
    io.imsave(mask_fullpath, mask)

def main(directory):

    mask_threshold = 0.0001

    for (dirpath, dirnames, filenames) in walk(directory):
        for file_i in filenames:
            create_mask(dirpath, file_i, mask_threshold)
        break

def usage():
    print('ERROR: Usage: main.py <path-to-directory-of-image-to-be-processed>')

import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()

    main(sys.argv[1])