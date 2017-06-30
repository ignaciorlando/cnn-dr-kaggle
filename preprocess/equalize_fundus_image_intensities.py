from skimage import io, color, measure, filters
from scipy import ndimage
import numpy as np

def equalize_fundus_image_intensities(image_rgb, fov_mask):

    # these constants were assigned according to van Grinsven et al. 2016, TMI
    alpha = 4.0
    beta = -4.0
    gamma = 128.0

    # get image size
    image_size = image_rgb.shape

    # initialize the output image with the same size that the input image
    equalized_image = np.zeros((image_size), dtype=np.uint8)

    # estimate the sigma parameter using the scaling approach by
    # Orlando et al. 2017, arXiv
    sigma = image_size[0] / 30.0

    # for each color band, apply:
    for color_band in range(0, image_size[2]):

        # apply a gaussian filter on the current band to estimate the background
        smoothed_band = ndimage.filters.gaussian_filter(image_rgb[:, :, color_band], sigma)
        # apply the equalization procedure on the current band
        equalized_image[:, :, color_band] = alpha * image_rgb[:, :, color_band] + beta * smoothed_band + gamma
        # remove elements outside the fov
        equalized_image[:, :, color_band] = np.multiply(equalized_image[:, :, color_band], fov_mask > 0)

    return equalized_image

def main(image_path, fov_mask_path):
    image = io.imread(image_path)
    fov_mask = io.imread(fov_mask_path)
    equalized_image = equalize_fundus_image_intensities(image, fov_mask)

    io.imsave('equalized_image.png', equalized_image.astype(np.uint8))

import sys

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
