from skimage import io, color, measure, filters
from scipy import ndimage
import numpy as np
from replace_out_of_fov_pixels import replace_out_of_fov_pixels

def equalize_fundus_image_intensities(image_rgb, fov_mask):

    # replace out of fov pixels with the average intensity
    image_rgb = replace_out_of_fov_pixels(image_rgb, fov_mask).astype(float)

    # these constants were assigned according to van Grinsven et al. 2016, TMI
    alpha = 4.0
    beta = -4.0
    gamma = 128.0

    # get image size
    image_size = image_rgb.shape

    # initialize the output image with the same size that the input image
    equalized_image = np.zeros(image_size).astype(float)

    # estimate the sigma parameter using the scaling approach by
    # Orlando et al. 2017, arXiv
    sigma = image_size[1] / 30.0

    # for each color band, apply:
    for color_band in range(0, image_size[2]):

        # apply a gaussian filter on the current band to estimate the background
        smoothed_band = ndimage.filters.gaussian_filter(image_rgb[:, :, color_band], sigma, truncate=3)
        # apply the equalization procedure on the current band
        equalized_image[:, :, color_band] = alpha * image_rgb[:, :, color_band] + beta * smoothed_band + gamma
        # remove elements outside the fov
        intermediate = np.multiply(equalized_image[:, :, color_band], fov_mask > 0)

        intermediate[intermediate>255] = 255
        intermediate[intermediate<0] = 0

        equalized_image[:, :, color_band] = intermediate
        #equalized_image[:, :, color_band] = 255 * (equalized_image[:, :, color_band] - np.min(equalized_image[:, :, color_band])) / (np.max(equalized_image[:, :, color_band]) - np.min(equalized_image[:, :, color_band]))

    return equalized_image.astype(np.uint8)

def main(image_path, fov_mask_path):
    image = io.imread(image_path)
    fov_mask = io.imread(fov_mask_path)
    equalized_image = equalize_fundus_image_intensities(image, fov_mask)

    io.imsave('equalized_image.png', equalized_image)

import sys

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
