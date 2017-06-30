from skimage import io, color, measure, filters
from scipy import ndimage
import numpy as np

def replace_out_of_fov_pixels(image_rgb, fov_mask):

    # get image size
    image_size = image_rgb.shape

    # for each color band, apply:
    for color_band in range(0, image_size[2]):

        # get current color band
        current_color_band = image_rgb[:, :, color_band]
        # compute the mean value inside the fov mask
        mean_value = (current_color_band[fov_mask>0]).mean()
        # assign this value to all pixels outside the fov
        current_color_band[fov_mask == 0] = mean_value
        # and copy back the color band
        image_rgb[:, :, color_band] = current_color_band

    return image_rgb


def main(image_path, fov_mask_path):
    image = io.imread(image_path)
    fov_mask = io.imread(fov_mask_path)
    output_image = replace_out_of_fov_pixels(image, fov_mask)

    io.imsave('output_image.png', output_image.astype(np.uint8))

import sys

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
