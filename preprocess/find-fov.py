from skimage import io, color
from scipy import ndimage

def get_fov_mask(image_rgb, threshold):
    illuminant = "D50" # default illuminant value from matlab implementation

    # format: [H, W, #channels]
    image_lab = color.rgb2lab(image_rgb, illuminant)

    image_lab[:, :, 0] /= 100.0

    mask = image_lab[:, :, 0] >= threshold

    # fill holes in the mask
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.filters.median_filter(mask, size=(5,5))

    return mask

def main(image_path, threshold):
    image = io.imread(image_path)
    mask = get_fov_mask(image, threshold).astype(float)

    io.imsave('mask.png', mask)

import sys

if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]))