from skimage import io, color, measure
from scipy import ndimage, stats
from numpy import nan, unique

def get_fov_mask(image_rgb, threshold):
    illuminant = "D50" # default illuminant value from matlab implementation

    # format: [H, W, #channels]
    image_lab = color.rgb2lab(image_rgb)

    image_lab[:, :, 0] /= 100.0

    mask = image_lab[:, :, 0] >= threshold

    # fill holes in the mask
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.filters.median_filter(mask, size=(5, 5))

    # get connected components
    connected_components = measure.label(mask).astype(float)

    # replace background found in [0][0] to nan so mode skips it
    connected_components[connected_components == mask[0][0]] = nan

    # get largest connected component (== mode of the image)
    largest_component_label = stats.mode(connected_components, axis=None, nan_policy='omit')[0]

    # use the modal value of the labels as the final mask
    mask = connected_components == largest_component_label

    # check if the resulting image is all 0s or all 1s
    minimum_threshold = 80
    while len(unique(mask))==1:
        minimum_threshold = minimum_threshold - 20
        mask = (image_rgb[:, :, 0] + image_rgb[:, :, 1] + image_rgb[:, :, 2]) > minimum_threshold

    return mask.astype(float)

def main(image_path, threshold):
    image = io.imread(image_path)
    mask = get_fov_mask(image, threshold)

    io.imsave('mask.png', mask)

import sys

if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]))
