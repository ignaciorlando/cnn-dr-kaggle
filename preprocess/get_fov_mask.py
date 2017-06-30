from skimage import io, color, measure
from scipy import ndimage, stats
from numpy import nan

def get_fov_mask(image_rgb, threshold):
    illuminant = "D50" # default illuminant value from matlab implementation

    # format: [H, W, #channels]
    image_lab = color.rgb2lab(image_rgb, illuminant)

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

    return mask

def main(image_path, threshold):
    image = io.imread(image_path)
    mask = get_fov_mask(image, threshold).astype(bool)

    io.imsave('mask.png', mask)

import sys

if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]))