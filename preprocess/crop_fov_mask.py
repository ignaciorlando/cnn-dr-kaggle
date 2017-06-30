from skimage import io
import numpy as np

def crop_fov_mask(image_rgb, fov_mask):
    rows = np.any(fov_mask, axis=1)
    cols = np.any(fov_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped_fov = fov_mask[rmin:rmax, cmin:cmax]
    cropped_image = image_rgb[rmin:rmax, cmin:cmax, :]

    return cropped_image, cropped_fov

def main(image_path, fov_mask_path):
    image = io.imread(image_path)
    fov_mask = io.imread(fov_mask_path)

    cropped_image, cropped_fov_mask = crop_fov_mask(image, fov_mask)

    io.imsave('cropped_image.png', cropped_image.astype(np.uint8))
    io.imsave('cropped_fov.png', cropped_fov_mask.astype(np.uint8))

import sys

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
