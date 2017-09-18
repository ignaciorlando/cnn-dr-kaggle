from skimage import io
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.interpolation import zoom
import numpy as np

def zoom_image(image, zoom_range, fill_mode='constant', cval=0.):
    if zoom_range[0] == 1 and zoom_range[1] == 1:
        z = 1
    else:
        z = np.random.uniform(zoom_range[0], zoom_range[1], 1)[0]

    zoomed_image = zoom(image, [z, z, 1], mode=fill_mode, cval=cval)

    target_size = image.shape
    zoomed_size = zoomed_image.shape

    return zoomed_image[int(zoomed_size[0]/2-target_size[0]/2):int(zoomed_size[0]/2+target_size[0]/2),
                        int(zoomed_size[1]/2-target_size[1]/2):int(zoomed_size[1]/2+target_size[1]/2),
                        :]

def data_augmentation(data_type):

    if (data_type=="training"):
        # datagen will be as follows:
        datagen = ImageDataGenerator(
            # For more information about these parameters
            # visit: https://keras.io/preprocessing/image/
            featurewise_center=True,
            featurewise_std_normalization=True,
            #zoom_range=0.5, # up to 25% zoom in
            #fill_mode="constant", cval=0.0, # TODO: try this with "reflect"
            #horizontal_flip=True,
            #vertical_flip=True, # TODO: try this with False
            rescale=1./255#,
            #preprocessing_function=lambda image: zoom_image(image, [1.0, 1.2])
            )
    else: # data_type == "validation" or data_type == "test"
        # datagen will be only rescale
        datagen = ImageDataGenerator(
                rescale=1./255,)

    # return our datagen object
    return datagen

def main(image_path, data_type):
    image = io.imread(image_path)

    image = image.reshape((1,) + image.shape)

    data_generator = data_augmentation(data_type)

    label = [0]

    i = 0
    for batch in data_generator.flow(image, label, batch_size=1, save_to_dir='./', save_prefix='aug_', save_format='png'):
        i += 1
        if i == 20:
            break

import sys

def usage():
    print("python3 data_augmentation.py <image_path> training|validation|test")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        usage()
        exit()

    main(sys.argv[1], sys.argv[2])
