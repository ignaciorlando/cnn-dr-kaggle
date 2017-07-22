
from skimage import io
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.interpolation import zoom
import numpy as np



def data_augmentation(data_type):

    datagen = ImageDataGenerator(rescale=1./255,)
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
    print("python3 no_augmentation.py <image_path> training|validation|test")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        usage()
        exit()

    main(sys.argv[1], sys.argv[2])
