import sys
from os import path, makedirs
# Import from sibling directory ..\api
sys.path.append(path.dirname(path.abspath(__file__)) + "/..")
from core.metrics.custom_binary_metrics import precision, recall, f1

from core.models import vgg16
from core.training import trainer
from keras.optimizers import SGD
from core.augmentation import data_augmentation, no_augmentation


def test(input_data_path, output_path, image_shape, batch_size):
    model = vgg16.build((310, 310))

    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)

    trainer.train(
        model = model,
        loss = 'binary_crossentropy',
        metrics = [precision, recall, f1],
        optimizer = sgd,
        data_augmentation_policy = no_augmentation,
        input_data_path = input_data_path,
        output_path = output_path,
        image_shape = image_shape,
        batch_size = batch_size)


def usage():
    print('ERROR: Usage: test.py <data_path> <output_path> [--image_shape] [--batch_size]')

import argparse
import sys

if __name__ == '__main__':

    if len(sys.argv) < 3:
        usage()
        exit()
    else:
        # create an argument parser to control the input parameters
        parser = argparse.ArgumentParser()
        parser.add_argument("data_path", help="directory with training/validation directories", type=str)
        parser.add_argument("output_path", help="directory to save the models", type=str)
        parser.add_argument("-ih", "--image_height", help="input image height", type=int, default=512)
        parser.add_argument("-iw", "--image_width", help="input image width", type=int, default=512)
        parser.add_argument("-ic", "--channels", help="input image channels", type=int, default=3)
        parser.add_argument("-b", "--batch_size", help="class threshold", type=int, default=32)
        args = parser.parse_args()

        # call the main function
        test(args.data_path, args.output_path, (args.image_height, args.image_width, args.channels), args.batch_size)
