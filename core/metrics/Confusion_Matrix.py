
import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback
import numpy as np

class Confusion_Matrix(Callback):

    def on_epoch_end(self, epoch, logs={}):

        #print(logs)
        tp = logs["val_tp"]
        fp = logs["val_fp"]
        fn = logs["val_fn"]
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2*((precision * recall)/(precision + recall + 1e-9))

        print('\n#### val precision {0:1.4f} - val recall {1:1.4f} - val f1 {2:1.4f}'.format(precision, recall, f1) )
