import sys
from os import path, makedirs
# Import from sibling directory ..\api
sys.path.append(path.dirname(path.abspath(__file__)) + "/..")
from core.metrics.custom_binary_metrics import precision, recall, f1

class validation_callback(Callback):
    
    def __init__(self, validation_data):
        self.x_val, self.y_true = validation_data
        self.y_val = self.model.predict(x_val)
        print(len(self.x_val))

    def on_epoch_end(self, epoch, logs={}):
        precision = precision(self.y_true, self.y_val)
        print "MAP"