import os

import numpy as np
import pandas as pd

def save_preds(file, labels):
    for clip in range(labels.shape[0]):
        file.write("GT,P\n")
        for frame in range(labels.shape[1]):
            np.savetxt(file, labels[clip,frame], newline=", ", fmt='%d')
            file.write('\n')
        file.write("]\n-----\n")

def save_predictions(file, targets, predictions):
    for clip in range(targets.shape[0]):
        file.write("Groundtruth:\n")
        file.write("[")
        np.savetxt(file, targets[clip], newline=", ", fmt='%d')
        file.write("]\n")
        file.write("Prediction:\n")
        file.write("[")
        np.savetxt(file, predictions[clip], newline=", ", fmt='%d')
        file.write("]\n-----\n")