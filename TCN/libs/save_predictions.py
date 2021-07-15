import os

import numpy as np
import pandas as pd

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