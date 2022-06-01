
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models

from models import *

from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ModelCheckpoint
from hyperopt import hp

import pickle
from scipy import stats, special, spatial
import importlib.util
import os

import warnings
warnings.filterwarnings('ignore')

def load_data(fname):
    # X is multi-variable array
    # Y contains single variable - fix shape for Keras

    npzfile = np.load(fname)
    Xh_train = npzfile['arr_0']
    Xh_test = npzfile['arr_1']
    Xv_train = npzfile['arr_2']
    Xv_test = npzfile['arr_3']
    Y_train = npzfile['arr_4']
    Y_test = npzfile['arr_5']

    X_train = list()
    X_train.append(Xh_train)
    X_train.append(Xv_train)
    X_test = list()
    X_test.append(Xh_test)
    X_test.append(Xv_test)

    Y_train = Y_train.astype(np.float32).reshape((-1,1))
    Y_test = Y_test.astype(np.float32).reshape((-1,1))

    return X_train, X_test, Y_train, Y_test


def Params():
    params = {
        'beta' : 1e-3,
        'kernel_size1': 3,
        'filters1': 5,
        'dilation1': 1,
        'pool_size1': 2,
        'stride1': 1,
        'dropout1': 0.5,
        'units1'  : 20,
        'kernel_size2': 30,
        'filters2': 32,
        'dilation2': 1,
        'pool_size2': 1,
        'stride2': 1,
        'dropout2': 0.5,
        'kernel_size3': 30,
        'filters3': 64,
        'dilation3': 4,
        'pool_size3': 1,
        'stride3': 1,
        'dropout3': 0.5,
        'dense5': 128,
        'dropout5': 0.5,
        'dense6': 50,
        'dropout6': 0.5
    }
    return params


if __name__ == "__main__":

    print("Reading data...")

    fname_data1 = 'data/test_data.npz'
    X_train, X_test, y_train, y_test = load_data(fname_data1)
    X_train = X_train[0]
    X_test  = X_test[0]

    print(X_train.shape)

    p = Params()

    shapes = X_train.shape[1:]
    model = key_small_model(shapes , p)
    
    #mcp = ModelCheckpoint('results/best_promoter_conv_small.model', save_best_only=True)

    history = model.fit(X_train, y_train, epochs=150, 
                validation_data=(X_test, y_test), batch_size=1024)
