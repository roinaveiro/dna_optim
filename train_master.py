import numpy as np
import pandas as pd

import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint

from models import *

from sklearn.model_selection import train_test_split

import pickle


def load_data(fname, stdz = False, filter = False):
    # X is multi-variable array
    # Y contains single variable - fix shape for Keras

    npzfile = np.load(fname)

    X = npzfile['full_seq']
    stats = npzfile['statistics']
    y = stats[:,0]

    if stdz:
        y = ( y - np.mean(y) ) / np.std(y)

    ## Filter
    if filter:
        indices = stats[:,2] < stats[:,0]
        y = y[indices]
        X = X[indices]

    
    print(X.shape)
    print(y.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                        test_size=0.1, random_state=1)
    
    return X_train, X_val, y_train, y_val


def Params():
    params = {
        'beta' : 1e-3,
        'kernel_size1': 40,
        'filters1': 128,
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
        'dense6': 32,
        'dropout6': 0.5
    }
    return params


if __name__ == "__main__":

    fname_data = 'data/data_atted/preprocessed/preprocessed.npz'
    X_train, X_val, y_train, y_val = load_data(fname_data)

    print("Training...")

    res_path = 'results/atted/'

    with open(res_path + 'X_val.pkl','wb') as f:
        pickle.dump(X_val, f)

    with open(res_path + 'y_val.pkl','wb') as f:
        pickle.dump(y_val, f)


    p = Params()
    model = key_small_model(X_train.shape[1:], p)
    mcp = ModelCheckpoint(res_path + 'best_full_conv.model', save_best_only=True)
    history = model.fit(X_train, y_train, epochs=150, callbacks=[mcp], 
                validation_data=(X_val, y_val), batch_size=1024)

