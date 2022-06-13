import tensorflow as tf

from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, Embedding, LSTM, Activation
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras import optimizers
from keras import backend as K

from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Input, Dense, Flatten, Concatenate


def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def build_conv(in_size, kernel_len1=7, kernel_len2=7):
    model = Sequential()
    #print model.output_shape
    model.add(Conv1D(30, kernel_len1, input_shape=(in_size, 4), activation='relu'))
    #model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    #print model.output_shape
    model.add(Conv1D(60, kernel_len2, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Conv1D(90, kernel_len2, activation='relu'))
    # model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Conv1D(120, kernel_len2, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Conv1D(120, 3, activation='relu'))
    # model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Conv1D(120, 3, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(optimizer='adadelta', loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse', coeff_determination])

    return model
    
def build_conv_no_reg(in_size):
    model = Sequential()
  
    model.add(Conv1D(128, 40, input_shape=(in_size, 4), dilation_rate=1))
    model.add(MaxPooling1D(2) )
    model.add(Conv1D(64, 30, dilation_rate=1))
    model.add(MaxPooling1D(2) )
    model.add(Conv1D(32, 20, dilation_rate=4))
    model.add(MaxPooling1D(2) )
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(32))
    model.add(Dense(1))

    optim = tf.keras.optimizers.Adam(
            learning_rate=0.01)

    model.compile(optimizer=optim, loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse', coeff_determination])

    return model

def build_conv_lstm(in_size, filters=20, kernel_len=7, lstm_hidden_size=100):
    beta = 1e-3
    model = Sequential()
    
    model.add(Conv1D(filters=filters, kernel_size=3, input_shape=(in_size, 4), kernel_regularizer=l2(beta), padding='same'))
    #model.add(Conv1D(filters, kernel_len, input_shape=(151, 4), kernel_regularizer=l2(beta), padding='same'))
    model.add(Activation('relu'))    
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    
    model.add(LSTM(units=20, return_sequences=True, kernel_regularizer=l2(beta)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())    
    #model.add(Dense(1, kernel_regularizer=l2(beta), activation='sigmoid'))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    optim = tf.keras.optimizers.Adam(
            learning_rate=0.003)
    model.compile(optimizer=optim, loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse', coeff_determination])

    return model

def build_small(in_size, filters=5):
    beta = 1e-3
    model = Sequential()
    
    model.add(Conv1D(filters=filters, kernel_size=3, input_shape=(in_size, 4), padding='same'))
    #model.add(Conv1D(filters, kernel_len, input_shape=(151, 4), kernel_regularizer=l2(beta), padding='same'))
    model.add(Activation('relu'))    
    model.add(MaxPooling1D())

    model.add(LSTM(units=20, return_sequences=True, kernel_regularizer=l2(beta)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())    
    model.add(Dense(50, kernel_regularizer=l2(beta), activation='relu'))
    
    model.add(Dense(1))

    optim = tf.keras.optimizers.Adam(
            learning_rate=0.003)
    model.compile(optimizer=optim, loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse', coeff_determination])

    return model


def key_model(shapes, p):

    X_input1 = Input(shape = [2150, 4])
    #X_input2 = Input(shape = shapes[1])

    X = Conv1D(filters=int(p['filters1']),kernel_size=int(p['kernel_size1']),strides=1,dilation_rate=int(p['dilation1']),activation='relu',kernel_initializer='he_uniform')(X_input1)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout1']))(X)
    X = MaxPooling1D(pool_size=int(p['pool_size1']), strides=int(p['stride1']), padding='same')(X)

    X = Conv1D(filters=int(p['filters2']),kernel_size=int(p['kernel_size2']),strides=1,dilation_rate=int(p['dilation2']),padding='same',activation='relu',kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout2']))(X)
    X = MaxPooling1D(pool_size=int(p['pool_size2']), strides=int(p['stride2']), padding='same')(X)
    
    X = Conv1D(filters=int(p['filters3']),kernel_size=int(p['kernel_size3']),strides=1,dilation_rate=int(p['dilation3']),padding='same',activation='relu',kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout3']))(X)
    X = MaxPooling1D(pool_size=int(p['pool_size3']), strides=int(p['stride3']), padding='same')(X)

    X = Flatten()(X)
    
    X = Dense(int(p['dense6']), activation='relu', kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout6']))(X)
    
    X = Dense(1)(X)

    model = Model(inputs = [X_input1], outputs = X)



    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse', coeff_determination])

    return model


def key_small_model(shapes, p):

    X_input1 = Input(shape = shapes)

    X = Conv1D(filters=int(p['filters1']),kernel_size=int(p['kernel_size1']),strides=1,dilation_rate=int(p['dilation1']),activation='relu',kernel_initializer='he_uniform')(X_input1)
    X = BatchNormalization()(X)
    # X = Dropout(float(p['dropout1']))(X)
    X = MaxPooling1D(pool_size=int(p['pool_size1']), strides=int(p['stride1']), padding='same')(X)

    X = LSTM(units=p['units1'], return_sequences=True, kernel_regularizer=l2(p['beta']))(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout2']))(X)

    X = Flatten()(X)

    X = Dense(int(p['dense6']), activation='relu', kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    #X = Dropout(float(p['dropout6']))(X)
    
    X = Dense(1)(X)

    model = Model(inputs = [X_input1], outputs = X)



    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse', coeff_determination])

    return model




