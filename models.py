from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, Embedding, LSTM, Activation
from keras.models import Sequential
from keras.regularizers import l2
from keras import optimizers

def build_conv(kernel_len1, kernel_len2):
    model = Sequential()
    #print model.output_shape
    model.add(Conv1D(30, kernel_len1, input_shape=(147, 4), activation='relu'))
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
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    return model