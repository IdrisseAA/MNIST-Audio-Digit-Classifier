from keras import Input
from keras.models import Sequential
from keras.layers import (Conv2D, LSTM, Conv1D, MaxPooling1D, MaxPool2D, Flatten, Dense, Dropout,
                          TimeDistributed, InputLayer, Concatenate, Reshape)
from keras.models import Model


def cnn_model(input_shape, num_classes):
    """
    Build a 2D CNN model

    :param input_shape: The shape of the input audio features
    :param num_classes: The number of classes to classify

    :return: model: The CNN model
    """

    model = Sequential()

    # Block 1: Convolutional Layer, Max Pooling, and Dropout
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Block 2: Convolutional Layer, Max Pooling, and Dropout
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Block 3: Convolutional Layer, Max Pooling, and Dropout
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Block 4: Convolutional Layer, Max Pooling, and Dropout
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Block 5: Convolutional Layer, Max Pooling, and Dropout
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Flatten and add fully connected layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def lstm_model(input_shape, num_classes):
    """
    Build a LSTM model

    :param input_shape: The shape of the input audio features
    :param num_classes: The number of classes to classify

    :return: model: The LSTM model
    """

    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    # Flatten and add fully connected layers
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def conv1d_model(input_shape, num_classes):
    """
    Build a 1D CNN model

    :param input_shape: The shape of the input audio features
    :param num_classes: The number of classes to classify

    :return: model: The 1D CNN model
    """

    model = Sequential()
    # Convolutional Layer 1
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    # Convolutional Layer 2
    model.add(Conv1D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    # Convolutional Layer 3
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    # Convolutional Layer 4
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    # Convolutional Layer 6
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    # Flatten and add fully connected layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def hybrid_model(input_shape, num_classes):
    """
    Build a hybrid (CNN + LSTM + CON1D) build

    :param input_shape: The shape of the input audio features
    :param num_classes: The number of classes to classify

    :return: hybrid_model: The hybrid (CNN + LSTM + CON1D) model
    """

    input_layer = Input(shape=input_shape, name='Input Layer')

    # CNN block
    # Block 1: Convolutional Layer, Max Pooling, and Dropout
    cnn_model = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    cnn_model = MaxPool2D(pool_size=(2, 2))(cnn_model)
    cnn_model = Dropout(0.2)(cnn_model)
    # Block 2: Convolutional Layer, Max Pooling, and Dropout
    cnn_model = Conv2D(64, kernel_size=(3, 3), activation='relu')(cnn_model)
    cnn_model = MaxPool2D(pool_size=(2, 2))(cnn_model)
    cnn_model = Dropout(0.2)(cnn_model)
    # Block 3: Convolutional Layer, Max Pooling, and Dropout
    cnn_model = Conv2D(128, kernel_size=(3, 3), activation='relu')(cnn_model)
    cnn_model = MaxPool2D(pool_size=(2, 2))(cnn_model)
    cnn_model = Dropout(0.2)(cnn_model)
    # Block 4: Convolutional Layer, Max Pooling, and Dropout
    cnn_model = Conv2D(256, kernel_size=(3, 3), activation='relu')(cnn_model)
    cnn_model = MaxPool2D(pool_size=(2, 2))(cnn_model)
    cnn_model = Dropout(0.2)(cnn_model)
    # Block 5: Convolutional Layer, Max Pooling, and Dropout
    cnn_model = Conv2D(512, kernel_size=(3, 3), activation='relu')(cnn_model)
    cnn_model = MaxPool2D(pool_size=(2, 2))(cnn_model)
    cnn_model = Dropout(0.2)(cnn_model)
    # Flatten and add fully connected layers
    cnn_model = Flatten()(cnn_model)

    # lstm block
    lstm_model = Reshape(target_shape=(128, 128))(input_layer)
    lstm_model = LSTM(units=128, return_sequences=True)(lstm_model)
    lstm_model = LSTM(units=128, return_sequences=True)(lstm_model)
    lstm_model = Dropout(0.4)(lstm_model)
    lstm_model = TimeDistributed(Dense(64, activation='relu'))(lstm_model)
    lstm_model = TimeDistributed(Dense(32, activation='relu'))(lstm_model)
    lstm_model = TimeDistributed(Dense(16, activation='relu'))(lstm_model)
    lstm_model = TimeDistributed(Dense(8, activation='relu'))(lstm_model)
    # Flatten and add fully connected layers
    lstm_model = Flatten()(lstm_model)

    # con1d block
    conv1d_model = Reshape(target_shape=(128, 128))(input_layer)
    # Convolutional Layer 1
    conv1d_model = Conv1D(32, kernel_size=3, activation='relu')(conv1d_model)
    conv1d_model = MaxPooling1D(pool_size=2)(conv1d_model)
    conv1d_model = Dropout(0.2)((conv1d_model))
    # Convolutional Layer 2
    conv1d_model = Conv1D(32, kernel_size=3, activation='relu')(conv1d_model)
    conv1d_model = MaxPooling1D(pool_size=2)(conv1d_model)
    conv1d_model = Dropout(0.2)(conv1d_model)
    # Convolutional Layer 3
    conv1d_model = Conv1D(64, kernel_size=3, activation='relu')(conv1d_model)
    conv1d_model = MaxPooling1D(pool_size=2)(conv1d_model)
    conv1d_model = Dropout(0.2)(conv1d_model)
    # Convolutional Layer 4
    conv1d_model = Conv1D(64, kernel_size=3, activation='relu')(conv1d_model)
    conv1d_model = MaxPooling1D(pool_size=2)(conv1d_model)
    conv1d_model = Dropout(0.2)(conv1d_model)
    # Convolutional Layer 6
    conv1d_model = Conv1D(128, kernel_size=3, activation='relu')(conv1d_model)
    conv1d_model = MaxPooling1D(pool_size=2)(conv1d_model)
    conv1d_model = Dropout(0.2)(conv1d_model)
    # Flatten and add fully connected layers
    conv1d_model = Flatten()(conv1d_model)

    # Merge the outputs from the three models
    merged = Concatenate()([cnn_model, lstm_model, conv1d_model])
    merged = Flatten()(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = Dense(num_classes, activation='softmax')(merged)

    hybrid_model = Model(inputs=input_layer, outputs=merged)
    # Compile the hybrid model
    hybrid_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return hybrid_model
