from models import cnn_model, lstm_model, conv1d_model, hybrid_model
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np


def split_and_prepare_data(dataset, test_size=0.1, val_size=0.2, random_state=42, model_type='cnn'):
    """
    Reshape the dataset according to the specified model type, and split it into training, validation, and testing sets.

    :param dataset: The dataset of audio features
    :param test_size: Percentage of data for the testing set (default: 10%).
    :param val_size: Percentage of data for the validation set (default: 20% of the remaining data after testing set).
    :param random_state: Random seed for reproducibility (default: 42).
    :param model_type: Type of model ('cnn', 'lstm', or 'conv1d').

    :return: X_train, X_val, X_test, y_train, y_val, y_test
    """

    # Access the labels and features
    labels = np.array(dataset['labels'])
    features = np.array(dataset['features'])

    # Reshape the data based on the model type
    if model_type in ['cnn', 'hybrid']:
        height, width = features.shape[1], features.shape[2]
        features_reshaped = features.reshape(features.shape[0], height, width, 1)
    elif model_type in ['lstm', 'conv1d']:
        time_steps, num_features = features.shape[1], features.shape[2]
        features_reshaped = features.reshape(features.shape[0], time_steps, num_features)

    else:
        raise ValueError("\nInvalid model_type. Use 'cnn', 'lstm', or 'conv1d'.")

    # One-hot encode the labels
    num_classes = len(np.unique(labels))
    labels_encoded = to_categorical(labels, num_classes)

    # Split the data into training, validation, and testing sets
    x_train, x_temp, y_train, y_temp = train_test_split(features_reshaped, labels_encoded, test_size=val_size,
                                                        random_state=random_state, stratify=labels_encoded)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=test_size, random_state=random_state,
                                                    stratify=y_temp)

    print("\nThe data was split, reshaped successfully and ready to be forwarded to the model.")

    return x_train, x_val, x_test, y_train, y_val, y_test


def model_training(x_train, x_val, y_train, y_val, model_type=None, num_classes=None, num_epochs=20, batch_size=32):
    """
    Train a deep learning model with the specified architecture and hyperparameters.

    :param x_train: training dataset features
    :param x_val: validation dataset features
    :param y_train: training dataset labels
    :param y_val: validation dataset labels
    :param model_type: Type of model to train ('cnn', 'lstm', or 'conv1d').
    :param num_classes: Number of output classes.
    :param num_epochs: Number of training epochs.
    :param batch_size: Batch size for training.

    :return: Training history (history) and the trained model (model).
    """

    # Prepare the input shape of the model
    input_shape = x_train.shape[1:]
    # print(input_shape)

    # Create and prepare the model based on the model type
    if model_type == 'cnn':
        model = cnn_model(input_shape, num_classes)
    elif model_type == 'lstm':
        model = lstm_model(input_shape, num_classes)
    elif model_type == 'conv1d':
        model = conv1d_model(input_shape, num_classes)
    elif model_type == 'hybrid':
        model = hybrid_model(input_shape, num_classes)
    else:
        raise ValueError("\nInvalid model_type. Use 'cnn', 'lstm', or 'conv1d'.")

    # Train the model
    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=num_epochs,
                        batch_size=batch_size)

    print(f"\n{model_type} model completed the training successfully.")

    return history, model
