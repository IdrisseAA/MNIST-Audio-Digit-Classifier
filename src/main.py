from src.evaluate_model import model_performance_and_assessment
from src.helpers import create_directories, get_random_digits_audio_to_plot
from src.make_dataset import arrange_dataset_into_folders, create_production_data
from src.visualize import plot_audio_features
from src.build_features import extract_audio_features
from src.train_model import split_and_prepare_data, model_training

# Preparing the paths of the directories
RAW_DATA_FOLDER = '../data/raw/free-spoken-digit-dataset'
PROCESSED_DATA_FOLDER = '../data/processed'
PRODUCTION_DATA_FOLDER = '../data/production_data'
REPORT_FOLDER = '../reports'
MODELS_FOLDER = '../models'

DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
MODEL_TYPES = ['hybrid', 'conv1d', 'lstm', 'cnn']

if __name__ == "__main__":
    # Beginning of the execution.
    print("\n-------------------------------START-------------------------------")

    # Call the function to create the directories (for arranged data and reports)
    print("\nLet prepare the necessary folders.")
    create_directories(processed_dataset_folder=PROCESSED_DATA_FOLDER,
                       production_dataset_folder=PRODUCTION_DATA_FOLDER,
                       reports_folder=REPORT_FOLDER,
                       models_folder=MODELS_FOLDER)

    # Call the function to arrange the dataset
    print('\nLet separate the dataset into folders by digits.')
    arrange_dataset_into_folders(raw_dataset_folder=RAW_DATA_FOLDER,
                                 processed_dataset_folder=PROCESSED_DATA_FOLDER)

    # Call the function to create the production dataset
    print('\nLet create the production datasets.')
    create_production_data(processed_dataset_folder=PROCESSED_DATA_FOLDER,
                           production_dataset_folder=PRODUCTION_DATA_FOLDER)

    # Get random digits audio files, and plot some graphical representations of them
    print("\nGet random digits audio files, and plot graphical representations of some audio features.")
    data_sample, num_rows, num_cols = get_random_digits_audio_to_plot(processed_dataset_folder=PROCESSED_DATA_FOLDER,
                                                                      digits=DIGITS)
    plot_audio_features(data_sample, num_rows, num_cols, report_folder=REPORT_FOLDER)

    # Call the function to extract the log mel spectrogram features form the audio files and constitute the dataset.
    print("\nLet extract the log mel spectrogram from the audio and constitute our dataset.")
    dataset = extract_audio_features(processed_dataset_folder=PROCESSED_DATA_FOLDER, num_mels=128, digits=DIGITS)

    # Modeling Section
    for model_type in MODEL_TYPES:
        # Call the function to split the data into train, val and test sets and prepare them for modeling.
        print("\nLet split the data into train, validation, and test sets.")
        X_train, X_val, X_test, y_train, y_val, y_test = split_and_prepare_data(dataset, val_size=0.2, test_size=0.1,
                                                                                model_type=model_type)

        # Call the function model_training to train a model (e.g.  cnn, lstm, conv1d).
        print(f"\nLet train the {model_type.upper()} model.")
        history, model = model_training(x_train=X_train, x_val=X_val, y_train=y_train, y_val=y_val,
                                        model_type=model_type, num_classes=len(DIGITS),
                                        num_epochs=5, batch_size=32)

        # Call the function model to visualise the performance of the model in training and assess the model
        print(f"\n{model_type} model performance in training & Assessment of the model ")
        model_performance_and_assessment(history=history, model=model, x_test=X_test, y_test=y_test, class_names=DIGITS,
                                         model_type=model_type, report_folder=REPORT_FOLDER)

        # Save the model
        model.save(f"{MODELS_FOLDER}/{model_type}.keras")

    # End of the execution.
    print("\n-------------------------------END---------------------------------")
