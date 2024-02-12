import random
import shutil
import os


def get_random_digits_audio_to_plot(processed_dataset_folder, digits):

    """
    This function will return 10 random samples (path) from the digit folders, one digit per folder, randomly.

    :param processed_dataset_folder: Path of the arranged or processed dataset,  to get the data from.
    :param digits: the list of digits we want to retrieve

    :return: data_sample, num_rows, num_cols

    """

    data_sample = {}
    for folder_name in os.listdir(processed_dataset_folder):
        folder_path = os.path.join(processed_dataset_folder, folder_name)

        if os.path.isdir(folder_path) and folder_name[0] in digits:
            files = os.listdir(folder_path)
            files = [file for file in files if os.path.isfile(os.path.join(folder_path, file))]

            if files:
                random_file = random.choice(files)
                random_file_path = os.path.join(folder_path, random_file)
                data_sample[folder_name] = random_file_path

    num_samples = len(data_sample)
    num_rows = (num_samples + 1) // 2
    num_cols = 2

    return data_sample, num_rows, num_cols


def create_directories(processed_dataset_folder, production_dataset_folder, reports_folder, models_folder):

    """
    This function will create the necessary directories

    :param processed_dataset_folder: Path of the arranged or processed dataset,  to get the data from
    :param production_dataset_folder: Path of the production dataset
    :param reports_folder: Path of the reports folder to save the generated images and files
    :param models_folder: Path of the models folder to save the different models

    :return: None
    """

    # Creating the processed dataset folder which will contain the arranged audio data.
    if os.path.exists(processed_dataset_folder):
        shutil.rmtree(processed_dataset_folder)
        os.mkdir(processed_dataset_folder)
    else:
        os.mkdir(processed_dataset_folder)

    # Creating the production dataset folder which will contain some data to use in production.
    if os.path.exists(production_dataset_folder):
        shutil.rmtree(production_dataset_folder)
        os.mkdir(production_dataset_folder)
    else:
        os.mkdir(production_dataset_folder)

    # Creating the folder that will have into the different figures and reports from our application.
    if os.path.exists(reports_folder):
        shutil.rmtree(reports_folder)
        os.mkdir(reports_folder)
    else:
        os.mkdir(reports_folder)

    # Crating the folder to save the different models into it.
    if os.path.exists(models_folder):
        shutil.rmtree(models_folder)
        os.mkdir(models_folder)
    else:
        os.mkdir(models_folder)
    print("Directories are created successfully.")

