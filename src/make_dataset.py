import os
import shutil
from tqdm import tqdm
import random


def arrange_dataset_into_folders(raw_dataset_folder, processed_dataset_folder):
    """
    This function will arrange the original dataset into separate folders for each digit.
    Meaning that  folder '0' will contain only audio files corresponding to digit zero etc...

    :param raw_dataset_folder: Path of the origin dataset, to get the data from
    :param processed_dataset_folder: Path of the arranged or processed dataset, to store the new data

    :return: None
    """

    DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for file in tqdm(os.listdir(raw_dataset_folder), desc="Arranging audio files according to digits"):
        file_path = os.path.join(raw_dataset_folder, file)
        file_first_char = file[0]

        if file_first_char in DIGITS:
            digit_folder = os.path.join(processed_dataset_folder, file_first_char)

            if not os.path.exists(digit_folder):
                os.makedirs(digit_folder)

            if os.path.isfile(file_path):
                shutil.copy(file_path, digit_folder)

    print("The raw dataset has been arranged into separate folders based on each digit successfully!")


def create_production_data(processed_dataset_folder, production_dataset_folder):
    """
    This function will get 2 random audio files from each digit folder,
    so we will have a small data to test in production.

    :param processed_dataset_folder: Path of the arranged or processed dataset,  to get the data from.
    :param production_dataset_folder:  Path to store the production dataset.

    :return: None
    """

    for digit_folder in tqdm(os.listdir(processed_dataset_folder), desc="Creating the production dataset"):
        digit_folder_path = os.path.join(processed_dataset_folder, digit_folder)

        if os.path.isdir(digit_folder_path):
            production_digit_folder = os.path.join(production_dataset_folder, digit_folder)

            if not os.path.exists(production_digit_folder):
                os.makedirs(production_digit_folder)

            audio_files = os.listdir(digit_folder_path)
            selected_files = random.sample(audio_files, min(2, (len(audio_files))))

            for selected_file in selected_files:
                file_path = os.path.join(digit_folder_path, selected_file)
                shutil.move(file_path, production_digit_folder)

    print("The production dataset has been created successfully!")
