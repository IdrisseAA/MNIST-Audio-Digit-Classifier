from tqdm import tqdm
import numpy as np
import librosa
import os


def pad_to_consistent_shape(arr, target_shape):

    """
    Pads a NumPy array 'arr' to have the specified 'target_shape'

    :param arr: actual feature array
    :param target_shape: Target shape

    :return: padded_arr: Padded NumPy array
    """

    current_shape = arr.shape
    pad_height = target_shape[0] - current_shape[0]
    pad_width = target_shape[1] - current_shape[1]

    # Ensure that padding values are non-negative
    pad_height = max(0, pad_height)
    pad_width = max(0, pad_width)

    padded_arr = np.pad(arr, ((0, pad_height), (0, pad_width)), mode='constant')

    return padded_arr


def extract_audio_features(processed_dataset_folder, num_mels, digits):
    """
    Extracts audio features from the processed dataset

    :param processed_dataset_folder: Path to the processed dataset
    :param num_mels: Number of Mel frequency bins
    :param digits: List of digits to extract features for

    :return: dataset (dict): The extracted log mel spectrograms that form the dataset composed of (X, y).
    """

    labels = []
    features = []

    for digit_folder_name in os.listdir(processed_dataset_folder):
        digit_folder_path = os.path.join(processed_dataset_folder, digit_folder_name)

        if os.path.isdir(digit_folder_path) and digit_folder_name[0] in digits:
            files = os.listdir(digit_folder_path)

            for file in tqdm(files, desc=f'Extracting features in folder {digit_folder_name[0]}'):
                if os.path.isfile(os.path.join(digit_folder_path, file)):
                    audio_file = os.path.join(digit_folder_path, file)

                    # Read the audio file
                    y, sr = librosa.load(audio_file)

                    # Compute log power spectrogram
                    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=num_mels)
                    log_power_spec = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)

                    min_val = np.min(log_power_spec)
                    max_val = np.max(log_power_spec)
                    normalized_log_power_spec = ((log_power_spec - min_val) / (max_val - min_val))

                    features.append(normalized_log_power_spec.tolist())
                    labels.append(digit_folder_name[0])

    # Calculate max_time_steps after processing all the log power spectrograms
    max_time_steps = max(len(spec) for spec in features)
    # print(f'Max time steps: {max_time_steps}')

    # Pad all the log power spectrograms to have the same number of time steps
    target_shape = (num_mels, max_time_steps)
    # print(f'Target shape: {target_shape}')
    padded_features = [pad_to_consistent_shape(np.array(feature), target_shape) for feature in features]
    final_features = [feature.tolist() for feature in padded_features]

    dataset = {"labels": labels,
               "features": final_features}

    return dataset


def prepare_audio_features_for_prediction(audio_file_path):
    """
    This function prepare the audio features for prediction

    :param audio_file_path: Path to the audio to be predicted

    :return: final_features: Audio features of the audio file
    """

    # Read the audio file
    y, sr = librosa.load(audio_file_path)

    # Compute log power mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_power_mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)

    # Normalize the log power mel spectrogram
    min_val = np.min(log_power_mel_spectrogram)
    max_val = np.max(log_power_mel_spectrogram)
    normalized_log_power_mel_spec = ((log_power_mel_spectrogram - min_val) / (max_val - min_val))

    padded_log_power_mel_spec = pad_to_consistent_shape(normalized_log_power_mel_spec, target_shape=(128, 128))

    final_features = padded_log_power_mel_spec[np.newaxis, ..., np.newaxis]

    return final_features
