import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt


def plot_digits_waves(data_sample, num_rows, num_cols, report_folder):
    """
    This function will plot the waves of one random digit per folder.

    :param data_sample: one random audio sample from each digit
    :param num_rows: required number of rows in the plot
    :param num_cols: required number of columns in the plot
    :param report_folder: folder to save the generated images

    :return: None
    """

    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))

    for idx, (folder_name, file_path) in enumerate(data_sample.items()):
        y, _ = librosa.load(file_path)

        row = idx // num_cols
        col = idx % num_cols

        axs[row, col].plot(y)
        axs[row, col].set_title(f'Waveform for {folder_name}')
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('Amplitude')

    plt.suptitle('Waveform fo the audio files')
    plt.tight_layout()
    plt.savefig(f"{report_folder}/wave.png")
    plt.show()


def plot_digits_mfccs(data_sample, num_rows, num_cols, report_folder):
    """
    This function will plot the Mel Frequency Cepstral Coefficients (MFCCs) of each random digits.

    :param data_sample: one random audio sample from each digit
    :param num_rows: required number of rows in the subplots
    :param num_cols: required number of columns in the subplots
    :param report_folder: folder to save the generated images

    :return: None
    """

    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))

    for idx, (folder_name, file_path) in enumerate(data_sample.items()):
        y, sr = librosa.load(file_path)

        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr)[1:]

        row = idx // num_cols
        col = idx % num_cols

        # Plot MFCCs
        librosa.display.specshow(mfccs, ax=axs[row, col])
        axs[row, col].set_title(f'MFCCs for {folder_name}')
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('MFCC')

    plt.suptitle('Mel Frequency Cepstral Coefficients of the audio files')
    plt.tight_layout()
    plt.savefig(f"{report_folder}/mfcc.png")
    plt.show()


def plot_digits_spectrogram(data_sample, num_rows, num_cols, report_folder):
    """
    This function will plot the Mel spectrogram of each random digit.

    :param data_sample: one random audio sample from each digit
    :param num_rows: required number of rows in the subplots
    :param num_cols: required number of columns in the subplots
    :param report_folder: folder to save the generated images

    :return: None
    """
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))

    for idx, (folder_name, file_path) in enumerate(data_sample.items()):
        y, sr = librosa.load(file_path)

        # Compute Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

        row = idx // num_cols
        col = idx % num_cols

        # Plot Mel spectrogram
        librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), ax=axs[row, col])
        axs[row, col].set_title(f'Mel Spectrogram for {folder_name}')
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('Frequency')

    plt.suptitle('MEL Spectrogram of the audio files')
    plt.tight_layout()
    plt.savefig(f"{report_folder}/mel_spectrogram.png")
    plt.show()


def plot_digits_rms(data_sample, num_rows, num_cols, report_folder):
    """
    This function will plot the Root Mean Square (RMS) of each random digit.

    :param data_sample: one random audio sample from each digit
    :param num_rows: required number of rows in the subplots
    :param num_cols: required number of columns in the subplots
    :param report_folder: folder to save the generated images

    :return: None
    """
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))

    for idx, (folder_name, file_path) in enumerate(data_sample.items()):
        y, _ = librosa.load(file_path)

        # Compute RMS
        rms = librosa.feature.rms(y=y)

        row = idx // num_cols
        col = idx % num_cols

        # Plot RMS
        axs[row, col].plot(rms[0])
        axs[row, col].set_title(f'RMS for {folder_name}')
        axs[row, col].set_xlabel('Frame')
        axs[row, col].set_ylabel('RMS Value')

    plt.suptitle('Root Mean Square of the audio files')
    plt.tight_layout()
    plt.savefig(f"{report_folder}/rms.png")
    plt.show()


def plot_digits_log_power_spectrogram(data_sample, num_rows, num_cols, report_folder):
    """
    This function will plot the log power spectrogram of each random digit.

    :param data_sample: one random audio sample from each digit
    :param num_rows: required number of rows in the subplots
    :param num_cols: required number of columns in the subplots
    :param report_folder: folder to save the generated images

    :return: None
    """
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))

    for idx, (folder_name, file_path) in enumerate(data_sample.items()):
        y, sr = librosa.load(file_path)

        # Compute log power spectrogram
        log_power_spec = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)

        row = idx // num_cols
        col = idx % num_cols

        # Plot log power spectrogram
        librosa.display.specshow(log_power_spec.T, ax=axs[row, col])
        axs[row, col].set_title(f'Log Power Spectrogram for {folder_name}')
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('Frequency (Mel)')

    plt.suptitle('LOG MEL SPECTROGRAM OF AUDIO FILES')
    plt.tight_layout()
    plt.savefig(f"{report_folder}/log_mel_spectrogram.png")
    plt.show()


def plot_audio_features(data_sample, num_rows, num_cols, report_folder):
    """
       Plot audio features (waveforms, MFCCs, RMS, Mel spectrogram, log power spectrogram) for random digits.

       :param data_sample: Dictionary containing audio file paths for each digit.
       :param num_rows: Number of rows for subplots.
       :param num_cols: Number of columns for subplots.
       :param report_folder: Folder to save the plots

       :return None
       """
    plot_digits_waves(data_sample=data_sample, num_rows=num_rows, num_cols=num_cols, report_folder=report_folder)

    # Call the function that plots the MFCCs
    plot_digits_mfccs(data_sample=data_sample, num_rows=num_rows, num_cols=num_cols, report_folder=report_folder)

    # Call the function that plots the rms
    plot_digits_rms(data_sample=data_sample, num_rows=num_rows, num_cols=num_cols, report_folder=report_folder)

    # Call the function that plots the mel spectrogram
    plot_digits_spectrogram(data_sample=data_sample, num_rows=num_rows, num_cols=num_cols, report_folder=report_folder)

    # Call the function that plots the log mel spectrogram
    plot_digits_log_power_spectrogram(data_sample=data_sample, num_rows=num_rows, num_cols=num_cols,
                                      report_folder=report_folder)
