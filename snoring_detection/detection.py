import os
import numpy as np
import tensorflow as tf
from datetime import timedelta
from preprocessing import MFCCProcessor  # Importing the MFCCProcessor class
import scipy.io.wavfile as wav
from sklearn.preprocessing import StandardScaler
from collections import Counter


# Define constants
AUDIO_FILE = 'C:/Users/tosic/Arduino_projects/sensor_com/old_files/recorded_audio_mono.wav'  # Replace with the actual path to your audio file
MODEL_PATH = 'C:/Users/tosics/repos/Apnea/snoring_detection/models/cnn.keras'  # Replace with the path to your trained model
SEGMENT_DURATION = 1  # in seconds
SAMPLE_RATE = 48000  # sampling rate of audio
THRESHOLD = 0.5  # Probability threshold for snoring classification
MIN_SNORE_SOUNDS = 1  # Minimum snore sounds in a 6-second window to confirm snoring
WINDOW_SIZE = 6  # Sliding window size in seconds

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize the MFCCProcessor with dummy directories, as we use only its method directly
processor = MFCCProcessor(directory='', save_directory='', save_img=False)


def classify_snoring(audio_file):
    """Classifies snoring events in the given audio file."""
    # Load the entire audio file
    fs, audio = wav.read(audio_file)
    if audio.ndim != 1:
        audio = audio[:, 0]  # Ensure mono channel

    if fs != 16000:
        print('Audio file must be 16kHz')

    mfcc = processor.apply_mfcc(fs, audio)


    # scaler = StandardScaler()
    # mfcc = scaler.fit_transform(mfcc)

    # mfcc = np.expand_dims(mfcc, axis=(0, -1))
    return mfcc
    # Predict snoring probability
    # prediction = model.predict(mfcc)[0][0]
    # print((prediction > 0.5).astype(int))
    # duration = len(audio) / fs
    # snore_timestamps = []
    #
    # # Process the audio in segments
    # for i in range(0, int(duration), SEGMENT_DURATION):
    #     start = i * fs
    #     end = start + SEGMENT_DURATION * fs
    #     if end > len(audio):
    #         break
    #
    #     segment = audio[start:end]
    #     mfcc = processor.apply_mfcc(fs, segment)  # Use MFCCProcessor's method for extraction
    #     mfcc = np.expand_dims(mfcc, axis=(0, -1))  # Add batch and channel dimensions for prediction

        # Predict snoring probability
    #    prediction = model.predict(mfcc)[0][0]
        # is_snore = prediction >= THRESHOLD
        #
        # # Use a sliding window to confirm snoring
        # if len(snore_timestamps) >= WINDOW_SIZE:
        #     snore_timestamps.pop(0)  # Maintain a fixed window size
        # snore_timestamps.append(is_snore)
        #
        # # Confirm snoring if at least 2 out of 6 are snoring sounds
        # if snore_timestamps.count(True) >= MIN_SNORE_SOUNDS:
        #     print(f"Snoring detected at {timedelta(seconds=i)}")
        #     # Log the time or perform other actions as needed


if __name__ == "__main__":
    mfccs = []
    for i in range(0, 500):  # 250-300 a lot of errors
        AUDIO_FILE = f'C:/Users/tosics/repos/Apnea/snoring_detection/data16/1/1_{i}.wav'
        mfcc = classify_snoring(AUDIO_FILE)
        mfccs.append(mfcc)

    mfccs_array = np.array(mfccs)
    x_mean = np.mean(mfccs_array, axis=0)
    x_std = np.std(mfccs_array, axis=0)
    mfccs_array_norm = (mfccs_array - x_mean) / x_std

    mfccs_array_norm = np.expand_dims(mfccs_array_norm, axis=-1)
    prediction = model.predict(mfccs_array_norm)
    #print(prediction)

    array = (prediction > 0.2).astype(int)
    indices_of_ones = np.where(array == 0)[0]
    print(indices_of_ones)

    flat_array = [item for sublist in array for item in sublist]

    counter = Counter(flat_array)

    print(f"Number of 1s: {counter[1]}")
    print(f"Number of 0s: {counter[0]}")