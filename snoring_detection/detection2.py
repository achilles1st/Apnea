import os
import numpy as np
import tensorflow as tf
from datetime import timedelta
from preprocessing import MFCCProcessor  # Importing the MFCCProcessor class
import scipy.io.wavfile as wav
from sklearn.preprocessing import StandardScaler
from collections import Counter
import librosa


# Define constants
# AUDIO_FILE = 'C:/Users/tosic/Arduino_projects/sensor_com/old_files/snoring_16k.wav'  # Replace with the actual path to your audio file
MODEL_PATH = './models/cnn_new_big_data.keras'  # Replace with the path to your trained model
SEGMENT_DURATION = 1  # in seconds
THRESHOLD = 0.5  # Probability threshold for snoring classification
MIN_SNORE_SOUNDS = 1  # Minimum snore sounds in a 6-second window to confirm snoring
WINDOW_SIZE = 6  # Sliding window size in seconds

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize the MFCCProcessor with dummy directories, as we use only its method directly
processor = MFCCProcessor(directory='', save_directory='', save_img=False)

def get_log_mel(waveform, sample_rate):
    stfts = tf.signal.stft(
        waveform, frame_length=512, frame_step=256)
    # Obtain the magnitude of the STFT.
    spectrograms = tf.abs(stfts)
    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 40.0, 6000.0, 30
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    log_mel_spectrograms = tf.reshape(log_mel_spectrograms, [-1, 1830])
    return log_mel_spectrograms

def classify_snoring_segment(segment, sample_rate):
    """Classifies snoring events in the given audio segment."""
    # Convert segment to tensor and normalize
    x = tf.convert_to_tensor(segment, dtype=tf.float32)
    x = x / tf.reduce_max(tf.abs(x))  # Normalize between -1 and 1

    # Ensure x is 1D
    x = tf.reshape(x, [-1])

    # Get log-mel spectrogram
    x = get_log_mel(x, sample_rate)

    # Make prediction
    prediction = model(x)
    probability = prediction[0, 0].numpy()
    return probability

if __name__ == "__main__":
    AUDIO_FILE = 'C:/Users/tosic/Arduino_projects/sensor_com/old_files/recorded_strong_breathing.wav'  # Replace with the actual path to your audio file

    OUTPUT_FILE = 'C:/Users/tosic/Arduino_projects/sensor_com/old_files/recorded_breathing_16k.wav'  # Replace with the path to save the resampled audio
    TARGET_SAMPLE_RATE = 16000  # Target sample rate for resampling
    # Load the original audio file
    audio_data, original_sample_rate = librosa.load(AUDIO_FILE, sr=None)  # Load with original sample rate

    # Check if resampling is needed
    if original_sample_rate != TARGET_SAMPLE_RATE:
        # Resample the audio to the target sample rate
        audio_data = librosa.resample(audio_data, orig_sr=original_sample_rate, target_sr=TARGET_SAMPLE_RATE)
        print(f"Resampled from {original_sample_rate} Hz to {TARGET_SAMPLE_RATE} Hz.")

        # Save the resampled audio to the output file
        wav.write(OUTPUT_FILE, TARGET_SAMPLE_RATE, audio_data)
        print(f"Resampled audio saved to {OUTPUT_FILE}")
        AUDIO_FILE = OUTPUT_FILE
    else:
        print(f"Audio is already at {TARGET_SAMPLE_RATE} Hz.")

    # Read the longer audio file
    sample_rate, data = wav.read(AUDIO_FILE)  # data is a numpy array

    # Ensure data is mono
    if data.ndim > 1:
        data = data[:, 0]  # Take first channel

    # Calculate number of samples per segment
    samples_per_segment = int(sample_rate * SEGMENT_DURATION)

    # Number of segments
    total_samples = data.shape[0]
    num_segments = total_samples // samples_per_segment

    features = []

    for i in range(num_segments):
        start_sample = i * samples_per_segment
        end_sample = start_sample + samples_per_segment
        segment = data[start_sample:end_sample]

        # Skip if segment is shorter than expected (can happen at the end)
        if len(segment) < samples_per_segment:
            continue

        # Classify the segment
        probability = classify_snoring_segment(segment, sample_rate)
        if probability >= THRESHOLD:
            features.append(1)
        else:
            features.append(0)

    # Analyze the results
    array = np.array(features)
    indices_of_snoring = np.where(array == 1)[0]
    print(f"Indices of snoring segments: {indices_of_snoring}")

    counter = Counter(array)
    print(f"Number of snoring segments (1s): {counter[1]}")
    print(f"Number of non-snoring segments (0s): {counter[0]}")
