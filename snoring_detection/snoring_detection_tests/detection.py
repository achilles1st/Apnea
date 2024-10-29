import numpy as np
import tensorflow as tf
from collections import Counter
import keras


# Define constants
AUDIO_FILE = 'C:/Users/tosic/Arduino_projects/sensor_com/old_files/recorded_audio_mono.wav'  # Replace with the actual path to your audio file
MODEL_PATH = './models/cnn_new_big_data.keras'  # Replace with the path to your trained model

# Load the trained model
model = keras.models.load_model(MODEL_PATH)

# Initialize the MFCCProcessor with dummy directories, as we use only its method directly

def get_log_mel(waveform):
  stfts = tf.signal.stft(
      waveform, frame_length=512, frame_step=256)
  # Obtain the magnitude of the STFT.
  spectrograms = tf.abs(stfts)
  # Warp the linear scale spectrograms into the mel-scale.
  num_spectrogram_bins = stfts.shape[-1]
  lower_edge_hertz, upper_edge_hertz, num_mel_bins = 40.0, 6000.0, 30
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins, num_spectrogram_bins, 16000, lower_edge_hertz,
    upper_edge_hertz)
  mel_spectrograms = tf.tensordot(
    spectrograms, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
    linear_to_mel_weight_matrix.shape[-1:]))

  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
  log_mel_spectrograms = tf.reshape(log_mel_spectrograms, [-1,1830])
  return log_mel_spectrograms

def classify_snoring(x):
    """Classifies snoring events in the given audio file."""

    x = tf.io.read_file(str(x))
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000, )
    x = tf.squeeze(x, axis=-1)
    waveform = x
    x = get_log_mel(x)
    prediction = model(x)
    probability = prediction[0, 0]
    #print(probability)


    return probability

if __name__ == "__main__":
    mfccs = []
    for i in range(0, 700):  # 250-300 a lot of errors
        #AUDIO_FILE = f'C:/Users/tosic/Arduino_projects/DatasetSonring/dataset_44100/0/0_{i}.wav'
        #AUDIO_FILE = f'C:/Users/tosic/Arduino_projects/sensor_com/snoring_detection/Snoring_Dataset_@16000/44100_additions/1/1_{i}.wav'

        AUDIO_FILE = f'C:/Users/tosic/Arduino_projects/sensor_com/snoring_detection/Snoring_Dataset_@16000/dataset_16/additions/1/1_{i}.wav'

        probability = classify_snoring(AUDIO_FILE)
        if probability >= 0.5:
            mfccs.append(1)
        else:
            mfccs.append(0)


    array = np.array(mfccs)
    indices_of_ones = np.where(array == 0)[0]
    print(indices_of_ones)

    #flat_array = [item for sublist in array for item in sublist]

    counter = Counter(array)

    print(f"Number of 1s: {counter[1]}")
    print(f"Number of 0s: {counter[0]}")