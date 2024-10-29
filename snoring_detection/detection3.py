import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np
from preprocessing import MFCCProcessor


def segment_audio(signal, fs, frame_length, frame_stride):
    """
    Splits the audio signal into frames.

    :param signal: The audio signal array.
    :param fs: Sampling frequency.
    :param frame_length: Frame length in seconds.
    :param frame_stride: Frame stride in seconds.
    :return: Array of frames.
    """
    frame_size = int(frame_length * fs)
    frame_step = int(frame_stride * fs)
    signal_length = len(signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_size)) / frame_step)) + 1

    pad_signal_length = num_frames * frame_step + frame_size
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)

    indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_size, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

def extract_features_from_frames(frames, fs):
    """
    Extracts MFCC features from audio frames.

    :param frames: Array of audio frames.
    :param fs: Sampling frequency.
    :return: 4D numpy array of features suitable for CNN input.
    """
    features = []
    for frame in frames:
        # Apply MFCC extraction
        mfcc_features = processor.apply_mfcc(fs, frame)
        # Normalize features as per training (we'll address this later)
        features.append(mfcc_features)
    features = np.array(features)
    # Reshape features to match input shape expected by the model
    features = features[..., np.newaxis]  # Add channel dimension if necessary
    return features

# Load the trained model
model = tf.keras.models.load_model('models/cnn.keras')
processor = MFCCProcessor(directory='', save_directory='', save_img=False)

