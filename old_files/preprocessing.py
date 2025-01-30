import os
import scipy.io.wavfile as wav
from speechpy import processing
import numpy as np
from scipy.fftpack import dct
from PIL import Image as im
import librosa
import tensorflow as tf


class MFCCProcessor:
    def __init__(self, directory, save_directory, save_img=False):
        """
        Initializes the MFCCProcessor with the directory paths and configuration.

        :param directory: Directory where the audio files are located.
        :param save_directory: Directory where the processed data will be saved.
        :param save_img: Boolean flag to indicate whether to save images of the MFCC features.
        """
        self.directory = directory
        self.save_directory = save_directory
        self.save_img = save_img

    def process_files(self):
        """
        Processes all .wav files in the specified directory and its subdirectories.
        Extracts MFCC features and saves them as .npy files, and optionally as images.
        """
        extensions = ["/0", "/1"]
        for extension in extensions:
            for filename in os.listdir(self.directory + extension):
                if filename.endswith(".wav"):
                    fs, signal = wav.read(self.directory + extension + "/" + filename)

                    if fs != 44100:
                        print("Sampling rate is not correct: " + str(fs) + ". Resampling...")


                    # Ensure mono signal
                    if signal.ndim != 1:
                        signal = signal[:, 0]

                    features = self.apply_mfcc(fs, signal)
                    np.save(self.save_directory + extension + '/' + filename[:-4] + 'mfcc' + '.npy', features)

                    # Save as image if enabled
                    if self.save_img:
                        data = im.fromarray(features, "L")
                        data.save(self.save_directory + extension + "/" + filename[:-3] + 'png')

    def apply_mfcc(self, fs, signal):
        """
        Applies MFCC feature extraction on the given signal.

        :param fs: Sampling frequency of the signal.
        :param signal: Audio signal array.
        :return: Array of MFCC features.
        """
        signal = processing.preemphasis(signal, cof=0.98)

        # Stacking frames
        frames = processing.stack_frames(signal, sampling_frequency=fs,
                                         frame_length=0.030,  # Frame size of 30 ms
                                         frame_stride=0.030,  # Frame stride
                                         zero_padding=False)

        # Extracting power spectrum
        power_spectrum = processing.power_spectrum(frames, fft_points=512)

        # Custom filterbanks calculation
        first_10_filterbanks = self.custom_filterbanks(nfilt=10, nfft=512, samplerate=fs, lowfreq=100, highfreq=1000)
        last22_filterbanks = self.custom_filterbanks(nfilt=22, nfft=512, samplerate=fs, lowfreq=1100, highfreq=None)

        # Combine filterbanks
        mel_matrix = np.concatenate((first_10_filterbanks, last22_filterbanks), axis=1).T

        # Compute spectrogram and filterbank energies
        energy = np.sum(power_spectrum, 1)
        energy = np.where(energy == 0, np.finfo(float).eps, energy)

        features = np.dot(power_spectrum, mel_matrix.T)
        features = np.where(features == 0, np.finfo(float).eps, features)
        log_features = np.log(features)

        # DCT
        numcep = 32
        dct_log_features = dct(log_features, type=2, axis=1)[:, :numcep]

        return log_features

    def custom_filterbanks(self, nfilt=10, nfft=512, samplerate=44100, lowfreq=0, highfreq=None):
        """
        Creates custom Mel-filterbanks for MFCC feature extraction.

        :param nfilt: Number of filters in the filterbank.
        :param nfft: FFT size.
        :param samplerate: Sample rate of the signal.
        :param lowfreq: Lowest band edge of Mel filters.
        :param highfreq: Highest band edge of Mel filters.
        :return: A numpy array of filterbanks.
        """
        highfreq = highfreq or samplerate / 2
        remn = 257
        bankpointsnormal = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        bankpoints = np.array(list(map(self.mel, bankpointsnormal)))

        if lowfreq != 100:
            lowerfreq = self.mel(lowfreq)
            highfreq = self.mel(highfreq)
            bankpoints, bankpointsnormal = self.createbankpoints(lowerfreq, highfreq, nfilt)

        bins = np.floor((nfft + 1) * bankpointsnormal / samplerate)
        flbank = np.zeros((remn, len(bins)))
        for i in range(1, len(bins) - 1):
            for j in range(1, remn - 1):
                if bins[i - 1] <= j <= bins[i]:
                    flbank[j][i] = (j - bins[i - 1]) / (bins[i] - bins[i - 1])
                elif bins[i] <= j <= bins[i + 1]:
                    flbank[j][i] = (bins[i + 1] - j) / (bins[i + 1] - bins[i])

        return flbank

    @staticmethod
    def mel(hz):
        """Convert a value in Hertz to Mels."""
        return 2595 * np.log10(1 + hz / 700.)

    @staticmethod
    def createbankpoints(low, high, nfilt):
        """Creates bank points between low and high frequencies for filterbanks."""
        step = (high - low) / (nfilt - 1)
        bankpoints = np.linspace(low, high, nfilt)
        bankpoints_normal = np.array([700 * (10 ** (mel / 2595.0) - 1) for mel in bankpoints])

        return bankpoints, bankpoints_normal


if __name__ == "__main__":
    #direc = 'C:/Users/tosic/Arduino_projects/sensor_com/snoring_detection/Snoring_Dataset_@16000/dataset_16'
    #direc = 'C:/Users/tosic/Arduino_projects/DatasetSonring/dataset_44100'
    direc = 'C:/Users/tosic/Arduino_projects/sensor_com/snoring_detection/Snoring_Dataset_@16000/44100_additions/new_idea'

    save_dir = "./processed_data"
    processor = MFCCProcessor(direc, save_dir, save_img=False)
    processor.process_files()
