"""
ECGClassification.py

This implementation was inspired by the research article:
Tao Wang, Changhua Lu, Guohao Shen. "Detection of Sleep Apnea from Single‐Lead ECG Signal Using a Time Window Artificial Neural Network."
BioMed Research International, 2019. DOI: 10.1155/2019/9768072.

Some of the code structures and ideas were adapted from the GitHub repository:
https://github.com/JackAndCole/Detection-of-sleep-apnea-from-single-lead-ECG-signal-using-a-time-window-artificial-neural-network/blob/master/main.py
"""

import numpy as np
import pandas as pd
import joblib
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from hrv.classical import frequency_domain, time_domain
from scipy.signal import medfilt
import biosppy.signals.tools as st
from sklearn.preprocessing import StandardScaler
from Helper import Helper
import configparser
import matplotlib.pyplot as plt


class ECGDetector(Helper):
    """
    ECGClassification provides methods to preprocess ECG data,
    detect beats, extract features, and classify arrhythmias.
    """
    def __init__(self, url, token, org, field, window_size=5, fs=128, model_path="model/mlp_classifier_model_base.pkl"):
        """
        Initialize the ECGClassification class with:
        :param: fs (int): Sampling rate in Hz
        :param: model_path (Any): A trained model or classifier for ECG classification
                       (could be a scikit-learn model, a PyTorch model, etc.)
        :param: window_size (int): The number of past steps to include in each row for classification

        :return: Apnea timestamps detected from the ECG signal.
        """
        self.target_fs = fs
        self.model_path = model_path
        self.window_size = window_size

        # Storage for intermediate processing results
        self.filtered_signal = None
        self.r_peaks = None
        self.features = None
        self.predictions = None
        super().__init__(url, token, org, field, fs)

    def load_model(self):
        """
        Loads the trained classifier model from disk.
        """
        self.model = joblib.load(self.model_path)
        print(f"Model loaded from {self.model_path}.")


    def feature_extraction(self, signal):
        """
        Extracts features from an ECG signal in 1-minute segments.

        Parameters
        ----------
        signal : np.ndarray
            The raw ECG signal (1D array).

        Returns
        -------
        np.ndarray
            A 2D array of shape (num_segments, 18), where each row contains
            a set of extracted features for one 1-minute segment.
        """
        fs = self.target_fs
        hr_min, hr_max = 20, 300
        data = []
        segment_length = 60  # 1-minute segment
        num_segments = len(signal) // (fs * segment_length)

        for i in range(num_segments):
            seg_start = i * fs * segment_length
            seg_end = (i + 1) * fs * segment_length
            segment = signal[seg_start:seg_end]

            # Basic bandpass filter
            segment, _, _ = st.filter_signal(
                segment,
                ftype='FIR',
                band='bandpass',
                order=int(0.3 * fs),
                frequency=[3, 45],
                sampling_rate=fs
            )

            # Find R peaks
            rpeaks, = hamilton_segmenter(segment, sampling_rate=fs)
            rpeaks, = correct_rpeaks(segment, rpeaks, sampling_rate=fs, tol=0.1)

            # Basic constraints on number of beats in 1 min
            if 40 <= len(rpeaks) <= 200:
                rri_tm = rpeaks[1:] / float(fs)
                rri = np.diff(rpeaks, axis=-1) / float(fs)

                # Median filtering on RRI
                rri = medfilt(rri, kernel_size=3)

                # EDR from the ECG (using R-peak amplitudes)
                edr_tm = rpeaks / float(fs)
                edr = segment[rpeaks]

                # Additional constraints on RRI → valid heart rate
                if np.all(np.logical_and(60 / rri >= hr_min, 60 / rri <= hr_max)):
                    rri_time_features = time_domain(rri * 1000)
                    rri_freq_features = frequency_domain(rri, rri_tm)
                    edr_freq_features = frequency_domain(edr, edr_tm)

                    data.append([
                        rri_time_features["rmssd"],
                        rri_time_features["sdnn"],
                        rri_time_features["nn50"],
                        rri_time_features["pnn50"],
                        rri_time_features["mrri"],
                        rri_time_features["mhr"],
                        rri_freq_features["vlf"] / rri_freq_features["total_power"],
                        rri_freq_features["lf"] / rri_freq_features["total_power"],
                        rri_freq_features["hf"] / rri_freq_features["total_power"],
                        rri_freq_features["lf_hf"],
                        rri_freq_features["lfnu"],
                        rri_freq_features["hfnu"],
                        edr_freq_features["vlf"] / edr_freq_features["total_power"],
                        edr_freq_features["lf"] / edr_freq_features["total_power"],
                        edr_freq_features["hf"] / edr_freq_features["total_power"],
                        edr_freq_features["lf_hf"],
                        edr_freq_features["lfnu"],
                        edr_freq_features["hfnu"]
                    ])
                else:
                    data.append([np.nan] * 18)
            else:
                data.append([np.nan] * 18)

        return np.array(data, dtype="float")

    @staticmethod
    def shift(xs, n):
        """
        Shift array elements by n positions (for building acquisition features).
        Elements shifted out of range are replaced with np.nan.
        """
        e = np.empty_like(xs)
        if n > 0:
            e[:n] = np.nan
            e[n:] = xs[:-n]
        elif n < 0:
            e[n:] = np.nan
            e[:n] = xs[-n:]
        else:
            e[:] = xs[:]
        return e

    def acquisition_features(self, data, time_window_size):
        """
        Stacks multiple shifted versions of data to capture
        a "window" of past features for each row.
        """
        features_list = []
        temp = []

        for w in range(time_window_size + 1):
            temp.append(self.shift(data, w))

        # Concatenate along feature dimension
        temp = np.concatenate(temp, axis=1)

        # Filter out rows that contain any NaN (because shifting can create NaNs)
        mask = ~np.isnan(temp).any(axis=1)
        valid = temp[mask]

        features_list.append(valid)
        features = np.concatenate(features_list, axis=0)
        return features

    def classify_ecg(self, data):
        """
        Main function to classify 1-minute ECG segments and produce merged apnea intervals.

        Returns
        -------
        list of tuples
            List of (start_time, end_time) for each detected apnea interval.
        """
        # Load model if not already loaded
        print("Loading model...")
        self.load_model()

        # Load and parse the ECG data
        print("Loading ECG data (with timestamps)...")
        df = data
        ecg_signal = df["ecg_value"].values  # numeric ECG signal

        print("Extracting features (1-minute segments)...")
        # shape = (num_segments, 18) before any shifting
        raw_features = self.feature_extraction(ecg_signal)

        # We build 'windowed' features using a shift of 5
        time_window_size = self.window_size
        features = self.acquisition_features(raw_features, time_window_size)

        # Remove NaN rows from final feature set (additional safety check)
        valid_indices = ~np.isnan(features).any(axis=1)
        features = features[valid_indices]

        if len(features) == 0:
            print("Error: No valid features extracted from the ECG data. Check data quality.")
            return []

        # Scale features before prediction
        print("Scaling features...")
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        print("Classifying...")
        predictions = self.model.predict(features_scaled)  # 0 or 1

        # -------------------------------------------------------------------
        # Map each prediction back to a 1-minute segment in the original data
        # -------------------------------------------------------------------
        fs = self.target_fs
        segment_length = 60
        # total number of segments from raw_features
        num_segments = raw_features.shape[0]

        def get_segment_times(s):
            """
            Simple function to get start/end timestamps for segment index s.
            """
            start_idx = s * fs * segment_length
            end_idx = (s + 1) * fs * segment_length - 1
            # clamp end_idx if it goes beyond the data
            end_idx = min(end_idx, len(df) - 1)
            start_time = df["time"].iloc[start_idx]
            end_time = df["time"].iloc[end_idx]
            return start_time, end_time

        apnea_events = []
        in_apnea = False
        event_start = None

        for i, pred in enumerate(predictions):
            original_seg_idx = i + time_window_size  # map back
            if pred == 1:
                # Apnea
                if not in_apnea:
                    in_apnea = True
                    event_start = original_seg_idx
            else:
                # Normal
                if in_apnea:
                    in_apnea = False
                    event_end = original_seg_idx - 1
                    apnea_events.append((event_start, event_end))

        # If we ended still in apnea, close the block
        if in_apnea:
            event_end = (i + time_window_size)
            apnea_events.append((event_start, event_end))

        # Convert these start/end segment indices to timestamps
        apnea_intervals = []
        for (s_start, s_end) in apnea_events:
            start_time, _ = get_segment_times(s_start)
            _, end_time = get_segment_times(s_end)
            apnea_intervals.append((start_time, end_time))

        return apnea_intervals


 # Example usage
if __name__ == "__main__":
    # Read configuration
    config = configparser.ConfigParser()
    config.read('config.ini')

    url = config['INFLUXDB']['URL']
    token = config['INFLUXDB']['TOKEN']
    org = config['INFLUXDB']['ORG']
    bucket = config['BUCKETS']['ECG_BUCKET']
    measurement = config['MEASUREMENTS']['ECG_Measurements']
    sensor_field = config['FIELDS']['ECG']

    # Create detector class
    detector = ECGDetector(url, token, org, sensor_field)

    # Load the ecg data from local file
    df = detector.get_data(bucket, measurement, source='local')
    signal = df[f'{sensor_field}'].values
    t = df["time"].values

    apnea_events = detector.classify_ecg(df)

    print("\nDetected Apnea Intervals (Merged):")
    if not apnea_events:
        print("No apnea intervals found.")
    else:
        for idx, (st, et) in enumerate(apnea_events, start=1):
            print(f"Apnea Event {idx}: Start = {st}, End = {et}")


    # plot detected events
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), sharex=True)

    # Plot the original signal
    ax.plot(t, signal, 'k', label='Respiration Signal')
    ax.set_title('Respiration Signal with Line-Like Events')
    ax.set_ylabel('Amplitude')

    # Mark line-like intervals
    for start, stop in apnea_events:
        ax.axvspan(start, stop, color='red', alpha=0.2, label='Line-Like Region')

    plt.tight_layout()
    plt.show()