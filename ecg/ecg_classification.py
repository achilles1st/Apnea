import os
import numpy as np
import pandas as pd
import joblib
from scipy.signal import resample
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from hrv.classical import frequency_domain, time_domain
from scipy.signal import medfilt
import biosppy.signals.tools as st
from sklearn.preprocessing import StandardScaler

# Set file paths directly (modify as needed)
CSV_FILE_PATH = "ecg_last_session_hold2.csv"  # Replace with your CSV file
MODEL_PATH = "mlp_classifier_model_base.pkl"
TARGET_FS = 128  # Target sampling frequency


def load_model():
    """Loads the trained classifier."""
    clf = joblib.load(MODEL_PATH)
    return clf


def load_ecg_csv():
    """
    Loads a DataFrame with time and ecg_value from a CSV file.
    Returns the DataFrame (not just the ecg values!).
    """
    df = pd.read_csv(CSV_FILE_PATH)
    if "time" not in df.columns or "ecg_value" not in df.columns:
        raise ValueError("CSV file must contain 'time' and 'ecg_value' columns.")
    return df


def feature_extraction(signal):
    """
    Extracts features from ECG signal in 1-minute segments for classification.
    Returns a (num_segments, num_features) numpy array.
    """
    fs = TARGET_FS
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

            # A bit of median filtering on RRI
            rri = medfilt(rri, kernel_size=3)

            # EDR from the ECG
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


def shift(xs, n):
    """Shift array elements by n positions (for building acquisition features)."""
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


def acquisition_features(data, time_window_size):
    """
    Stacks multiple shifted versions of data to capture
    a "window" of past features for each row.
    """
    # We keep collecting features in a list:
    features = []

    # Shift from 0 to time_window_size
    temp = []
    for w in range(time_window_size + 1):
        temp.append(shift(data, w))

    # Concatenate along feature dimension
    temp = np.concatenate(temp, axis=1)
    # Filter out rows that contain any NaN (because shifting can create NaNs)
    mask = ~np.isnan(temp).any(axis=1)
    valid = temp[mask]

    features.append(valid)
    features = np.concatenate(features, axis=0)
    return features


def classify_ecg():
    """Main function to classify 1-minute ECG segments and produce merged apnea intervals."""
    print("Loading model and scaler...")
    clf = load_model()

    print("Loading ECG data (with timestamps)...")
    df = load_ecg_csv()
    ecg_signal = df["ecg_value"].values  # numeric ECG signal

    print("Extracting features (1-minute segments)...")
    # shape = (num_segments, 18) before any shifting
    raw_features = feature_extraction(ecg_signal)

    # We build 'windowed' features using a shift of 5
    time_window_size = 5
    features = acquisition_features(raw_features, time_window_size)
    # Remove NaN rows from final feature set
    valid_indices = ~np.isnan(features).any(axis=1)
    features = features[valid_indices]

    if len(features) == 0:
        print("Error: No valid features extracted from the ECG data. Check data quality.")
        return

    # Scale and predict
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    print("Classifying...")
    predictions = clf.predict(features_scaled)  # 0 or 1

    # -------------------------------------------------------------------
    # Map each prediction back to a 1-minute segment in the original data
    # -------------------------------------------------------------------
    #
    # raw_features had shape (num_segments, 18)
    # after acquisition_features(..., 5), we lose the first 5 segments
    #
    # So if predictions[i] is the i-th valid row after shifting,
    # that corresponds to the segment index "i + time_window_size"
    # in the original 0..(num_segments-1) space.
    #
    # A 1-minute segment i (in the original space) covers:
    #    samples [ i*fs*60 : (i+1)*fs*60 )
    #
    # We'll get its start time from df["time"].iloc[i*fs*60]
    # We'll get its end time   from df["time"].iloc[(i+1)*fs*60 - 1]
    #
    # (We need to check array bounds carefully at the end.)
    # -------------------------------------------------------------------

    fs = TARGET_FS
    segment_length = 60
    # total number of segments from raw_features
    num_segments = raw_features.shape[0]

    # Simple function to get start/end timestamps for segment index s
    def get_segment_times(s):
        start_idx = s * fs * segment_length
        end_idx = (s + 1) * fs * segment_length - 1
        # clamp end_idx if it goes beyond the data
        end_idx = min(end_idx, len(df) - 1)
        start_time = df["time"].iloc[start_idx]
        end_time = df["time"].iloc[end_idx]
        return start_time, end_time

    # Build a list of (start_segment, end_segment) for apnea blocks
    apnea_events = []
    in_apnea = False
    event_start = None

    for i, pred in enumerate(predictions):
        original_seg_idx = i + time_window_size  # map back
        if pred == 1:
            # Apnea
            if not in_apnea:
                # just entered an apnea block
                in_apnea = True
                event_start = original_seg_idx
        else:
            # Normal
            if in_apnea:
                # just left an apnea block
                in_apnea = False
                # we ended at previous segment
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

    # Print or return the final merged intervals
    print("\nDetected Apnea Intervals (Merged):")
    if not apnea_intervals:
        print("No apnea intervals found.")
    else:
        for idx, (st, et) in enumerate(apnea_intervals, start=1):
            print(f"Apnea Event {idx}: Start = {st}, End = {et}")


if __name__ == "__main__":
    classify_ecg()
