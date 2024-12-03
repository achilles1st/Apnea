import numpy as np
import scipy.signal as signal
import scipy.signal.windows as windows
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import sys
from influxdb_client import InfluxDBClient
import pandas as pd


url = "http://sleep-apnea:8086"
token = "GDX7_fFUFmHYR9B17dnlg_mdc3xK87_yO9QlRUovGqDtsYroInFXII-jJuWYKM8SLHzp7H5psGbRO72Z2UB7mQ=="
org = "TU"

# Initialize InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()

def get_data(bucket, measurement, field, source='local'):
    if source == 'influx':
        try:
            # Flux query to get data from the specified time range
            flux_query = f'''
                from(bucket: "{bucket}")
                |> range(start: -{24}h)
                |> filter(fn: (r) => r._measurement == "{measurement}")
                |> filter(fn: (r) => r._field == "{field}")
            '''
            result = query_api.query(flux_query)

            data_points = []

            for table in result:
                for record in table.records:
                    data_points.append({
                        'time': record.get_time(),
                        'value': record.get_value()
                    })

            return pd.DataFrame(data_points)

        except Exception as e:
            print(f"Error in get_data for {field}: {e}")
            print("Reading data from ecg_last_session.csv")
            return pd.read_csv('ecg_last_session.csv')

    elif source == 'local':
        print("Reading data from ecg_last_session.csv")
        return pd.read_csv('ecg_last_session.csv')

    else:
        raise ValueError("Invalid source specified. Use 'influx' or 'local'.")


def detect_r_peaks(ecg_signal, fs):
    """
    Detect R-peaks in the ECG signal.
    """
    # High-pass filter to remove baseline wander
    sos = signal.butter(1, 5, 'hp', fs=fs, output='sos')
    filtered_ecg = signal.sosfilt(sos, ecg_signal)

    # Find peaks - R-peaks are the prominent peaks in the ECG signal
    distance = int(0.2 * fs)  # Minimum distance between peaks in samples (200 ms)
    peaks, _ = signal.find_peaks(filtered_ecg, distance=distance, height=np.mean(filtered_ecg))
    return peaks


def calculate_rr_intervals(r_peaks, fs):
    """
    Calculate R-R intervals from detected R-peaks.
    """
    # Calculate R-R intervals in seconds
    rr_intervals = np.diff(r_peaks) / fs
    return rr_intervals


def detect_apnea_events(rr_intervals, fs):
    """
    Detect apnea events based on patterns in R-R intervals.
    """
    apnea_events = []
    window_size = 60  # Number of beats to consider for trend
    for i in range(len(rr_intervals) - window_size * 2):
        window1 = rr_intervals[i:i + window_size]
        window2 = rr_intervals[i + window_size:i + window_size * 2]
        mean1 = np.mean(window1)
        mean2 = np.mean(window2)
        if mean1 > mean2 * 1.1:
            # Possible apnea event detected
            apnea_events.append((i, i + window_size * 2))
    return apnea_events


def plot_ecg_with_apnea(ecg_signal, r_peaks, apnea_events, fs):
    """
    Plot ECG signal and highlight detected apnea events.
    """
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()

    win = pg.GraphicsLayoutWidget(show=True, title="ECG Signal with Apnea Detection")
    p = win.addPlot(title="ECG Signal")
    time_axis = np.arange(len(ecg_signal)) / fs
    p.plot(time_axis, ecg_signal, pen='b')

    # Mark R-peaks
    p.plot(time_axis[r_peaks], ecg_signal[r_peaks], pen=None, symbol='o', symbolBrush='r')

    # Highlight apnea events
    for event in apnea_events:
        start_sample = r_peaks[event[0]]
        end_sample = r_peaks[event[1]]
        rect = QtWidgets.QGraphicsRectItem(
            time_axis[start_sample], np.min(ecg_signal),
            time_axis[end_sample] - time_axis[start_sample], np.max(ecg_signal) - np.min(ecg_signal)
        )
        rect.setBrush(pg.mkBrush(255, 0, 0, 50))
        rect.setPen(pg.mkPen(None))
        p.addItem(rect)

    # Use exec() or exec_() depending on PyQt version
    app = QtWidgets.QApplication.instance()
    if hasattr(app, 'exec_'):
        app.exec_()
    else:
        app.exec()


if __name__ == "__main__":
    # Load ECG data from numpy array
    fs = 128  # Sampling frequency
    df = get_data("ECG", "ecg_samples", "ecg_value", source='local')

    # 2. Preprocessing the ECG Signal
    ecg_signal = np.array(df['ecg_value'].values)

    # Run the detection algorithm
    r_peaks = detect_r_peaks(ecg_signal, fs)
    rr_intervals = calculate_rr_intervals(r_peaks, fs)
    apnea_events = detect_apnea_events(rr_intervals, fs)

    # Plot the ECG signal and highlight apnea events
    plot_ecg_with_apnea(ecg_signal, r_peaks, apnea_events, fs)

