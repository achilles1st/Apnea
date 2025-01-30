import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Helper import Helper
import configparser
from scipy.signal import butter, filtfilt, hilbert


class EnvelopeBasedApneaDetector(Helper):
    """
    A class to detect apnea events from respiration data using a Hilbert transform.

    Parameters
    ----------
    url : str
        The URL of the InfluxDB instance.
    token : str
        The authentication token for InfluxDB.
    org : str
        The organization name in InfluxDB.
    field : str
        The field name in the InfluxDB measurement that contains the respiration data.
    window_size : float, optional
        The size of the rolling window in seconds (default is 80).
    overlap : float, optional
        The fraction of overlap between consecutive windows (default is 0.5).
    threshold_scale : float, optional
        A scale factor to determine the threshold for detecting apnea events (default is 0.3).
    min_duration : float, optional
        The minimum duration (in seconds) that the signal must remain below the threshold to consider an apnea event (default is 4).
    fs : float, optional
        The sampling frequency of the respiration data (default is 10.0).
    """

    def __init__(self, url, token, org, field,
                 window_size=80,
                 overlap=0.5,
                 threshold_scale=0.3,
                 min_duration=4,
                 fs=10.0):
        super().__init__(url, token, org, field, fs)
        self._window_size = window_size
        self._threshold_scale = threshold_scale
        self._min_duration_event = min_duration
        self._fs = fs
        self._overlap = overlap


    def detect_apnea_events(self, df):
        """
        Detect apnea events from a given respiration signal using envelope analysis.

        Parameters
        ----------
        df: respiration_signal, pandas DataFrame consisting of time, and resp_value columns

        Returns
        -------
        apnea_events : list of tuples
            A list of apnea event intervals as (start_time, end_time) in seconds.
        """
        respiration_signal = df[f'{self.field}'].values
        timestamps = pd.to_datetime(df['time']).values

        signal_length = len(respiration_signal)
        step_size = int(self._window_size * (1 - self._overlap) * self._fs)
        window_samples = int(self._window_size * self._fs)
        apnea_events = []

        for start_idx in range(0, signal_length, step_size):
            end_idx = start_idx + window_samples
            if end_idx > signal_length:
                end_idx = signal_length

            window_signal = respiration_signal[start_idx:end_idx]
            time_window = timestamps[start_idx:end_idx]
            # Check signal length against padlen
            nyquist = 0.5 * self._fs
            low = 0.1 / nyquist
            high = 1.4 / nyquist
            b, a = butter(2, [low, high], btype='band')
            padlen = 3 * max(len(b), len(a))

            if len(window_signal) < padlen:
                print(f"Skipping window starting at index {start_idx} due to insufficient length.")
                continue

            # 2. Filter the signal to remove slow offset/drift
            filtered = self.bandpass_filter(window_signal, self._fs, lowcut=0.16, highcut=2)

            # 3. Compute the amplitude envelope
            envelope = self.amplitude_envelope(filtered)

            # Set a threshold as a fraction of the maximum RMS
            threshold = 0.2 * np.max(envelope)

            # Identify regions where RMS is below the threshold
            low_energy_mask =  envelope < threshold

            line_start = []
            line_stop = []
            previous_state = False

            for i in range(len(low_energy_mask)):
                if low_energy_mask[i] and not previous_state:
                    line_start.append(time_window[i])
                elif not low_energy_mask[i] and previous_state:
                    line_stop.append(time_window[i])
                previous_state = low_energy_mask[i]

            if previous_state:
                line_stop.append(time_window[-1])

            window_apnea_events = [(start, stop) for start, stop in zip(line_start, line_stop)
                                   if
                                   (stop - start).astype('timedelta64[s]').item().total_seconds() >= self._min_duration_event]

            apnea_events.extend(window_apnea_events)

            # ONLY FOR DEBUGGING
            # apnea_events_buffer = window_apnea_events
            # self.plot_envelope(time_window, window_signal, apnea_events_buffer, envelope, threshold)

        merged_apnea_events = self._merge_events(apnea_events)

        return merged_apnea_events

    def _merge_events(self, apnea_events):
        merged_apnea_events = []
        for event in apnea_events:
            if not merged_apnea_events:
                merged_apnea_events.append(event)
            else:
                last_start, last_stop = merged_apnea_events[-1]
                current_start, current_stop = event

                if current_start <= last_stop:
                    merged_apnea_events[-1] = (last_start, max(last_stop, current_stop))
                else:
                    merged_apnea_events.append(event)

        return merged_apnea_events

    def bandpass_filter(self, data, fs, lowcut=0.1, highcut=1.0, order=2):
        """
        Applies a bandpass Butterworth filter to remove offsets
        and frequencies outside the normal respiration band.
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data


    def amplitude_envelope(self, data):
        """
        Computes the amplitude envelope via the Hilbert transform.
        """
        analytic_signal = hilbert(data)
        envelope = np.abs(analytic_signal)
        return envelope

    def plot_envelope(self, t, signal, apnea_events, rms, threshold):
        """
        Plot the respiration signal and power spectrum with detected apnea events.
        """
        # Step 3: Plotting the Results
        fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # Plot the original signal
        ax[0].plot(t, signal, 'k', label='Signal')
        ax[0].set_title('Original Signal with Pauses')
        ax[0].set_ylabel('Amplitude')
        ax[0].text(-0.1, 1.1, 'a)', transform=ax[0].transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

        # Mark apnea intervals
        for start, stop in apnea_events:
            ax[0].axvspan(start, stop, color='red', alpha=0.2, label='Detected Low-Power Period')

        # Avoid duplicate legends
        handles, labels = ax[0].get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax[0].legend(unique_labels.values(), unique_labels.keys())

        # Plot the bandpassed signal
        bandpassed_signal = self.bandpass_filter(signal, self._fs, lowcut=0.1, highcut=1.0)
        ax[1].plot(t, bandpassed_signal, 'g', label='Bandpassed Signal')
        ax[1].set_title('Bandpassed Signal')
        ax[1].set_ylabel('Amplitude')
        ax[1].legend()
        ax[1].text(-0.1, 1.1, 'b)', transform=ax[1].transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

        # Plot the power and threshold
        ax[2].plot(t, rms, 'b', label='RMS')
        ax[2].axhline(threshold, color='r', linestyle='--', label='Threshold')
        ax[2].set_title('Signal Power and Threshold')
        ax[2].set_ylabel('Power')
        ax[2].set_xlabel('Time (s)')
        ax[2].legend()
        ax[2].text(-0.1, 1.1, 'c)', transform=ax[2].transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

        plt.tight_layout()
        plt.show()


# EXAMPLE USAGE load from csv file
if __name__ == "__main__":
    # Read configuration
    config = configparser.ConfigParser()
    config.read('config.ini')

    url = config['INFLUXDB']['URL']
    token = config['INFLUXDB']['TOKEN']
    org = config['INFLUXDB']['ORG']
    bucket = config['BUCKETS']['RESPIRATORY_BUCKET']
    measurement = config['MEASUREMENTS']['Respiration_Measurements']
    sensor_field = config['FIELDS']['Respiration']

    # Create detector with 60 Hz data and a 5-minute rolling window
    detector = EnvelopeBasedApneaDetector(url, token, org, sensor_field)

    # Load the respiration data
    df = detector.get_data(bucket, measurement, source='local')
    signal = df[f'{sensor_field}'].values
    t = df["time"].values
    # Detect apnea events
    apnea_events = detector.detect_apnea_events(df)

    # plot detected events
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), sharex=True)

    # Plot the original signal
    ax.plot(t, signal, 'k', label='Respiration Signal')
    ax.set_title('Respiration Signal with Line-Like Events')
    ax.set_ylabel('Amplitude')

    # Mark line-like intervals
    for start, stop in apnea_events:
        ax.axvspan(start, stop, color='red', alpha=0.2, label='Line-Like Region')

    # Avoid duplicate legends
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())


    plt.tight_layout()
    plt.show()

