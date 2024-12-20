import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd
from Helper import Helper
import configparser
from datetime import datetime


class CWTBasedApneaDetector(Helper):
    """
    A class to detect apnea events from respiration data using a continuous wavelet transform (CWT).

    Parameters
    ----------
    wavelet_name : str
        The name of the wavelet to use for the CWT.
    scales : array-like
        An array of scales to use for the CWT.
    _threshold_scale : float
        a scale factor in % determine how low the lower threshold should be from the mean power of the signal.
    min_duration : float
        The minimum duration (in seconds) that the signal must remain below the threshold to consider an apnea event.
    fs : float
        Sampling frequency of the respiration data.
    """

    def __init__(self, url, token, org, field,
                 window_size=80,
                 overlap=0.5,
                 wavelet_name='mexh',
                 scales=np.arange(5, 12),
                 threshold_scale=0.3,
                 min_duration=10,
                 fs=10.0):
        super().__init__(url, token, org, field, fs)
        self._wavelet_name = wavelet_name
        self._window_size = window_size
        self._scales = scales
        self._threshold_scale = threshold_scale
        self._min_duration = min_duration
        self._fs = fs
        self._overlap = overlap


    def detect_apnea_events(self, df):
        """
        Detect apnea events from a given respiration signal using CWT-based analysis.

        Parameters
        ----------
        respiration_signal : array-like

        Returns
        -------
        apnea_events : list of tuples
            A list of apnea event intervals as (start_time, end_time) in seconds.
        """
        respiration_signal = df[f'{self.field}'].values
        if not isinstance(respiration_signal, np.ndarray):
            respiration_signal = np.array(respiration_signal)

        signal_length = len(respiration_signal)
        step_size = int(self._window_size * (1 - self._overlap) * self._fs)
        window_samples = int(self._window_size * self._fs)
        apnea_events = []

        for start_idx in range(0, signal_length, step_size):
            end_idx = start_idx + window_samples
            if end_idx > signal_length:
                end_idx = signal_length

            window_signal = respiration_signal[start_idx:end_idx]
            # TODO: delete this timevector and replace the below with the df of the app
            #time = np.arange(start_idx, end_idx) / self._fs

            # Compute the Continuous Wavelet Transform for the window
            coefficients, _ = pywt.cwt(window_signal, self._scales, self._wavelet_name)
            respiration_energy = np.sum(np.abs(coefficients), axis=0)

            # Define a threshold to detect low-energy regions
            threshold = np.mean(respiration_energy) * self._threshold_scale
            low_energy_mask = respiration_energy < threshold

            # Find intervals of line-like regions
            line_start = []
            line_stop = []
            previous_state = False

            # TODO: replace time vector with the timestamps from application
            # e.g. line_start = df.loc[i, 'time']
            for i in range(len(low_energy_mask)):
                if low_energy_mask[i] and not previous_state:
                    event_start_time = datetime.fromisoformat(str(df.loc[i, 'time']))
                    line_start.append(event_start_time)
                elif not low_energy_mask[i] and previous_state:
                    event_stop_time = datetime.fromisoformat(str(df.loc[i, 'time']))
                    line_stop.append(event_stop_time)
                previous_state = low_energy_mask[i]

            if previous_state:
                end = datetime.fromisoformat(str(df.loc[len(low_energy_mask) - 1, 'time']))
                line_stop.append(end)

            window_apnea_events = [(start, stop) for start, stop in zip(line_start, line_stop)
                                   if (stop - start).total_seconds() >= self._min_duration]

            # Collect apnea events, adjusting time to the full signal
            apnea_events.extend([(start, stop) for start, stop in window_apnea_events])

            # ONLY FOR DEBUGGING
            # uncomment to plot the signal with the power spectrum and detected events
            #self.plot_cwt(time, window_signal, window_apnea_events, coefficients, self._scales, respiration_energy, threshold)

        # merge overlapping or adjacent events from the moving average filter
        merged_apnea_events = self._merge_events(apnea_events)

        return merged_apnea_events

    def _merge_events(self, apnea_events):
        # Merge overlapping or adjacent events
        merged_apnea_events = []
        for event in apnea_events:
            if not merged_apnea_events:
                merged_apnea_events.append(event)
            else:
                last_start, last_stop = merged_apnea_events[-1]
                current_start, current_stop = event

                # If the current event starts before or right after the last one ends, merge them
                if current_start <= last_stop:
                    merged_apnea_events[-1] = (last_start, max(last_stop, current_stop))
                else:
                    merged_apnea_events.append(event)

        return merged_apnea_events

    def plot_cwt(self, time, signal, apnea_events, coefficients, scales, respiration_energy, threshold):
        """
        Plot the respiration signal, CWT power spectrum, and power envelope with detected apnea events.
        """
        fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Plot the original signal
        ax[0].plot(time, signal, 'k', label='Respiration Signal')
        ax[0].set_title('Respiration Signal with Line-Like Events')
        ax[0].set_ylabel('Amplitude')

        # Mark line-like intervals
        for start, stop in apnea_events:
            ax[0].axvspan(start, stop, color='red', alpha=0.2, label='Line-Like Region')

        # Avoid duplicate legends
        handles, labels = ax[0].get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax[0].legend(unique_labels.values(), unique_labels.keys())

        # Plot the wavelet power spectrum
        im = ax[1].imshow(np.abs(coefficients), extent=[time[0], time[-1], scales[-1], scales[0]],
                          aspect='auto', cmap='jet')
        ax[1].set_title('Continuous Wavelet Transform (CWT) Power Spectrum')
        ax[1].set_ylabel('Scale (Width)')
        fig.colorbar(im, ax=ax[1], label='Power')

        # Plot the power envelope and threshold
        ax[2].plot(time, respiration_energy, 'b', label='Power Envelope')
        ax[2].axhline(threshold, color='r', linestyle='--', label='Threshold')
        ax[2].set_title('Integrated Power over Selected Scales')
        ax[2].set_ylabel('Power')
        ax[2].set_xlabel('Time (s)')
        ax[2].legend()

        # Adjust layout and show the plot
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
    bucket = config['BUCKETS']['PULSEOXY_BUCKET']
    measurement = config['MEASUREMENTS']['Respiration_Measurements']
    sensor_field = config['FIELDS']['Respiration']

    # Create detector with 60 Hz data and a 5-minute rolling window
    detector = CWTBasedApneaDetector(url, token, org, sensor_field)

    # Load the respiration data
    df = pd.read_csv("resp_value_last_session.csv")
    signal = df[f'{sensor_field}'].values
    scales = np.arange(5, 12)  # Corresponding to frequencies between 0.2 and 0.5 Hz
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

