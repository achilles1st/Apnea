"""
For Debugging purposes only!!!
This script is used to extract the ECG data from the InfluxDB and save it to a CSV file.
"""

import pandas as pd
from Helper import Helper
from tqdm import tqdm
import configparser



class PulseOxiDetector(Helper):
    def __init__(self, url, token, org, field, fs=60, baseline_window_minutes=10):
        super().__init__(url, token, org, field, fs)
        self.baseline_window_minutes = baseline_window_minutes
        self.baseline_window_samples = self.baseline_window_minutes * 60 * self.fs
        # Hysteresis thresholds:
        if self.field == 'PulseRate':
            self.event_start_threshold_function = lambda baseline, std: baseline - (2 * std)
            self.event_end_threshold_function = lambda baseline, std: baseline
        elif self.field == 'spO2':
            self.event_start_threshold_function = lambda baseline, std: baseline - (3 * std)
            self.event_end_threshold_function = lambda baseline, std: baseline
        elif self.field == 'PulseWave':
            print("reading PulseWave data")
        else:
            raise ValueError(f"Unknown field: {field}")

    def compute_rolling_baseline(self, df, min_periods_minutes=5):
        """
        Compute a rolling baseline for sensor type using a rolling median over a fixed number of minutes/samples.
        Ignores values where sensor == 0, which indicates a disconnection or inability to find the finger.
        """
        min_periods_samples = int(min_periods_minutes * 60 * self.fs)
        df_filtered = df[df[f'{self.field}'] > 0]  # Filter out invalid sensor values
        df['baseline'] = df_filtered[f'{self.field}'].rolling(self.baseline_window_samples, min_periods=min_periods_samples).median()
        df['std'] = df_filtered[f'{self.field}'].rolling(self.baseline_window_samples, min_periods=min_periods_samples).std()

        return df

    def detect_apnea_events(self, df):
        """
        Detect apnea events with hysteresis:

        - start_condition: value < event_start_threshold(baseline)
        - end_condition:   value > event_end_threshold(baseline)

        Ignores rows where value == 0 (indicating a disconnection) and ignores events
        that last longer than one minute.
        """
        events = []
        currently_in_event = False
        event_start_time = None
        max_event_duration = pd.Timedelta(minutes=2)  # two minute threshold

        for i in tqdm(range(len(df)), desc="Detecting Apnea Events", unit="rows"):
            value = df.loc[i, f'{self.field}']

            # Skip rows with invalid values
            if value == 0:
                continue

            baseline = df.loc[i, 'baseline']
            std = df.loc[i, 'std']

            # If baseline isn't computed yet (e.g. at the very start), skip
            if pd.isna(baseline):
                continue

            start_threshold = self.event_start_threshold_function(baseline, std)
            end_threshold = self.event_end_threshold_function(baseline, std)

            if currently_in_event:
                # Currently in an event, check if we should end it
                if value >= end_threshold:
                    event_end_time = df.loc[i, 'time']
                    # Check event duration before appending
                    if event_end_time - event_start_time <= max_event_duration:
                        events.append((event_start_time, event_end_time))
                    # Else: if the event is longer than one minute, ignore it
                    currently_in_event = False
            else:
                # Not in an event, check if we should start one
                if value <= start_threshold:
                    currently_in_event = True
                    event_start_time = df.loc[i, 'time']

        # If we reach the end of the data and are still in an event,
        # finalize the event and check its duration.
        if currently_in_event:
            event_end_time = df.loc[len(df) - 1, 'time']
            if event_end_time - event_start_time <= max_event_duration:
                events.append((event_start_time, event_end_time))

        return events



# EXAMPLE USAGE
if __name__ == "__main__":
    # Read configuration
    config = configparser.ConfigParser()
    config.read('config.ini')

    url = config['INFLUXDB']['URL']
    token = config['INFLUXDB']['TOKEN']
    org = config['INFLUXDB']['ORG']
    bucket = config['BUCKETS']['PULSEOXY_BUCKET']
    measurement = config['MEASUREMENTS']['PULSEOXY_Measurements']
    sensor_field = config['FIELDS']['PulseRate']

    # Create detector with 60 Hz data and a 5-minute rolling window
    detector = PulseOxiDetector(url, token, org, sensor_field, fs=50, baseline_window_minutes=10)
    df = detector.get_data(bucket, measurement, source='influx')

    detector.verify_sampling_rate(df, 60)

    df = detector.compute_rolling_baseline(df, min_periods_minutes=5)
    # Calculate the unique baselines
    unique_baselines = df['baseline'].unique()
    # Print the unique baselines
    print("Unique baselines:")
    print(unique_baselines)

    apnea_events = detector.detect_apnea_events(df)
    print(apnea_events)
    if apnea_events:
        print("Detected apnea events")
        for event in apnea_events:
            print(event)
    else:
        print("No apnea events detected.")



