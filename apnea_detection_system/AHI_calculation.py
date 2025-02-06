import pandas as pd
import configparser
from datetime import timedelta
from ecg_detection import ECGDetector
from old_detection.upload_apnea_events import upload_apnea_events
from pulse_oxi_detection import PulseOxiDetector
from Respiration_envelope import EnvelopeBasedApneaDetector
from Helper import Helper
import pytz


class AHICalculator:
    def __init__(self, significance_weights=None, threshold=1.0, merge_window=timedelta(seconds=30)):
        """
        Initialize the AHI Calculator with weights for each sensor modality.
        :param significance_weights: Dictionary with weights for each modality {"respiration": 1.0, "pulse_oxi": 0.7, "ecg": 0.5, "snore": 0.3}
        :param threshold: Minimum cumulative score for an event to count as apnea
        :param merge_window: Time window within which events are merged (default: 5 seconds)
        """
        self.weights = significance_weights if significance_weights else {
            "respiration": 1.0,
            "pulse_oxi": 0.7,
            "pulse_rate": 0.5,
            "ecg": 0.3,
            "snore": 0.3
        }
        self.threshold = threshold
        self.merge_window = merge_window

    def merge_events(self, *event_lists):
        """
        Merge events from different sources and compute weighted significance.
        """
        all_events = []
        for source, events in event_lists:
            if source == "snore":
                for timestamp in events:
                    ts = pd.to_datetime(timestamp).tz_localize('UTC') if pd.to_datetime(
                        timestamp).tzinfo is None else pd.to_datetime(timestamp)
                    all_events.append((ts, ts, source))
            else:
                for start, end in events:
                    start_ts = pd.to_datetime(start).tz_localize('UTC') if pd.to_datetime(
                        start).tzinfo is None else pd.to_datetime(start)
                    end_ts = pd.to_datetime(end).tz_localize('UTC') if pd.to_datetime(
                        end).tzinfo is None else pd.to_datetime(end)
                    all_events.append((start_ts, end_ts, source))

        all_events.sort(key=lambda x: x[0])  # Sort by start time

        merged_events = []
        current_event = None
        current_score = 0

        for start, end, source in all_events:
            weight = self.weights.get(source, 0)
            if current_event is None:
                current_event = (start, end)
                current_score = weight
            else:
                if start <= current_event[1] + self.merge_window:  # Merge if within merge window
                    current_event = (current_event[0], max(current_event[1], end))
                    current_score += weight
                else:
                    if current_score >= self.threshold:
                        merged_events.append((current_event[0], current_event[1]))
                    current_event = (start, end)
                    current_score = weight

        if current_event and current_score >= self.threshold:
            merged_events.append((current_event[0], current_event[1]))

        return merged_events

    def calculate_ahi(self, merged_events, total_sleep_time_hours):
        """
        Compute the AHI given the apnea events and total sleep time.
        :param merged_events: List of detected apnea events (start, end)
        :param total_sleep_time_hours: Total duration of sleep in hours
        :return: AHI value
        """
        num_events = len(merged_events)
        ahi = num_events / total_sleep_time_hours
        return ahi



# start calculation of AHI of all sensors
def main(user_name, upload_apnea_events=True):
    config = configparser.ConfigParser()
    config.read('config.ini')

    url = config['INFLUXDB']['URL']
    token = config['INFLUXDB']['TOKEN']
    org = config['INFLUXDB']['ORG']

    # Extract ECG events
    bucket_ecg = config['BUCKETS']['ECG_BUCKET']
    measurment_ecg = config['MEASUREMENTS']['ECG_Measurements']
    ecg_detector = ECGDetector(url, token, org, "ecg_value")
    df_ecg = ecg_detector.get_data(bucket_ecg, measurment_ecg, source='influx')
    ecg_events = ecg_detector.classify_ecg(df_ecg)
    # upload apnea events to influxdb
    if upload_apnea_events:
        ecg_detector.upload_apnea_events(ecg_events, "ecg_events", user_name, bucket_ecg)

    # Extract Pulse rate events
    bucket_pulseoxy = config['BUCKETS']['PULSEOXY_BUCKET']
    measurement_pulseoxy = config['MEASUREMENTS']['PULSEOXY_Measurements']
    pulse_detector = PulseOxiDetector(url, token, org, "PulseRate", fs=60, baseline_window_minutes=20)
    df_pulse = pulse_detector.get_data(bucket_pulseoxy, measurement_pulseoxy, source='influx')
    df_pulse_baseline = pulse_detector.compute_rolling_baseline(df_pulse, min_periods_minutes=10)
    pulse_events = pulse_detector.detect_apnea_events(df_pulse_baseline)
    # upload apnea events to influxdb
    if upload_apnea_events:
        pulse_detector.upload_apnea_events(pulse_events, "pulse_events", user_name, bucket_pulseoxy)

    # Extract Oximetry events
    spo2_detector = PulseOxiDetector(url, token, org, "spO2", fs=60, baseline_window_minutes=20)
    df_pulse = spo2_detector.get_data(bucket_pulseoxy, measurement_pulseoxy, source='influx')
    df_pulse_baseline = spo2_detector.compute_rolling_baseline(df_pulse, min_periods_minutes=10)
    spo2_events = spo2_detector.detect_apnea_events(df_pulse_baseline)
    # upload apnea events to influxdb
    if upload_apnea_events:
        spo2_detector.upload_apnea_events(spo2_events, "spo2_events", user_name, bucket_pulseoxy)

    # Extract Respiration events
    bucket_resp = config['BUCKETS']['RESPIRATORY_BUCKET']
    measurement_resp = config['MEASUREMENTS']['Respiration_Measurements']
    resp_detector = EnvelopeBasedApneaDetector(url, token, org, "resp_value", fs=10, min_duration=10)
    df_resp = resp_detector.get_data(bucket_resp, measurement_resp, source='influx')
    respiration_events = resp_detector.detect_apnea_events(df_resp)

    timezone = pytz.UTC
    respiration_events = [(pd.Timestamp(start).tz_localize(timezone), pd.Timestamp(end).tz_localize(timezone)) for
                          start, end in respiration_events]

    # upload apnea events to influxdb
    if upload_apnea_events:
        resp_detector.upload_apnea_events(respiration_events, "respiration_events", user_name, bucket_resp)

    # Extract Snoring timestamps
    snore_events = Helper.get_snoring_event_timestamps(url, token, org, "audio", start="-24h")

    # Calculate total sleep time in hours
    start_time = df_resp["time"].min()
    end_time = df_resp["time"].max()
    total_sleep_time_hours = (end_time - start_time).total_seconds() / 3600

    ahi_calculator = AHICalculator()
    merged_events = ahi_calculator.merge_events(
        ("respiration", respiration_events),
        ("pulse_rate", pulse_events),
        ("spo2", spo2_events),
        ("ecg", ecg_events),
        ("snore", snore_events)
    )

    ahi = ahi_calculator.calculate_ahi(merged_events, total_sleep_time_hours)
    print(f"Apnea-Hypopnea Index (AHI): {ahi:.2f}")

    # Upload AHI to InfluxDB
    if upload_apnea_events:
        ecg_detector.upload_ahi_index(ahi, total_sleep_time_hours, "apnea_hypopnea_index", user_name, "AHI")


if __name__ == "__main__":
    upload_apnea_events = True
    user_name = "Stefan"
    main(user_name, upload_apnea_events)