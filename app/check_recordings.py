import tensorflow as tf
import numpy as np
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import ASYNCHRONOUS, WriteOptions
from influxdb_client.domain.write_precision import WritePrecision
import datetime
import configparser

class SnoringDetector:
    def __init__(self):
        # Constants
        self.MODEL_PATH = '/app/model/cnn_new_big_data_deviation_44k.keras'  # Path of the model
        self.THRESHOLD = 0.5  # Probability threshold for snoring classification
        self.SAMPLE_RATE = 44100  # Expected sample rate of audio
        self.SEGMENT_DURATION = 1  # Duration of each audio segment in seconds

        # Load configuration from config.ini
        self.config = configparser.ConfigParser()
        self.config.read('/app/config.ini')  # Adjust the path as needed

        # InfluxDB connection settings
        self.INFLUXDB_URL = self.config['INFLUXDB']['URL']
        self.INFLUXDB_TOKEN = self.config['INFLUXDB']['TOKEN']
        self.INFLUXDB_ORG = self.config['INFLUXDB']['ORG']
        self.INFLUXDB_BUCKET = 'audio'

        # Initialize InfluxDB client
        self.client = InfluxDBClient(
            url=self.INFLUXDB_URL,
            token=self.INFLUXDB_TOKEN,
            org=self.INFLUXDB_ORG
        )
        self.write_api = self.client.write_api(write_options=WriteOptions(
            write_type=ASYNCHRONOUS,
            batch_size=1000,
            flush_interval=1000,
            jitter_interval=0,
            retry_interval=5000,
            max_retries=3,
            max_retry_delay=30000,
            exponential_base=2
        ))

        # Load the trained model
        self.model = tf.keras.models.load_model(self.MODEL_PATH)

    def get_log_mel(self, waveform, sample_rate):
        frame_length = int(0.032 * sample_rate)  # Approximately 32ms
        frame_step = int(0.016 * sample_rate)  # Approximately 16ms
        stfts = tf.signal.stft(waveform, frame_length=frame_length, frame_step=frame_step)

        # Obtain the magnitude of the STFT.
        spectrograms = tf.abs(stfts)
        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 40.0, 6000.0, 30
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(
            spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        log_mel_spectrograms = tf.reshape(log_mel_spectrograms, [-1, 1830])
        return log_mel_spectrograms

    def classify_snoring_segment(self, segment, sample_rate):
        """Classifies snoring events in the given audio segment."""
        # Convert segment to tensor and normalize
        x = tf.convert_to_tensor(segment, dtype=tf.float32)
        x = x / (tf.reduce_max(tf.abs(x)) + 1e-6)  # Normalize between -1 and 1

        # Ensure x is 1D
        x = tf.reshape(x, [-1])

        # Get log-mel spectrogram
        x = self.get_log_mel(x, sample_rate)

        # Make prediction
        prediction = self.model(x)
        probability = prediction[0, 0].numpy()
        return probability

    def process_audio_segment(self, segment, user_name):
        """Processes an audio segment for snoring detection."""
        probability = self.classify_snoring_segment(segment, self.SAMPLE_RATE)

        if probability >= self.THRESHOLD:
            event_time = datetime.datetime.utcnow()

            # Write snoring event to InfluxDB
            point = Point("snoring_events") \
                .tag("user_name", user_name) \
                .field("probability", probability) \
                .time(event_time, WritePrecision.NS)
            self.write_api.write(bucket=self.INFLUXDB_BUCKET, org=self.INFLUXDB_ORG, record=point)

    def close(self):
        """Closes the InfluxDB client."""
        self.write_api.close()
        self.client.close()
