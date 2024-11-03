import os
import tensorflow as tf
import soundfile as sf
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import ASYNCHRONOUS, WriteOptions
from influxdb_client.domain.write_precision import WritePrecision
import datetime
import configparser


class SnoringDetector:
    def __init__(self):
        # Constants
        self.MODEL_PATH = '/app/model/cnn_new_big_data_deviation_44k.keras'  # Path of the model
        self.RECORDINGS_DIR = '/data/recordings'
        self.SEGMENT_DURATION = 1  # Duration of each audio segment in seconds
        self.THRESHOLD = 0.5  # Probability threshold for snoring classification
        self.SAMPLE_RATE = 44100  # Expected sample rate of audio files

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

    def process_audio_file(self, file_path):
        print(f"Processing file: {file_path}")

        # Open the audio file as a stream
        with sf.SoundFile(file_path) as f:
            # Check sample rate
            if f.samplerate != self.SAMPLE_RATE:
                print(f"Sample rate mismatch: Expected {self.SAMPLE_RATE}, got {f.samplerate}. Resampling is required.")
                return

            samples_per_segment = int(f.samplerate * self.SEGMENT_DURATION)
            total_snoring_duration = 0  # Initialize total snoring duration
            total_segments = 0  # Total number of segments processed

            while True:
                # Read a segment
                segment = f.read(samples_per_segment)
                if len(segment) == 0:
                    break  # End of file

                total_segments += 1  # Increment total segments processed

                # Process the segment
                probability = self.classify_snoring_segment(segment, f.samplerate)

                if probability >= self.THRESHOLD:
                    event_time = datetime.datetime.utcnow()

                    # Increment total snoring duration
                    total_snoring_duration += self.SEGMENT_DURATION  # Each segment is of SEGMENT_DURATION seconds

                    # Write individual snoring events to InfluxDB (optional)
                    point = Point("snoring_events") \
                        .field("probability", probability) \
                        .time(event_time, WritePrecision.NS)
                    self.write_api.write(bucket=self.INFLUXDB_BUCKET, org=self.INFLUXDB_ORG, record=point)

            # After processing all segments, calculate snoring percentage
            recording_duration = total_segments * self.SEGMENT_DURATION  # Total recording duration
            snoring_percentage = (total_snoring_duration / recording_duration) * 100 if recording_duration > 0 else 0

            # Create a summary point for total snoring duration
            summary_point = Point("snoring_summary") \
                .tag("file_name", os.path.basename(file_path)) \
                .field("total_snoring_duration", total_snoring_duration) \
                .field("recording_duration", recording_duration) \
                .field("snoring_percentage", snoring_percentage) \
                .time(datetime.datetime.now(datetime.timezone.utc), WritePrecision.NS)

            self.write_api.write(bucket=self.INFLUXDB_BUCKET, org=self.INFLUXDB_ORG, record=summary_point)

        print(f"Finished processing file: {file_path}")

    def process_new_recording(self):
        # Check if the directory exists and is accessible
        if os.path.isdir(self.RECORDINGS_DIR):
            # List audio files in the directory
            audio_files = [f for f in os.listdir(self.RECORDINGS_DIR) if f.endswith('.wav') or f.endswith('.mp3')]

            if not audio_files:
                # Directory is empty; do nothing
                print("No audio files to process.")
            else:
                # Process each audio file
                for audio_file in audio_files:
                    file_path = os.path.join(self.RECORDINGS_DIR, audio_file)
                    self.process_audio_file(file_path)

                    # Optionally, move or delete the processed file
                    os.remove(file_path)  # Delete the file after processing

                # Close InfluxDB client
                self.write_api.close()
                self.client.close()
        else:
            print(f"The directory {self.RECORDINGS_DIR} does not exist.")


if __name__ == "__main__":
    detector = SnoringDetector()
    detector.process_new_recording()
