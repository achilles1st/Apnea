from flask import Flask, render_template, request
import sounddevice as sd
import threading
import numpy as np
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import time
from influxdb_client.domain.write_precision import WritePrecision
from serial.tools import list_ports
from pulox_250_realtime import CMS50Dplus

app = Flask(__name__)

recording = False
amplification_factor = 2  # Amplification factor (e.g., 2.0 will double the loudness)
channels = 1
sample_rate = 44100
device_index = 1
frames = []  # Global list to store audio frames

# InfluxDB connection settings
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "9RdNMHQLhHuujCeLFCJpunRbkOOq7PDN2RoxW55-sl8x3_CBPuaHvLXQxL9QqJfdv7Kod82hnrPJkZi8xVUfjQ=="
INFLUXDB_ORG = "TU"
INFLUXDB_BUCKET = "audio"

# Initialize InfluxDB client
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

user_name = ""  # Global variable to store the username

def audio_callback(indata):
    global frames, write_api
    if recording:
        # Amplify the audio data safely
        amplified_data = np.clip(indata * amplification_factor, -32768, 32767)
        frames.append(amplified_data.astype(np.int16).tobytes())

        # Current timestamp for each sample (in nanoseconds)
        current_time = time.time_ns()

        # Prepare points for batch upload
        audio_points = [
            Point("audio_samples")
            .tag("user_name", user_name)
            .field("amplitude", int(sample[0]))
            .time(current_time + i, WritePrecision.NS)
            for i, sample in enumerate(amplified_data)
        ]

        # Batch write all points at once
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=audio_points)

def record_audio():
    global recording, frames

    frames = []  # Clear frames list at the start
    start_time = time.time()  # Record start time for metadata

    # Start the stream using sounddevice with specified device index
    with sd.InputStream(device=device_index, channels=channels, samplerate=sample_rate, dtype='int16', callback=audio_callback):
        while recording:
            sd.sleep(100)  # Small delay to prevent high CPU usage

    frames = []  # Clear frames after

def record_oximeter():
    global recording
    port = list_ports.comports()[0].device  # Select the first available serial port
    while recording:
        oximeter = CMS50Dplus(port)
        datapoints = oximeter.get_realtime_data()

        for datapoint in datapoints:
            # Prepare points for batch upload
            points = Point("pulseoxy_samples") \
                .tag("user_name", user_name) \
                .field("spO2", int(datapoint.spO2)) \
                .field("PulseWave", int(datapoint.pulse_waveform)) \
                .field("PulseRate", int(datapoint.pulse_rate)) \
                .time(int(datapoint.time.timestamp() * 1e9), WritePrecision.NS)


            # Batch write all points at once
            write_api.write(bucket="Pulseoxy", org=INFLUXDB_ORG, record=points)

@app.route('/')
def index():
    return render_template('start.html')

@app.route('/start', methods=['POST'])
def start():
    global recording, user_name
    user_name = request.form.get('user_name')  # Get the username from the form
    recording = True
    threading.Thread(target=record_audio).start()
    threading.Thread(target=record_oximeter).start()  # Start oximeter recording
    return render_template('stop.html')

@app.route('/stop', methods=['POST'])
def stop():
    global recording
    recording = False
    return render_template('start.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
