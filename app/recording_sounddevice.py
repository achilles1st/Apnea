from flask import Flask, render_template, request
import sounddevice as sd
import threading
import wave
import numpy as np
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import time
from datetime import datetime

app = Flask(__name__)

recording = False
amplification_factor = 2  # Amplification factor (e.g., 2.0 will double the loudness)
channels = 1
sample_rate = 44100
device_index = 1
frames = []  # Global list to store audio frames

# InfluxDB connection settings
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN ="Y3iGvFrvVpH6NA-r_q7HYi27s8RCR6Roz6gJMrysfq9DP8OgtZ9MF7goBBIjkDQOiL14x02SeveFKo0mvDKTxQ=="
INFLUXDB_ORG = "TU"
INFLUXDB_BUCKET = "audio"

# Initialize InfluxDB client
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

user_name = ""  # Global variable to store the username

def audio_callback(indata, frames_count, time_info, status):
    global frames
    if recording:
        # Amplify the audio data
        amplified_data = np.clip(indata * amplification_factor, -32768, 32767)
        frames.append(amplified_data.astype(np.int16).tobytes())

def record_audio():
    global recording, frames

    frames = []  # Clear frames list at the start
    start_time = time.time()  # Record start time for metadata

    # Start the stream using sounddevice
    with sd.InputStream(channels=channels, samplerate=sample_rate, device=device_index, callback=audio_callback):
        while recording:
            sd.sleep(100)  # Small delay to prevent high CPU usage

    # Save the audio to a file
    current_timestamp_datetime = datetime.fromtimestamp(datetime.now().timestamp()).strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f"output_{current_timestamp_datetime}.wav"
    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # Sample width for int16 is 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    end_time = time.time()  # Record end time for metadata
    duration = end_time - start_time

    # Save metadata to InfluxDB
    point = Point("audio_metadata") \
        .tag("file_name", file_name) \
        .tag("user_name", user_name) \
        .field("duration", duration) \
        .field("sample_rate", sample_rate) \
        .field("channels", channels) \
        .field("start_time", int(start_time * 1e9))  # InfluxDB expects time in nanoseconds
    write_api.write(bucket=INFLUXDB_BUCKET, record=point)

    frames = []  # Clear frames after saving

@app.route('/')
def index():
    return render_template('start.html')

@app.route('/start', methods=['POST'])
def start():
    global recording, user_name
    user_name = request.form.get('user_name')  # Get the username from the form
    recording = True
    threading.Thread(target=record_audio).start()
    return render_template('stop.html')

@app.route('/stop', methods=['POST'])
def stop():
    global recording
    recording = False
    return render_template('start.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
