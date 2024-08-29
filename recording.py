from flask import Flask, render_template, request
import pyaudio
import threading
import wave
import numpy as np
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import time

app = Flask(__name__)

recording = False
stream = None
amplification_factor = 4  # Amplification factor (e.g., 2.0 will double the loudness)

# InfluxDB connection settings
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "your-influxdb-token"
INFLUXDB_ORG = "your-org"
INFLUXDB_BUCKET = "your-bucket"

# Initialize InfluxDB client
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)


def record_audio():
    global stream, recording, p, frames

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=2, rate=44100, input=True, frames_per_buffer=1024,
                    input_device_index=2)

    frames = []
    start_time = time.time()  # Record start time for metadata

    while recording:
        data = stream.read(1024)
        # Convert the byte data to numpy array for amplification
        audio_data = np.frombuffer(data, dtype=np.int16)
        # Amplify the audio data
        amplified_data = np.clip(audio_data * amplification_factor, -32768, 32767)
        # Convert back to bytes and append to frames
        frames.append(amplified_data.astype(np.int16).tobytes())

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the audio to a file
    file_name = "output3.wav"
    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))

    end_time = time.time()  # Record end time for metadata
    duration = end_time - start_time

    # Save metadata to InfluxDB
    point = Point("audio_metadata") \
        .tag("file_name", file_name) \
        .field("duration", duration) \
        .field("sample_rate", 44100) \
        .field("channels", 2) \
        .field("amplification_factor", amplification_factor) \
        .time(int(start_time * 1e9))  # InfluxDB expects time in nanoseconds
    write_api.write(bucket=INFLUXDB_BUCKET, record=point)

    frames = []  # Clear frames after saving


@app.route('/')
def index():
    return render_template('start.html')


@app.route('/start', methods=['POST'])
def start():
    global recording
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
