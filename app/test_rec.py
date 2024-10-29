from flask import Flask, render_template, request
import sounddevice as sd
import threading
import os
import wave
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import time
from influxdb_client.domain.write_precision import WritePrecision
from serial.tools import list_ports
from pulseoxy.pulox_250_realtime import CMS50Dplus
from queue import Queue


app = Flask(__name__)

recording = False
channels = 1
sample_rate = 44100
device_index = 1  # for windows, use 1

# InfluxDB connection settings
INFLUXDB_URL = "http://localhost:8086"  # for windows
INFLUXDB_TOKEN ="9RdNMHQLhHuujCeLFCJpunRbkOOq7PDN2RoxW55-sl8x3_CBPuaHvLXQxL9QqJfdv7Kod82hnrPJkZi8xVUfjQ=="
# INFLUXDB_URL = "http://influxdb:8086"  # for raspi
# INFLUXDB_TOKEN ="v2tyk3MkEpDdKCopMIkUlRBSo66tJVkomdZ_7i71SEjFd44qAnr4dWjmi5kssQ6nRc-91j0Mz8vBa8R4PHgkdQ=="  # for raspi
INFLUXDB_ORG = "TU"
INFLUXDB_BUCKET = "audio"

RECORDINGS_DIR = 'C:/Users/tosic/Arduino_projects/sensor_com/app/data/recordings'

# Initialize InfluxDB client
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)

user_name = ""  # Global variable to store the username

# Thread-safe queues for audio and oximeter data
audio_queue = Queue()
oximeter_queue = Queue()


def audio_callback(indata, frames_count, time_info, status):
    if status:
        print(f"Status: {status}")
    if recording:
        try:
            # Enqueue the raw data for asynchronous processing
            audio_queue.put(indata.copy())
        except Exception as e:
            print(f"Exception in audio_callback: {e}")


def audio_writer():
    global user_name

    # Create the filename based on username and timestamp
    filename = os.path.join(RECORDINGS_DIR, f"{user_name}_{int(time.time())}.wav")

    # Open the file in write mode
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)

        # Process and save the audio data
        while recording or not audio_queue.empty():
            try:
                indata = audio_queue.get(timeout=1)
                wf.writeframes(indata.tobytes())
            except Exception as e:
                print(f"Exception in audio_writer: {e}")


def record_audio():
    global recording

    # Start the audio writer thread
    threading.Thread(target=audio_writer).start()

    # Start the stream using sounddevice with specified device index and adjusted blocksize
    with sd.InputStream(device=device_index, channels=channels, samplerate=sample_rate,
                        dtype='int16', callback=audio_callback):
        while recording:
            sd.sleep(100)  # Small delay to prevent high CPU usage


def oximeter_writer():
    write_api = client.write_api(write_options=SYNCHRONOUS)
    while recording or not oximeter_queue.empty():
        try:
            datapoint = oximeter_queue.get(timeout=1)
            point = Point("pulseoxy_samples") \
                .tag("user_name", user_name) \
                .field("spO2", int(datapoint.spO2)) \
                .field("PulseWave", int(datapoint.pulse_waveform)) \
                .field("PulseRate", int(datapoint.pulse_rate)) \
                .time(int(datapoint.time.timestamp() * 1e9), WritePrecision.NS)
            write_api.write(bucket="Pulseoxy", org=INFLUXDB_ORG, record=point)
        except Exception as e:
            print(f"Exception in oximeter_writer: {e}")

def record_oximeter():
    global recording
    port = list_ports.comports()[0].device  # Select the first available serial port

    # Start the oximeter writer thread
    threading.Thread(target=oximeter_writer).start()

    while recording:
        try:
            oximeter = CMS50Dplus(port)
            datapoints = oximeter.get_realtime_data()

            for datapoint in datapoints:
                # Enqueue the data for asynchronous writing
                oximeter_queue.put(datapoint)
        except Exception as e:
            print(f"Exception in record_oximeter: {e}")

@app.route('/')
def index():
    return render_template('/Start.html')

@app.route('/start', methods=['POST'])
def start():
    global recording, user_name
    user_name = request.form.get('user_name')  # Get the username from the form
    recording = True
    threading.Thread(target=record_audio).start()
    threading.Thread(target=record_oximeter).start()  # Start oximeter recording
    return render_template('/Stop.html')

@app.route('/stop', methods=['POST'])
def stop():
    global recording
    recording = False
    return render_template('/Start.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
