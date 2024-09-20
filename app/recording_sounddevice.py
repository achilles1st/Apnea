from flask import Flask, render_template, request, redirect, url_for
import sounddevice as sd
import threading
import os
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
import time
from influxdb_client.domain.write_precision import WritePrecision
from serial.tools import list_ports
from pulseoxy.pulox_250_realtime import CMS50Dplus
import wave
import datetime
import numpy as np
import sys
from influxdb_client import WriteOptions
from queue import Queue, Empty


app = Flask(__name__)

recording_event = threading.Event()  # Event to signal when to stop recording
threads = []  # List to track active threads

channels = 1
sample_rate = 44100
device_index = 1  # for windows, use 1

# InfluxDB connection settings
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "9RdNMHQLhHuujCeLFCJpunRbkOOq7PDN2RoxW55-sl8x3_CBPuaHvLXQxL9QqJfdv7Kod82hnrPJkZi8xVUfjQ=="
INFLUXDB_ORG = "TU"
INFLUXDB_BUCKET = "audio"

RECORDINGS_DIR = 'C:/Users/tosic/Arduino_projects/sensor_com/app/data/recordings'

# Initialize InfluxDB client
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=WriteOptions(
    write_type=ASYNCHRONOUS,
    batch_size=1000,
    flush_interval=1000,
    jitter_interval=0,
    retry_interval=5000,
    max_retries=3,
    max_retry_delay=30000,
    exponential_base=2
))

data_queue = Queue(maxsize=10000)  # Adjust the maxsize as needed

user_name = ""  # Global variable to store the username

error_event = threading.Event()
error_message = ""

# Initialize an empty NumPy array
audio_data = np.array([], dtype='int16')

def audio_callback(indata, frames_count, time_info, status):
    global user_name, audio_data
    if recording_event.is_set():
        if status:
            print(f"Status: {status}")
        audio_data = np.append(audio_data, indata.copy())


def record_audio():
    global error_message
    try:
        # Start the stream using sounddevice with specified device index
        with sd.InputStream(device=device_index, channels=channels, samplerate=sample_rate, dtype='int16', callback=audio_callback):
            while recording_event.is_set():
                sd.sleep(100)  # Small delay to prevent high CPU usage

    except Exception as e:
        error_message = f"Error recording audio: {e}"
        print(error_message)
        error_event.set()
        recording_event.clear()

def write_to_influxdb():
    global error_message
    batch_points = []
    while recording_event.is_set() or not data_queue.empty():
        try:
            datapoint = data_queue.get(timeout=1)  # Wait for data
            point = Point("pulseoxy_samples") \
                .tag("user_name", user_name) \
                .field("spO2", int(datapoint.spO2)) \
                .field("PulseWave", int(datapoint.pulse_waveform)) \
                .field("PulseRate", int(datapoint.pulse_rate)) \
                .time(int(datapoint.time.timestamp() * 1e9), WritePrecision.NS)
            batch_points.append(point)
            data_queue.task_done()

            # Write in batches
            if len(batch_points) >= 60:
                write_api.write(bucket="Pulseoxy", org=INFLUXDB_ORG, record=batch_points)
                batch_points = []
        except Empty:
            continue
        except Exception as e:
            error_message = f"Error writing to InfluxDB: {e}"
            print(error_message)
            error_event.set()
            recording_event.clear()
            break

    # Write any remaining points
    if batch_points:
        write_api.write(bucket="Pulseoxy", org=INFLUXDB_ORG, record=batch_points)

def record_oximeter():
    global error_message

    try:
        port = list_ports.comports()[0].device  # Select the first available serial port
        oximeter = CMS50Dplus(port)
    except Exception as e:
        error_message = f"Error initializing oximeter: {e}"
        print(error_message)
        error_event.set()
        recording_event.clear()
        return

    while recording_event.is_set():
        try:
            datapoints = oximeter.get_realtime_data()
            for datapoint in datapoints:
                sys.stdout.write(
                    "\rSignal: {:>2}"
                    " | PulseRate: {:>3}"
                    " | PulseWave: {:>3}"
                    " | SpO2: {:>2}%"
                    " | ProbeError: {:>1}".format(
                        datapoint.signal_strength,
                        datapoint.pulse_rate,
                        datapoint.pulse_waveform,
                        datapoint.spO2,
                        datapoint.probe_error))
                sys.stdout.flush()

                if not recording_event.is_set():
                    break  # Stop processing if recording is stopped

                # Put the datapoint into the queue
                data_queue.put(datapoint)
        except Exception as e:
            error_message = f"Error reading data from oximeter: {e}"
            print(error_message)
            error_event.set()
            recording_event.clear()
            break

@app.route('/')
def index():
    global error_message
    error_event.clear()  # Reset the error event
    error_message = ""  # Reset the error message
    return render_template('/Start.html')


@app.route('/start', methods=['POST'])
def start():
    global user_name, error_message
    error_event.clear()  # Reset the error event
    error_message = ""  # Reset the error message

    user_name = request.form.get('user_name')  # Get the username from the form

    recording_event.set()  # Set the recording event to start
    audio_thread = threading.Thread(target=record_audio, daemon=True)
    oximeter_thread = threading.Thread(target=record_oximeter, daemon=True)
    influxdb_thread = threading.Thread(target=write_to_influxdb, daemon=True)

    audio_thread.start()
    oximeter_thread.start()
    influxdb_thread.start()

    threads.extend([audio_thread, oximeter_thread, influxdb_thread])  # Add threads to the list
    return render_template('/Stop.html')


@app.route('/stop', methods=['POST'])
def stop():
    global audio_data, user_name, error_event, error_message

    recording_event.clear()  # Signal threads to stop
    data_queue.join()  # Wait until all data has been processed
    write_api.flush()  # Flush any remaining data to InfluxDB

    # Wait for threads to finish
    for thread in threads:
        thread.join()

    # Save audio data only if no error occurred
    if not error_event.is_set():
        # save audio data to a file
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(RECORDINGS_DIR, f"{user_name}_{timestamp}.wav")

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
    else:
        error_message = f"Error saving audio file!"
        return redirect(url_for('/error'))


    threads.clear()  # Clear the list of threads

    # Reset the error event and message for future recordings
    error_event.clear()
    error_message = ""

    return render_template('/Start.html')

@app.route('/check_error')
def check_error():
    if error_event.is_set():
        return {'error': True, 'message': error_message}
    else:
        return {'error': False}

@app.route('/error')
def error():
    global error_message
    return render_template('error.html', error_message=error_message)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
