from flask import Flask, render_template, request, redirect, url_for
import sounddevice as sd
import threading
import os
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
from influxdb_client.domain.write_precision import WritePrecision
from serial.tools import list_ports
from pulseoxy.pulox_250_realtime import CMS50Dplus
import datetime
import configparser
from influxdb_client import WriteOptions
from queue import Queue, Empty
import time
import numpy as np
from check_recordings import SnoringDetector
# imports for raspberry pi i2c communication
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from adafruit_ads1x15.ads1x15 import Mode
import logging
from logging.handlers import RotatingFileHandler
import asyncio
from bleak import BleakScanner, BleakClient
import math


app = Flask(__name__)

def find_device_index(device_name):
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device_name in device['name']:
            logger.debug(f"Found device '{device_name}' at index {idx}")
            return int(idx)
    print(f"Device '{device_name}' not found")
    return 0

# Configure Logging
def setup_logging():
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create log directory if it doesn't exist
    log_directory = 'logs'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Formatter for logs
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler with Rotation
    file_handler = RotatingFileHandler(
        os.path.join(log_directory, 'app.log'),
        maxBytes=20*1024*1024,  # 20 MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Suppress overly verbose logs from third-party libraries
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('influxdb_client').setLevel(logging.WARNING)
    logging.getLogger('sounddevice').setLevel(logging.WARNING)
    logging.getLogger('adafruit_ads1x15').setLevel(logging.WARNING)

# Initialize logging
setup_logging()

# logger for this module
logger = logging.getLogger(__name__)

ble_connected_event = threading.Event()
ble_active = False

recording_event = threading.Event()  # Event to signal when to stop recording
threads = []  # List to track active threads

channels = 1
sample_rate = 44100
device_index = find_device_index("USB PnP Sound Device: Audio (hw:2,0)")  # for windows, use 1

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# InfluxDB connection settings
INFLUXDB_URL = config['INFLUXDB']['URL']
INFLUXDB_TOKEN = config['INFLUXDB']['TOKEN']
INFLUXDB_ORG = config['INFLUXDB']['ORG']
INFLUXDB_BUCKET = config['INFLUXDB']['AUDIO_BUCKET']
ECG_BUCKET = config['INFLUXDB']['ECG_BUCKET']
PULSEOXY_BUCKET = config['INFLUXDB']['PULSEOXY_BUCKET']
RESPIRATORY_BUCKET = config['INFLUXDB']['RESPIRATORY_BUCKET']

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

data_queue = Queue(maxsize=10000)  # For pulse oximeter data
ecg_data_queue = Queue(maxsize=10000)  # For ECG data
audio_queue = Queue(maxsize=10000)  # For audio segments
respiratory_data_queue = Queue(maxsize=10000)

user_name = ""  # Global variable to store the username

error_event = threading.Event()
error_message = ""

# Initialize SnoringDetector
snoring_detector = SnoringDetector()

# Define the UUID for the BLE characteristic
RESP_BELT_CHAR_UUID = "4fafc202-1fb5-459e-8fcc-c5c9c331914b"
BLE_DEVICE_NAME = "ESP32_BLE"  # Adjust this to match your device name
DATA_BATCH_SIZE = 60  # Adjust batch size for writing to InfluxDB as needed

def write_oxy_to_influxdb():
    global error_message

    if ble_active:
        ble_connected_event.wait()  # Wait until BLE is connected
    batch_points = []
    logger.info("Starting write_oxy_to_influxdb thread.")
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
                write_api.write(bucket=PULSEOXY_BUCKET, org=INFLUXDB_ORG, record=batch_points)
                batch_points = []
        except Empty:
            continue
        except Exception as e:
            error_message = f"Error writing to InfluxDB: {e}"
            logger.error(error_message, exc_info=True)
            error_event.set()
            recording_event.clear()
            break

    # Write any remaining points
    if batch_points:
        write_api.write(bucket=PULSEOXY_BUCKET, org=INFLUXDB_ORG, record=batch_points)
        logger.debug(f"Wrote remaining {len(batch_points)} pulse oximeter points to InfluxDB.")

    logger.info("write_oxy_to_influxdb thread has stopped.")


def write_ecg_to_influxdb():
    global error_message

    if ble_active:
        ble_connected_event.wait()  # Wait until BLE is connected
    batch_points = []
    logger.info("Starting write_ecg_to_influxdb thread.")
    while recording_event.is_set() or not ecg_data_queue.empty():
        try:
            data_point = ecg_data_queue.get(timeout=1)  # Wait for data
            point = Point("ecg_samples") \
                .tag("user_name", user_name) \
                .field("ecg_value", data_point['ecg_value']) \
                .time(int(data_point['time'].timestamp() * 1e9), WritePrecision.NS)
            batch_points.append(point)
            ecg_data_queue.task_done()

            # Write in batches
            if len(batch_points) >= 128:
                write_api.write(bucket=ECG_BUCKET, org=INFLUXDB_ORG, record=batch_points)
                batch_points = []
        except Empty:
            continue
        except Exception as e:
            error_message = f"Error writing ECG data to InfluxDB: {e}"
            logger.error(error_message, exc_info=True)
            error_event.set()
            recording_event.clear()
            break

    # Write any remaining points
    if batch_points:
        write_api.write(bucket=ECG_BUCKET, org=INFLUXDB_ORG, record=batch_points)

def calculate_pitch_and_roll(accX, accY, accZ):
    pitch = math.atan2(accY, math.sqrt(accX**2 + accZ**2)) * (180.0 / math.pi)
    roll = math.atan2(-accX, accZ) * (180.0 / math.pi)
    return pitch, roll

def notification_handler(sender, data):
    global user_name
    # Decode data (assuming the data is simple text
    try:
        decoded_data = data.decode('utf-8')
    except UnicodeDecodeError:
        decoded_data = data.decode('latin1', errors='replace')

    timestamp = datetime.datetime.now(datetime.timezone.utc)

    entry = f"{timestamp}, {decoded_data}\n"
    semicolon_index = entry.find(';')
    result = entry[:semicolon_index] if semicolon_index != -1 else decoded_data
    values = result.split(',')

    accX = accY = accZ = analogValue = None

    for value in values:
        value = value.strip()
        if 'accX' in value:
            accX = float(value.split(':')[1].strip())
        elif 'accY' in value:
            accY = float(value.split(':')[1].strip())
        elif 'accZ' in value:
            accZ = float(value.split(':')[1].strip())
        elif 'analogValue' in value:
            analogValue = int(value.split(':')[1].strip())

    # Compute pitch and roll if acceleration data is available
    if accX is not None and accY is not None and accZ is not None:
        pitch, roll = calculate_pitch_and_roll(accX, accY, accZ)
    else:
        pitch, roll = None, None

    # Determine sleeping position
    if pitch is not None and roll is not None:
        if abs(roll) < 15 and abs(pitch) < 15:
            position = 1  # flat
        elif 65 < roll < 100:
            position = 2  # left
        elif -100 < roll < -65:
            position = 3  # right
        elif 150 < abs(pitch) < 200:
            position = 4  # flat st
        else:
            position = 5  # unknown
    else:
        position = 5  # Default to unknown if pitch/roll not available

    # Store data in queue
    data_point = {
        'time': timestamp,
        'resp_value': analogValue,
        'position': position
    }

    # Put data into the queue
    #respiratory_data_queue.put_nowait(data_point)
    respiratory_data_queue.put(data_point)

async def ble_run():
    global error_message

    devices = await BleakScanner.discover()
    target_device = None
    for device in devices:
        if device.name == BLE_DEVICE_NAME:
            target_device = device
            break

    if not target_device:
        error_message = f"Could not find device {BLE_DEVICE_NAME}"
        logger.error(error_message)
        error_event.set()
        recording_event.clear()
        return

    logger.info(f"Found device: {target_device.name} [{target_device.address}]")

    try:
        async with BleakClient(target_device.address) as client:
            logger.info("Connected to BLE device.")
            if not client.is_connected:
                raise ConnectionError("Failed to establish BLE connection.")

            await client.start_notify(RESP_BELT_CHAR_UUID, notification_handler)
            logger.info("Started receiving respiratory belt notifications.")
            # Signal that BLE is connected
            ble_connected_event.set()
            try:
                while recording_event.is_set():
                    await asyncio.sleep(1)  # Keeps the connection alive
            except asyncio.CancelledError:
                print("Notification loop cancelled.")

    except asyncio.TimeoutError:
        error_message = "BLE connection attempt timed out."
        logger.error(error_message, exc_info=True)
        error_event.set()
        recording_event.clear()
    except Exception as e:
        error_message = f"Unexpected error during BLE operation: {e}"
        logger.error(error_message, exc_info=True)
        error_event.set()
        recording_event.clear()

def run_ble_loop():
    # Runs BLE asyncio code in a thread
    loop = asyncio.new_event_loop()
    loop.run_until_complete(ble_run())


def write_respiratory_to_influxdb():
    global error_message

    if ble_active:
        ble_connected_event.wait()  # Wait until BLE is connected
    batch_points = []
    logger.info("Starting write_respiratory_to_influxdb thread.")
    while recording_event.is_set() or not respiratory_data_queue.empty():
        try:
            data_point = respiratory_data_queue.get(timeout=1)
            point = Point("resp_belt_samples") \
                .tag("user_name", user_name) \
                .field("resp_value", data_point['resp_value']) \
                .field("position", data_point['position']) \
                .time(int(data_point['time'].timestamp() * 1e9), WritePrecision.NS)
            batch_points.append(point)
            respiratory_data_queue.task_done()

            if len(batch_points) >= 30:  # all 3 seconds update
                write_api.write(bucket=RESPIRATORY_BUCKET, org=INFLUXDB_ORG, record=batch_points)
                batch_points = []
        except Empty:
            continue
        except Exception as e:
            error_message = f"Error writing respiratory data to InfluxDB: {e}"
            logger.error(error_message, exc_info=True)
            error_event.set()
            recording_event.clear()
            break

    # Write remaining points
    if batch_points:
        write_api.write(bucket=RESPIRATORY_BUCKET, org=INFLUXDB_ORG, record=batch_points)

    logger.info("write_respiratory_to_influxdb thread has stopped.")


def process_audio_segments():
    global error_message

    if ble_active:
        ble_connected_event.wait()  # Wait until BLE is connected
    logger.info("Starting audio processing thread.")
    while recording_event.is_set() or not audio_queue.empty():
        try:
            segment = audio_queue.get(timeout=1)  # Wait for audio segment
            snoring_detector.process_audio_segment(segment, user_name)
            audio_queue.task_done()
        except Empty:
            continue
        except Exception as e:
            error_message = f"Error processing audio segment: {e}"
            logger.error(error_message, exc_info=True)
            error_event.set()
            recording_event.clear()
            break

    logger.info("process_audio_segments thread has stopped.")



def record_audio():
    global error_message
    try:
        if ble_active:
            ble_connected_event.wait()  # Wait until BLE is connected
        logger.info("Starting audio recording thread.")
        # Initialize buffer for audio segments
        buffer = np.array([], dtype=np.int16)
        segment_size = sample_rate * snoring_detector.SEGMENT_DURATION  # Number of samples per segment

        # Start the stream using sounddevice with specified device index
        with sd.InputStream(device=device_index, channels=channels, samplerate=sample_rate, dtype='int16') as stream:
            logger.debug("Audio stream opened successfully.")
            while recording_event.is_set():
                indata, _ = stream.read(1024)  # Read 1024 frames
                if indata.size > 0:
                    buffer = np.concatenate((buffer, indata.flatten()))
                    if len(buffer) >= segment_size:
                        # Extract the segment
                        segment = buffer[:segment_size]
                        buffer = buffer[segment_size:]  # Remove the processed segment
                        # Normalize audio to float32 between -1 and 1
                        segment = segment.astype(np.float32) / 32768.0
                        # Put the segment into the queue for processing
                        audio_queue.put(segment)

        # After recording, process any remaining buffer
        if buffer.size > 0:
            # Pad the buffer if necessary
            if len(buffer) < segment_size:
                padding = np.zeros(segment_size - len(buffer), dtype=np.float32)
                segment = np.concatenate((buffer.astype(np.float32) / 32768.0, padding))
            else:
                segment = buffer.astype(np.float32) / 32768.0
            audio_queue.put(segment)
            logger.debug("Queued final audio segment.")

    except Exception as e:
        error_message = f"Error recording audio: {e}"
        logger.error(error_message, exc_info=True)
        error_event.set()
        recording_event.clear()
    finally:
        logger.info("record_audio thread has stopped.")


def record_oximeter():
    global error_message

    try:
        if ble_active:
            ble_connected_event.wait()  # Wait until BLE is connected
        logger.info("Starting oximeter recording thread.")
        port = list_ports.comports()[0].device  # Select the first available serial port
        oximeter = CMS50Dplus(port)
        logger.debug(f"Oximeter initialized on port {port}.")
    except Exception as e:
        error_message = f"Error initializing oximeter: {e}"
        logger.error(error_message, exc_info=True)
        error_event.set()
        recording_event.clear()
        return

    while recording_event.is_set():
        try:
            datapoints = oximeter.get_realtime_data()
            for datapoint in datapoints:

                if not recording_event.is_set():
                    break  # Stop processing if recording is stopped

                # Put the datapoint into the queue
                data_queue.put(datapoint)

        except Exception as e:
            error_message = f"Error reading data from oximeter: {e}"
            logger.error(error_message, exc_info=True)
            error_event.set()
            recording_event.clear()
            break

    logger.info("record_oximeter thread has stopped.")



def record_ecg():
    global error_message, intervals
    try:
        if ble_active:
            ble_connected_event.wait()  # Wait until BLE is connected
        logger.info("Starting ECG recording thread.")
        # Initialize I2C bus and ADS1115 ADC
        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS.ADS1115(i2c)
        ads.gain = 1
        ads.data_rate = 128  # 128 samples per second
        ads.mode = Mode.CONTINUOUS  # Set continuous conversion mode
        ecg_channel = AnalogIn(ads, ADS.P0)
        logger.debug("ECG sensor initialized successfully.")
    except Exception as e:
        error_message = f"Error initializing ECG sensor: {e}"
        logger.error(error_message, exc_info=True)
        error_event.set()
        recording_event.clear()
        return

    last_sample_time = None  # To store the time of the last sample
    sampling_interval = 1.0 / ads.data_rate  # Calculate expected interval (7.8125 ms)

    while recording_event.is_set():
        try:
            # Wait for the next sample based on the sampling interval
            if last_sample_time is not None:
                elapsed_time = (datetime.datetime.now(datetime.timezone.utc) - last_sample_time).total_seconds()
                remaining_time = sampling_interval - elapsed_time
                if remaining_time > 0:
                    time.sleep(remaining_time)

            # Read the voltage value from the ADC
            current_time = datetime.datetime.now(datetime.timezone.utc)
            ecg_value = ecg_channel.voltage

            last_sample_time = current_time

            # Create a data point
            data_point = {
                'time': current_time,
                'ecg_value': ecg_value
            }

            # Put the data point into the queue
            ecg_data_queue.put(data_point)

        except Exception as e:
            error_message = f"Error reading ECG data: {e}"
            logger.error(error_message, exc_info=True)
            error_event.set()
            recording_event.clear()
            break

    logger.info("record_ecg thread has stopped.")


@app.route('/')
def index():
    global error_message
    logger.info("Accessed the index page.")
    error_event.clear()  # Reset the error event
    error_message = ""  # Reset the error message
    return render_template('/Start.html')


@app.route('/start', methods=['POST'])
def start():
    global user_name, error_message, ble_active

    logger.info("Received request to start recording.")
    error_event.clear()  # Reset the error event
    error_message = ""  # Reset the error message
    ble_connected_event.clear()  # Reset the BLE connection event

    user_name = request.form.get('user_name')  # Get the username from the form
    selected_sensors = request.form.getlist('sensors')  # Get selected sensors as a list
    logger.debug(f"Recording started for user: {user_name} with sensors: {selected_sensors}")

    recording_event.set()  # Set the recording event to start

    # Initialize and start threads based on selected sensors
    sensor_threads = []

    if 'respiration' in selected_sensors:
        ble_active = True
        resp_influxdb_thread = threading.Thread(target=write_respiratory_to_influxdb, daemon=True, name='RespInfluxdbWriteThread')
        ble_thread = threading.Thread(target=run_ble_loop, daemon=True, name='BLEThread')
        sensor_threads.append(resp_influxdb_thread)
        sensor_threads.append(ble_thread)
        resp_influxdb_thread.start()
        ble_thread.start()

    if 'snoring' in selected_sensors:
        audio_thread = threading.Thread(target=record_audio, daemon=True, name='AudioThread')
        audio_processing_thread = threading.Thread(target=process_audio_segments, daemon=True, name='AudioProcessingThread')
        sensor_threads.append(audio_thread)
        sensor_threads.append(audio_processing_thread)
        audio_thread.start()
        audio_processing_thread.start()

    if 'oximeter' in selected_sensors:
        oximeter_thread = threading.Thread(target=record_oximeter, daemon=True, name='OximeterThread')
        influxdb_thread = threading.Thread(target=write_oxy_to_influxdb, daemon=True, name='OxygenInfluxdbWriteThread')
        sensor_threads.append(oximeter_thread)
        sensor_threads.append(influxdb_thread)
        oximeter_thread.start()
        influxdb_thread.start()

    if 'ecg' in selected_sensors:
        ecg_thread = threading.Thread(target=record_ecg, daemon=True, name='ECGThread')
        ecg_influxdb_thread = threading.Thread(target=write_ecg_to_influxdb, daemon=True, name='ECGInfluxdbWriteThread')
        sensor_threads.append(ecg_thread)
        sensor_threads.append(ecg_influxdb_thread)
        ecg_thread.start()
        ecg_influxdb_thread.start()

    if not selected_sensors:
        error_message = "No sensors selected. Please select at least one sensor."
        logger.error(error_message)
        return render_template('error.html', error_message=error_message)

    threads.extend(sensor_threads)

    logger.info(f"Selected sensor threads have started: {selected_sensors}")

    return render_template('/Stop.html')


@app.route('/stop', methods=['POST'])
def stop():
    global user_name, error_event, error_message, ble_active

    logger.info("Received request to stop recording.")
    recording_event.clear()  # Signal threads to stop
    ble_connected_event.set()  # Unblock any threads waiting on BLE connection
    ble_active = False  # Reset BLE active flag
    logger.debug("Signaled all threads to stop and unblocked BLE event.")

    # Wait until all data has been processed
    try:
        data_queue.join()
        ecg_data_queue.join()
        audio_queue.join()
        respiratory_data_queue.join()
        write_api.flush()
        logger.debug("All queues have been joined and InfluxDB has been flushed.")
    except Exception as e:
        logger.error(f"Error during stopping process: {e}", exc_info=True)

    # Wait for threads to finish
    for thread in threads:
        logger.debug(f"Waiting for thread {thread.name} to finish.")
        thread.join()
        logger.debug(f"Thread {thread.name} has finished.")

    threads.clear()  # Clear the list of threads
    logger.info("All recording threads have been stopped and cleared.")

    # Close SnoringDetector's InfluxDB client
    snoring_detector.close()
    logger.debug("SnoringDetector's InfluxDB client has been closed.")

    # Reset the error event and message for future recordings
    error_event.clear()
    error_message = ""
    logger.info("Recording state has been reset.")

    return render_template('/Start.html')


@app.route('/check_error')
def check_error():
    if error_event.is_set():
        logger.warning(f"Error detected: {error_message}")
        return {'error': True, 'message': error_message}
    else:
        return {'error': False}


@app.route('/error')
def error():
    global error_message
    logger.info("Accessed the error page.")
    return render_template('error.html', error_message=error_message)


if __name__ == '__main__':
    logger.info("Starting Flask application.")
    app.run(host='0.0.0.0', port=5000)
