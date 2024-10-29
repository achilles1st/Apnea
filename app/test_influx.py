import numpy as np
import sounddevice as sd
from influxdb_client import InfluxDBClient, QueryApi
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.client.flux_table import FluxStructureEncoder
import wave
import time

# InfluxDB connection settings
INFLUXDB_URL = "http://192.168.0.24:8086"
INFLUXDB_TOKEN = "v2tyk3MkEpDdKCopMIkUlRBSo66tJVkomdZ_7i71SEjFd44qAnr4dWjmi5kssQ6nRc-91j0Mz8vBa8R4PHgkdQ=="
INFLUXDB_ORG = "TU"
INFLUXDB_BUCKET = "audio"

# Initialize InfluxDB client
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()


# Query to fetch audio samples from InfluxDB
def fetch_audio_data(user_name, start_time, end_time):
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: {start_time}, stop: {end_time})
      |> filter(fn: (r) => r["_measurement"] == "audio_samples")
      |> filter(fn: (r) => r["user_name"] == "{user_name}")
      |> sort(columns: ["_time"])
      |> keep(columns: ["_time", "_value"])
    '''
    result = query_api.query(query)

    # Extract amplitude values and sort by time
    amplitudes = [[record.get_value()] for table in result for record in table.records]
    return np.array(amplitudes, dtype=np.int16)


# Function to play audio data using sounddevice
def save_audio(frames):
    file_name = f"output_influx.wav"
    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # Sample width for int16 is 2 bytes
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))


# Main function to retrieve and play audio
def main():
    user_name = "Stefan"  # Replace with actual user name
    start_time = "-5m"  # Example: Fetch data from the last hour
    end_time = "now()"  # Fetch data up to the current time
    sample_rate = 44100  # Ensure this matches the original recording rate

    # Fetch audio data from InfluxDB
    audio_data = fetch_audio_data(user_name, start_time, end_time)

    # save the fetched audio data
    save_audio(audio_data)


if __name__ == "__main__":
    main()
