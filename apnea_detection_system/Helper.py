from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.domain.write_precision import WritePrecision
from influxdb_client.client.write_api import ASYNCHRONOUS
import pandas as pd
from datetime import timedelta
import warnings


class Helper:
    def __init__(self, url, token, org, field, fs):
        self.url = url
        self.token = token
        self.org = org
        self.field = field
        self.fs = fs

    def get_data(self, bucket, measurement, source='local'):
        '''
        gets the data from the last session within the last 24h or can load saved .csv file <field>_last_session.csv
        '''

        client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        query_api = client.query_api()

        if source == 'influx':
            try:
                flux_query = f'''
                    from(bucket: "{bucket}")
                    |> range(start: -{48}h)
                    |> filter(fn: (r) => r._measurement == "{measurement}")
                    |> filter(fn: (r) => r._field == "{self.field}")
                '''
                result = query_api.query(flux_query)

                data_points = []
                for table in result:
                    for record in table.records:
                        data_points.append({
                            'time': record.get_time(),
                            f'{self.field}': record.get_value()
                        })

                gap_threshold = timedelta(minutes=1)
                sessions = []
                current_session = []

                for i in range(len(data_points)):
                    if i == 0:
                        current_session.append(data_points[i])
                    else:
                        time_diff = data_points[i]['time'] - data_points[i - 1]['time']
                        if time_diff <= gap_threshold:
                            current_session.append(data_points[i])
                        else:
                            sessions.append(current_session)
                            current_session = [data_points[i]]

                if current_session:
                    sessions.append(current_session)

                if not sessions:
                    raise Exception("No data found in the last 24 hours.")
                else:
                    last_session = sessions[-1]
                    print(f"Found {len(sessions)} sessions. Last session has {len(last_session)} data points.")
                    df = pd.DataFrame(last_session)
                    return df

            except Exception as e:
                warnings.warn(f"Error in get_data: {e}\n Reading data from {self.field}_last_session.csv", RuntimeWarning)
                print(f"Reading data from {self.field}_last_session.csv")
                df = pd.read_csv(f'{self.field}_last_session.csv')
                if pd.api.types.is_numeric_dtype(df['time']):
                    # Get the current date as the origin
                    current_date = pd.Timestamp.now()
                    # Convert `time` to timedelta (in seconds) and add to the origin
                    df["time"] = current_date + pd.to_timedelta(df["time"], unit="s")
                    df = df.sort_values('time').reset_index(drop=True)
                else:
                    df['time'] = pd.to_datetime(df['time'], format="mixed")
                    df = df.sort_values('time').reset_index(drop=True)
                return df
            finally:
                client.close()

        elif source == 'local':
            print(f"Reading data from {self.field}_last_session.csv")
            df = pd.read_csv(f'{self.field}_last_session.csv')
            if pd.api.types.is_numeric_dtype(df['time']):
                # Get the current date as the origin
                current_date = pd.Timestamp.now()
                # Convert `time` to timedelta (in seconds) and add to the origin
                df["time"] = current_date + pd.to_timedelta(df["time"], unit="s")
                df = df.sort_values('time').reset_index(drop=True)
            else:
                df['time'] = pd.to_datetime(df['time'], format="mixed")
                df = df.sort_values('time').reset_index(drop=True)
            return df
        else:
            raise ValueError("Invalid source specified. Use 'influx' or 'local'.")

    def upload_apnea_events(self, events, measurement_name, user_name, bucket):
        client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
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

        points = []
        for start, stop in events:
            start_point = Point(f"{measurement_name}") \
                .tag("user_name", f"{user_name}") \
                .field(f"start_{self.field}_event", int(1)) \
                .time(int(start.timestamp() * 1e9), WritePrecision.NS)
            points.append(start_point)
            # Add a duration field or another marker for the stop time
            stop_point = Point(f"{measurement_name}") \
                .tag("user_name", f"{user_name}") \
                .field(f"stop_{self.field}_event", int(0)) \
                .time(int(stop.timestamp() * 1e9), WritePrecision.NS)
            points.append(stop_point)

        # Write points to InfluxDB
        write_api.write(bucket=bucket, record=points)
        print(f"{measurement_name}: Apnea events uploaded successfully.")

        client.close()
        write_api.flush()

    def upload_ahi_index(self, ahi, total_sleep_time_hours, measurement_name, user_name, bucket):
        client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        write_api = client.write_api(write_options=WriteOptions(
            write_type=ASYNCHRONOUS,
            batch_size=1,
            flush_interval=1000,
            jitter_interval=0,
            retry_interval=5000,
            max_retries=3,
            max_retry_delay=30000,
            exponential_base=2
        ))

        point = Point(measurement_name) \
            .tag("user_name", user_name) \
            .field("ahi_value", float(ahi)) \
            .field("total_sleep_time_hours", float(total_sleep_time_hours)) \
            .time(int(pd.Timestamp.utcnow().timestamp() * 1e9), WritePrecision.NS)

        # Write point to InfluxDB
        write_api.write(bucket=bucket, record=point)
        print("AHI index uploaded successfully.")

        # Flush pending writes before closing the client
        write_api.flush()
        client.close()

    @staticmethod
    def get_snoring_event_timestamps(url, token, org, bucket, start="-24h"):
        """
        Retrieves the timestamps of snoring events from InfluxDB for a given user.
        """
        # Initialize InfluxDB client
        client = InfluxDBClient(url=url, token=token, org=org)

        # Flux query to fetch snoring event timestamps
        query = f"""
        from(bucket: "{bucket}")
          |> range(start: {start})
          |> filter(fn: (r) => r._measurement == "snoring_events")
          |> filter(fn: (r) => r._field == "probability")
          |> keep(columns: ["_time"])
        """

        # Execute query
        query_api = client.query_api()
        tables = query_api.query(query)

        # Extract timestamps
        timestamps = [pd.Timestamp(record["_time"]) for table in tables for record in table.records]

        # Close the InfluxDB client connection
        client.close()

        return timestamps

    @staticmethod
    def verify_sampling_rate(df, fs):
        # Ensure the 'time' column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'], format="mixed")

        if isinstance(df['time'].iloc[0], pd.Timestamp):
            total_duration = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()
        else:
            total_duration = df['time'].iloc[-1] - df['time'].iloc[0]

        num_samples = len(df)
        overall_sampling_rate = num_samples / total_duration

        print(f"Overall sampling rate: {overall_sampling_rate:.2f} Hz")

        tolerance = 0.1 * fs
        if abs(overall_sampling_rate - fs) > tolerance:
            print(
                f"Overall sampling rate verification failed. Expected: {fs} Hz, Found: {overall_sampling_rate:.4f} Hz")
        else:
            print("Sampling rate verified successfully.")

        return overall_sampling_rate