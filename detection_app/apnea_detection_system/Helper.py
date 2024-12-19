import pandas as pd
from influxdb_client import InfluxDBClient
from datetime import timedelta


class Helper:
    def __init__(self, url, token, org, field, fs):
        self.url = url
        self.token = token
        self.org = org
        self.field = field
        self.fs = fs

    def get_data(self, bucket, measurement, source='local'):
        client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        query_api = client.query_api()

        if source == 'influx':
            try:
                flux_query = f'''
                    from(bucket: "{bucket}")
                    |> range(start: -{24}h)
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
                print(f"Error in get_data: {e}")
                print("Reading data from PulseRate_last_session.csv")
                df = pd.read_csv(f'{self.field}_last_session.csv')
                df['time'] = pd.to_datetime(df['time'], format="mixed")
                df = df.sort_values('time').reset_index(drop=True)
                return df
            finally:
                client.close()

        elif source == 'local':
            print("Reading data from PulseRate_last_session.csv")
            df = pd.read_csv('PulseRate_last_session.csv')
            df['time'] = pd.to_datetime(df['time'], format="mixed")
            df = df.sort_values('time').reset_index(drop=True)
            return df
        else:
            raise ValueError("Invalid source specified. Use 'influx' or 'local'.")

    def verify_sampling_rate(self, df):
        total_duration = (df['time'].iloc[-1] - df['time'].iloc[0]).total_seconds()
        num_samples = len(df)
        overall_sampling_rate = num_samples / total_duration

        print(f"Overall sampling rate: {overall_sampling_rate:.2f} Hz")

        tolerance = 0.1 * self.fs
        if abs(overall_sampling_rate - self.fs) > tolerance:
            print(
                f"Overall sampling rate verification failed. Expected: {self.fs} Hz, Found: {overall_sampling_rate:.2f} Hz")
        else:
            print("Sampling rate verified successfully.")