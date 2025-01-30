"""
For Debugging purposes only!!!
This script is used to extract the ECG data from the InfluxDB and save it to a CSV file.
"""

import datetime
import configparser
from influxdb_client import InfluxDBClient
import csv
from datetime import timedelta
import pandas as pd
import plotly.express as px


def main():
    # Load configuration from config.ini
    config = configparser.ConfigParser()
    config.read('config.ini')

    # InfluxDB connection settings
    INFLUXDB_URL = config['INFLUXDB']['URL']
    INFLUXDB_TOKEN = config['INFLUXDB']['TOKEN']
    INFLUXDB_ORG = config['INFLUXDB']['ORG']
    ECG_BUCKET = config['INFLUXDB']['ECG_BUCKET']

    # Initialize InfluxDB client
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
    query_api = client.query_api()

    # Flux query to get ECG data from the last 24 hours
    flux_query = f'''
    from(bucket: "{ECG_BUCKET}")
      |> range(start: -24h)
      |> filter(fn: (r) => r._measurement == "ecg_samples")
      |> filter(fn: (r) => r._field == "ecg_value")
      |> sort(columns: ["_time"])
    '''

    result = query_api.query(flux_query)

    data_points = []

    for table in result:
        for record in table.records:
            data_points.append({
                'time': record.get_time(),
                'value': record.get_value()
            })

    gap_threshold = timedelta(minutes=1)

    sessions = []
    current_session = []

    for i in range(len(data_points)):
        if i == 0:
            current_session.append(data_points[i])
        else:
            time_diff = data_points[i]['time'] - data_points[i-1]['time']
            if time_diff <= gap_threshold:
                current_session.append(data_points[i])
            else:
                # Gap detected, save the current session and start a new one
                sessions.append(current_session)
                current_session = [data_points[i]]

    # After processing all data points, add the last session
    if current_session:
        sessions.append(current_session)

    if not sessions:
        print("No ECG data found in the last 24 hours.")
    else:
        last_session = sessions[-1]
        print(f"Found {len(sessions)} sessions. Last session has {len(last_session)} data points.")

        # Save the data to a CSV file
        output_filename = 'ecg_last_session_hold2.csv'

        with open(output_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['time', 'ecg_value'])
            for point in last_session:
                csv_writer.writerow([point['time'].isoformat(), point['value']])

        print(f"Data from last session saved to {output_filename}.")

        # Create a DataFrame for plotting
        df = pd.DataFrame(last_session)

        # Plotting using Plotly
        # fig = px.line(df, x='time', y='value', title='ECG Data from Last Session')
        #
        # # Update layout for better interactivity
        # fig.update_layout(
        #     xaxis_title='Time',
        #     yaxis_title='ECG Value',
        #     xaxis=dict(
        #         rangeselector=dict(
        #             buttons=list([
        #                 dict(count=15, label='15min', step='minute', stepmode='backward'),
        #                 dict(count=1, label='1h', step='hour', stepmode='backward'),
        #                 dict(count=6, label='6h', step='hour', stepmode='backward'),
        #                 dict(step='all')
        #             ])
        #         ),
        #         rangeslider=dict(visible=True),
        #         type='date'
        #     )
        # )
        #
        # # Show the interactive plot
        # fig.show()

    # Close the client explicitly
    client.close()

if __name__ == '__main__':
    main()