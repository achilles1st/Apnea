import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pandas as pd
import configparser
import datetime


class TimeAxisItem(pg.AxisItem):
    """Custom AxisItem to display timestamps instead of numeric seconds."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enableAutoSIPrefix(False)

    def tickStrings(self, values, scale, spacing):
        # Convert numeric timestamp values to human-readable time strings
        return [datetime.datetime.fromtimestamp(value, datetime.UTC).strftime("%Y-%m-%d %H:%M:%S") for value in values]


class ApneaViewer(QtWidgets.QMainWindow):
    def __init__(self, field, times, values, apnea_events, baselines=None):
        super().__init__()
        self.field = field
        self.setWindowTitle(f"{self.field} values with Apnea Detection")
        self.original_start_time = None

        # Convert times into a numpy array of timestamps (POSIX epoch)
        # If `times` are numpy.datetime64 or pandas.Timestamp, convert them to datetime and then to POSIX time
        if np.issubdtype(times.dtype, np.datetime64):
            # Numpy datetime64 to datetime
            datetimes = pd.to_datetime(times).to_pydatetime()
        elif isinstance(times[0], pd.Timestamp):
            # Pandas Timestamp to datetime
            datetimes = [ts.to_pydatetime() for ts in times]
        else:
            # If already numeric seconds, interpret them as relative to start
            # In this case, let's assume we need a reference epoch (like now)
            # But better to handle it explicitly if this case arises
            raise ValueError("times must be datetime-like for timestamp axis.")

        # Use the first timestamp as reference if needed
        self.original_start_time = datetimes[0]

        # Convert all datetimes to POSIX timestamps
        self.times_in_seconds = np.array([dt.timestamp() for dt in datetimes])

        self.values = values
        self.baselines = baselines
        self.events = apnea_events

        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # Define the unit for the left axis label
        unit = "%" if self.field == "spO2" else "bpm"

        # Create a custom bottom axis item
        time_axis = TimeAxisItem(orientation='bottom')
        self.pr_plot_widget = pg.PlotWidget(axisItems={'bottom': time_axis}, title=f"{self.field} Over Time")
        self.pr_plot_widget.setLabel('left', f"{self.field}", units=unit)
        self.pr_plot_widget.setLabel('bottom', "Time")

        # Plot pulse/resp data
        self.curve = self.pr_plot_widget.plot(self.times_in_seconds, self.values, pen='b', name="Data")

        # Plot the baselines if available
        if baselines is not None:
            self.baseline_curve = self.pr_plot_widget.plot(self.times_in_seconds, self.baselines, pen='g', name="Baseline")

        # Highlight detected apnea events
        for event in self.events:
            if isinstance(event[0], np.datetime64):
                event_start_dt = pd.to_datetime(event[0]).to_pydatetime().replace(tzinfo=None)
                event_end_dt = pd.to_datetime(event[1]).to_pydatetime().replace(tzinfo=None)
            elif isinstance(event[0], pd.Timestamp):
                event_start_dt = event[0].to_pydatetime().replace(tzinfo=None)
                event_end_dt = event[1].to_pydatetime().replace(tzinfo=None)
            else:
                # If events are in another numeric form, you'd need to convert accordingly
                continue

            event_start = event_start_dt.timestamp()
            event_end = event_end_dt.timestamp()

            region = pg.LinearRegionItem(values=[event_start, event_end], brush=(255, 0, 0, 50))
            region.setMovable(False)
            self.pr_plot_widget.addItem(region)

        main_layout.addWidget(self.pr_plot_widget)

        # Add a vertical line and a text item for hover display
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen='g')
        self.pr_plot_widget.addItem(self.vline, ignoreBounds=True)

        self.text_item = pg.TextItem("", anchor=(0, 1))
        self.pr_plot_widget.addItem(self.text_item)

        # Connect the mouse movement signal
        self.proxy = pg.SignalProxy(self.pr_plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

    def mouse_moved(self, evt):
        pos = evt[0]  # pos is a QtCore.QPointF
        if self.pr_plot_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.pr_plot_widget.getPlotItem().vb.mapSceneToView(pos)
            x_val = mousePoint.x()
            y_val = mousePoint.y()

            # Find closest time index
            idx = np.searchsorted(self.times_in_seconds, x_val)
            if idx < 0:
                idx = 0
            elif idx >= len(self.times_in_seconds):
                idx = len(self.times_in_seconds) - 1

            # Check if idx-1 is closer
            if idx > 0 and idx < len(self.times_in_seconds):
                if abs(self.times_in_seconds[idx] - x_val) > abs(self.times_in_seconds[idx - 1] - x_val):
                    idx = idx - 1

            # Update line position
            closest_time = self.times_in_seconds[idx]
            closest_pr = self.values[idx]

            # Convert closest_time back to a datetime for display
            closest_dt = datetime.datetime.fromtimestamp(closest_time, datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")

            self.vline.setPos(closest_time)
            if self.baselines is not None:
                closest_baseline = self.baselines[idx]
                self.text_item.setText(f"Time: {closest_dt}\n{self.field}: {closest_pr:.1f}bpm\nBaseline: {closest_baseline:.1f}%")
            else:
                self.text_item.setText(f"Time: {closest_dt}\n{self.field}: {closest_pr:.1f}")

            self.text_item.setPos(closest_time, closest_pr)


def main():
    # Read configuration
    config = configparser.ConfigParser()
    config.read('config.ini')

    url = config['INFLUXDB']['URL']
    token = config['INFLUXDB']['TOKEN']
    org = config['INFLUXDB']['ORG']
    bucket = config['BUCKETS']['PULSEOXY_BUCKET']
    measurement = config['MEASUREMENTS']['Respiration_Measurements']
    sensor_field = config['FIELDS']['Respiration']

    from apnea_detection_system.pulse_oxi_detection import PulseOxiDetector
    from apnea_detection_system.Respiration_detection import CWTBasedApneaDetector

    if sensor_field == "spO2" or sensor_field == "PulseRate":
        detector = PulseOxiDetector(url, token, org, sensor_field, fs=60, baseline_window_minutes=30)
        df = detector.get_data(bucket, measurement, source='local')
        df_pf_baseline = detector.compute_rolling_baseline(df, min_periods_minutes=15)
        apnea_events = detector.detect_apnea_events(df_pf_baseline)
        baselines = df_pf_baseline["baseline"].values
        data = df[f"{sensor_field}"].values
        t = df["time"].values  # numpy array of datetime64 objects
    elif sensor_field == "resp_value":
        detector = CWTBasedApneaDetector(url, token, org, sensor_field, fs=10, min_duration=6)
        df = detector.get_data(bucket, measurement, source='local')
        apnea_events = detector.detect_apnea_events(df)
        data = df[f"{sensor_field}"].values
        t = df["time"].values
        baselines = None
    else:
        raise ValueError(f"Invalid sensor field: {sensor_field}")

    # Launch the GUI
    app = QtWidgets.QApplication(sys.argv)
    viewer = ApneaViewer(sensor_field, t, data, apnea_events, baselines)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
