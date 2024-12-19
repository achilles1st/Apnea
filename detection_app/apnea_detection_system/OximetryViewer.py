import sys
from dataclasses import field

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from pulse_oxi_detection import PulseOxiDetector
import pandas as pd  # Ensure pandas is imported


class OximetryViewer(QtWidgets.QMainWindow):
    def __init__(self, field, times, values, apnea_events, baselines):
        super().__init__()
        self.field = field
        self.setWindowTitle(f"{self.field}values with Apnea Detection")
        # Initialize original_start_time
        self.original_start_time = None

        # Check if pr_times are numpy.datetime64, pandas.Timestamp, or floats
        if np.issubdtype(times.dtype, np.datetime64):
            # Convert numpy.datetime64 to datetime.datetime (timezone-naive)
            start_time = pd.to_datetime(times[0]).to_pydatetime().replace(tzinfo=None)
            self.original_start_time = start_time

            # Convert all pr_times to float seconds relative to start_time
            self.times_in_seconds = np.array([
                (pd.to_datetime(ts).to_pydatetime().replace(tzinfo=None) - start_time).total_seconds()
                for ts in times
            ])
        elif isinstance(times[0], pd.Timestamp):
            # Convert pandas.Timestamp to datetime.datetime (timezone-naive)
            start_time = times[0].to_pydatetime().replace(tzinfo=None)
            self.original_start_time = start_time

            # Convert all pr_times to float seconds relative to start_time
            self.times_in_seconds = np.array([
                (ts.to_pydatetime().replace(tzinfo=None) - start_time).total_seconds()
                for ts in times
            ])
        else:
            # Assume pr_times are already in float seconds
            self.times_in_seconds = times.astype(float)
            self.original_start_time = 0.0  # Dummy value

        self.values = values
        self.baselines = baselines
        self.events = apnea_events

        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # Create the apnea_detection_system plot
        self.pr_plot_widget = pg.PlotWidget(title="apnea_detection_system Over Time")
        self.pr_plot_widget.setLabel('left', "apnea_detection_system", units='bpm')
        self.pr_plot_widget.setLabel('bottom', "Time", units='s')

        # Plot pulse rate data
        self.curve = self.pr_plot_widget.plot(self.times_in_seconds, self.values, pen='b', name="Pulse Rate")

        # Plot the baselines
        self.baseline_curve = self.pr_plot_widget.plot(self.times_in_seconds, self.baselines, pen='g', name="Baseline")

        # Highlight detected apnea events
        for event in self.events:
            if isinstance(event[0], np.datetime64):
                # Convert numpy.datetime64 to datetime.datetime (timezone-naive)
                event_start_dt = pd.to_datetime(event[0]).to_pydatetime().replace(tzinfo=None)
                event_end_dt = pd.to_datetime(event[1]).to_pydatetime().replace(tzinfo=None)

                # Calculate seconds relative to original_start_time
                event_start = (event_start_dt - self.original_start_time).total_seconds()
                event_end = (event_end_dt - self.original_start_time).total_seconds()
            elif isinstance(event[0], pd.Timestamp):
                # Convert pandas.Timestamp to datetime.datetime (timezone-naive)
                event_start_dt = event[0].to_pydatetime().replace(tzinfo=None)
                event_end_dt = event[1].to_pydatetime().replace(tzinfo=None)

                # Calculate seconds relative to original_start_time
                event_start = (event_start_dt - self.original_start_time).total_seconds()
                event_end = (event_end_dt - self.original_start_time).total_seconds()
            else:
                # Assume events are already in float seconds
                event_start = event[0]
                event_end = event[1]

            # Create and add the LinearRegionItem
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
            closest_baseline = self.baselines[idx]
            self.vline.setPos(closest_time)

            self.text_item.setText(f"Time: {closest_time:.2f}s\n{self.field}: {closest_pr:.1f}bpm\nBaseline: {closest_baseline:.1f}%")
            self.text_item.setPos(closest_time, closest_pr)

def main():
    url = "http://sleep-apnea:8086"
    token = "nFshsCSH5OyLFv9tSjPBIyOPvwXzJpt4zEAnm9OJFpVlEcUWOzSCAia3MRFrN-C8ljfQbKu6VgoRlTBQZoXTrg=="
    org = "TU"
    bucket = "Pulseoxy"
    measurement = "pulseoxy_samples"
    field = "PulseRate"

    # Create detector with 60 Hz data and a 5-minute rolling window
    detector = PulseOxiDetector(url, token, org, field, fs=60, baseline_window_minutes=10)
    df_pr = detector.get_data(bucket, measurement, source='influx')
    df_pf_baseline = detector.compute_rolling_baseline(df_pr, min_periods_minutes=5)
    apnea_events = detector.detect_apnea_events(df_pf_baseline)

    pr_data = df_pr["PulseRate"].values
    baselines = df_pf_baseline["baseline"].values
    t = df_pr["time"].values  # numpy array of datetime64 objects

    # Launch the GUI
    app = QtWidgets.QApplication(sys.argv)
    viewer = OximetryViewer(field, t, pr_data, apnea_events, baselines)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()