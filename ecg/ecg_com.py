import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import sys
import time
import numpy as np
import heartpy as hp

# Serial port configuration
serial_port = 'COM3'  # Replace with your Arduino's serial port
baud_rate = 9600  # Must match the Arduino baud rate

# Initialize serial communication
try:
    ser = serial.Serial(serial_port, baud_rate)
    time.sleep(2)  # Wait for the serial connection to initialize
except serial.SerialException as e:
    print(f"Error opening serial port {serial_port}: {e}")
    sys.exit(1)

app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Real-time ECG Signal with Heart Rate")
plot = win.addPlot(title="Filtered ECG Signal")
curve = plot.plot(pen='r')
hr_text = pg.TextItem(anchor=(1, 0))  # Position at the upper right corner
hr_text.setPos(1, 1)
#plot.addItem(hr_text)
plot.setLabel('left', 'Amplitude', units='mV')
plot.setLabel('bottom', 'Time', units='s')

# Data lists
times = []
values = []
filtered_values = []
heart_rates = []

window_size = 10  # Number of samples for moving average
start_time = time.time()
last_detection_time = 0  # Initialize last detection time

def update():
    global times, values, filtered_values, heart_rates, last_detection_time
    try:
        data = ser.readline().decode('utf-8').strip()
    except UnicodeDecodeError:
        # Skip lines that can't be decoded
        return
    current_time = time.time() - start_time

    if data == '!':
        print("Lead off detected!")
    else:
        try:
            value = int(data)
            times.append(current_time)
            values.append(value)
            print(f"Value: {value}")

            # Apply moving average filter
            if len(values) >= window_size:
                window_values = values[-window_size:]
                filtered_value = sum(window_values) / window_size
            else:
                filtered_value = sum(values) / len(values)
            filtered_values.append(filtered_value)

            # Limit data to last 10 seconds
            min_time = current_time - 10  # Adjusted to 10 seconds
            while times and times[0] < min_time:
                times.pop(0)
                values.pop(0)
                filtered_values.pop(0)

            # Update the plot with filtered data
            curve.setData(times, filtered_values)
            plot.enableAutoRange('y', True)
            plot.setXRange(current_time - 5, current_time)  # Keep last 10 seconds on x-axis

            # Call detect_r_peaks every 10 seconds if enough data is collected
            if current_time - last_detection_time >= 10 and len(filtered_values) >= 1000:
                detect_r_peaks(times, filtered_values)
                last_detection_time = current_time

        except ValueError:
            print(f"Invalid data received: {data}")

def detect_r_peaks(times, signal):
    global heart_rates

    # Use HeartPy to process the ECG signal
    try:
        duration = times[-1] - times[0]
        if duration == 0:
            return  # Avoid division by zero

        # Use fixed sample rate
        sample_rate = 200
        #print(f"Detecting heart rate with duration {duration:.2f}s and sample_rate {sample_rate:.2f}Hz")

        wd, m = hp.process(np.array(signal), sample_rate=sample_rate)
        # Extract heart rate
        heart_rate = m['bpm']
        heart_rates.append(heart_rate)
        print(f"Heart Rate: {heart_rate:.1f} BPM")

        # Update heart rate text
        hr_text.setText(f"Heart Rate: {heart_rate:.1f} BPM", color='r')
        hr_text.setPos(times[-1], max(filtered_values))

    except Exception as e:
        print(f"HeartPy processing error: {e}")

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(5)

if __name__ == '__main__':
    QtWidgets.QApplication.instance().exec_()
    ser.close()
