import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

def bandpass_filter(data, fs, lowcut=0.1, highcut=0.1, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def amplitude_envelope(data):
    analytic_signal = hilbert(data)
    envelope = np.abs(analytic_signal)
    return envelope, analytic_signal

def detect_pauses(envelope, threshold, min_duration_s, fs):
    below_thresh = envelope < threshold
    pause_segments = []
    start = None
    for i, val in enumerate(below_thresh):
        if val and start is None:
            start = i
        elif not val and start is not None:
            length = i - start
            if length >= min_duration_s * fs:
                pause_segments.append((start, i))
            start = None
    if start is not None:
        length = len(below_thresh) - start
        if length >= min_duration_s * fs:
            pause_segments.append((start, len(below_thresh)))
    return pause_segments

# Example usage with synthetic data
fs = 50.0
time = np.arange(0, 60, 1 / fs)
freq = 0.3
offset = 200.0
amplitude = 30.0
baseline = offset + 10 * np.sin(0.01 * 2 * np.pi * time)
resp_signal = baseline + amplitude * np.sin(2 * np.pi * freq * time)
resp_signal[1000:1200] = 200
resp_signal[3000:3400] = 200

filtered = bandpass_filter(resp_signal, fs, lowcut=0.1, highcut=1)
envelope, analytic_signal = amplitude_envelope(filtered)
threshold = np.mean(envelope) * 0.3
min_duration_s = 2.0
pauses = detect_pauses(envelope, threshold, min_duration_s, fs)

print("Detected pauses (sample indices):", pauses)
if len(pauses) > 0:
    for seg in pauses:
        start_time = seg[0] / fs
        end_time = seg[1] / fs
        print(f"  Pause from ~{start_time:.2f}s to {end_time:.2f}s")

plt.figure(figsize=(10, 6))
plt.plot(time, filtered, label='Filtered Signal', color='blue')
#plt.plot(time, np.real(analytic_signal), label='Hilbert Transform', color='blue')
plt.plot(time, envelope, label='Amplitude Envelope', color='green')
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
for seg in pauses:
    start_t = seg[0] / fs
    end_t = seg[1] / fs
    plt.axvspan(start_t, end_t, color='red', alpha=0.3)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('respiration_signal_plot.jpg')
plt.show()