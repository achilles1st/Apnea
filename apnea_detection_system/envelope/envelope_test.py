import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert


def bandpass_filter(data, fs, lowcut=0.1, highcut=1.0, order=1):
    """
    Applies a bandpass Butterworth filter to remove offsets
    and frequencies outside the normal respiration band.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def amplitude_envelope(data):
    """
    Computes the amplitude envelope via the Hilbert transform.
    """
    analytic_signal = hilbert(data)
    envelope = np.abs(analytic_signal)
    return envelope


def detect_pauses(envelope, threshold, min_duration_s, fs):
    """
    Detects periods where 'envelope' stays below 'threshold'
    for at least 'min_duration_s' seconds.

    Returns a list of (start_index, end_index) samples where
    the signal is considered paused.
    """
    below_thresh = envelope < threshold
    pause_segments = []

    # We’ll scan through 'below_thresh' and group consecutive True values.
    start = None
    for i, val in enumerate(below_thresh):
        if val and start is None:
            start = i  # beginning of a potential pause
        elif not val and start is not None:
            # just ended a below-threshold segment
            length = i - start
            # Check if it meets duration requirement
            if length >= min_duration_s * fs:
                pause_segments.append((start, i))
            start = None
    # Handle the case if a pause extends to the end of the signal
    if start is not None:
        length = len(below_thresh) - start
        if length >= min_duration_s * fs:
            pause_segments.append((start, len(below_thresh)))

    return pause_segments


# ---------------------------------------------------------------------
# Example usage with synthetic data. Replace 'signal' with your data.
# ---------------------------------------------------------------------

# 1. Generate some synthetic respiration data for demonstration
fs = 50.0  # sampling frequency in Hz
time = np.arange(0, 60, 1 / fs)  # 60 seconds of data
freq = 0.2  # a typical respiration frequency in Hz (0.3 ~ 18 breaths/min)
offset = 200.0  # some offset/baseline
amplitude = 30.0

# Create a sinusoidal breathing wave plus a wandering baseline
baseline = offset + 10 * np.sin(0.01 * 2 * np.pi * time)  # slow drift
#baseline = offset +  np.sin(0.01 * 2 * np.pi * time)  # slow drift

resp_signal = baseline + amplitude * np.sin(2 * np.pi * freq * time)

# Insert artificially low-amplitude regions (simulating pauses)
resp_signal[1000:1200] = 200 # short pause
resp_signal[3000:3400] = 200  # longer pause

# 2. Filter the signal to remove slow offset/drift
filtered = bandpass_filter(resp_signal, fs, lowcut=0.1, highcut=1)

# 3. Compute the amplitude envelope
envelope = amplitude_envelope(filtered)

# 4. Detect pause regions
threshold = np.mean(envelope) * 0.3  # amplitude threshold (tweak this)
min_duration_s = 2.0  # must remain below threshold for at least 2 seconds
pauses = detect_pauses(envelope, threshold, min_duration_s, fs)

# Print pause intervals in sample indices, plus approximate times
print("Detected pauses (sample indices):", pauses)
if len(pauses) > 0:
    for seg in pauses:
        start_time = seg[0] / fs
        end_time = seg[1] / fs
        print(f"  Pause from ~{start_time:.2f}s to {end_time:.2f}s")

# 5. Plot results
fig, ax = plt.subplots(3, 1, figsize=(10, 7))

# (a) Raw signal
ax[0].plot(time, resp_signal, label='Raw Respiration Signal')
ax[0].set_ylabel('Amplitude')
ax[0].legend(loc='lower left')
ax[0].set_title('Raw Respiration Signal')
ax[0].text(-0.1, 1.1, '1)', transform=ax[0].transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

# (b) Filtered signal
ax[1].plot(time, filtered, label='Filtered Signal', color='C2')
ax[1].set_ylabel('Amplitude')
ax[1].legend(loc='lower left')
ax[1].set_title('Filtered Signal')
ax[1].text(-0.1, 1.1, '2)', transform=ax[1].transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

# (c) Envelope with pause detection
ax[2].plot(time, envelope, label='Amplitude Envelope', color='C3')
ax[2].axhline(y=threshold, color='orange', linestyle='--', label='Threshold')
# Shade the pause intervals
for seg in pauses:
    start_t = seg[0] / fs
    end_t = seg[1] / fs
    ax[2].axvspan(start_t, end_t, color='red', alpha=0.3)
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Envelope')
ax[2].legend(loc='lower left')
ax[2].set_title('Envelope with Pause Detection')
ax[2].text(-0.1, 1.1, '3)', transform=ax[2].transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

plt.tight_layout()

plt.savefig('respiration_envelope_steps.jpg')

plt.show()



# example form paper
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import hilbert
#
# # Generate the time vector
# t = np.arange(1, 131073) * 1e-4
#
# # Generate the signal
# gt = np.sin(t) * np.sin(10 * t)
#
# # Compute the Hilbert transform and the envelope
# analytic_signal = hilbert(gt)
# envelope = np.abs(analytic_signal)
#
# # Plot the results
# plt.figure()
# plt.plot(t, gt, 'b-', label='Original Signal')
# plt.plot(t, np.imag(analytic_signal), 'k--', label='Hilbert Transform (Imaginary Part)')
# plt.plot(t, envelope, 'r:', label='Envelope')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()
