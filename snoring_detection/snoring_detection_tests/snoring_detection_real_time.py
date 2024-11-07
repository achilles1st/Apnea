import numpy as np
import tensorflow as tf
import sounddevice as sd
import threading
import tkinter as tk
from tkinter import ttk

# Define constants
MODEL_PATH = '../models/cnn_latest_44k.keras'  # Replace with the path to your trained model
SEGMENT_DURATION = 1  # Duration of each audio segment in seconds
THRESHOLD = 0.5  # Probability threshold for snoring classification

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

def get_log_mel(waveform, sample_rate):
    frame_length = int(0.032 * sample_rate)  # Approximately 32ms
    frame_step = int(0.016 * sample_rate)    # Approximately 16ms
    stfts = tf.signal.stft(waveform, frame_length=frame_length, frame_step=frame_step)

    # Obtain the magnitude of the STFT.
    spectrograms = tf.abs(stfts)
    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 40.0, 6000.0, 30
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    log_mel_spectrograms = tf.reshape(log_mel_spectrograms, [-1, 1830])
    return log_mel_spectrograms

def classify_snoring_segment(segment, sample_rate):
    """Classifies snoring events in the given audio segment."""
    # Convert segment to tensor and normalize
    x = tf.convert_to_tensor(segment, dtype=tf.float32)
    x = x / (tf.reduce_max(tf.abs(x)) + 1e-6)  # Normalize between -1 and 1

    # Ensure x is 1D
    x = tf.reshape(x, [-1])

    # Get log-mel spectrogram
    x = get_log_mel(x, sample_rate)

    # Make prediction
    prediction = model(x)
    probability = prediction[0, 0].numpy()
    return probability

class SnoreDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Snore Detector")
        self.root.geometry("800x600")  # Set window size
        self.create_widgets()
        self.is_running = False
        self.sample_rate = 44100  # Sample rate for recording
        self.samples_per_segment = int(self.sample_rate * SEGMENT_DURATION)

    def create_widgets(self):
        # Create a frame that fills the window, but above the buttons
        self.frame = tk.Frame(self.root)
        self.frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Initialize background color
        self.frame.configure(bg='grey')

        # Create a label for the text
        self.label_text = tk.StringVar()
        self.label = tk.Label(self.frame, textvariable=self.label_text, font=("Helvetica", 32), bg='grey')
        self.label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Start and Stop buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.BOTTOM, pady=10)

        self.start_button = ttk.Button(self.button_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=20)

        self.stop_button = ttk.Button(self.button_frame, text="Stop Detection", command=self.stop_detection)
        self.stop_button.pack(side=tk.RIGHT, padx=20)
        self.stop_button.state(['disabled'])  # Disable stop button initially

    def start_detection(self):
        if not self.is_running:
            self.is_running = True
            self.start_button.state(['disabled'])  # Disable start button
            self.stop_button.state(['!disabled'])  # Enable stop button
            threading.Thread(target=self.detect_snoring, daemon=True).start()

    def stop_detection(self):
        if self.is_running:
            self.is_running = False
            self.start_button.state(['!disabled'])  # Enable start button
            self.stop_button.state(['disabled'])    # Disable stop button
            # Reset background to grey and clear text
            self.frame.configure(bg='grey')
            self.label_text.set("")

    def detect_snoring(self):
        print("Starting real-time snoring detection...")
        try:
            while self.is_running:
                # Record audio segment
                segment = sd.rec(self.samples_per_segment, samplerate=self.sample_rate, channels=1, dtype='float32')
                sd.wait()  # Wait until recording is finished

                # Convert to 1D numpy array
                segment = np.squeeze(segment)

                # Classify the segment
                probability = classify_snoring_segment(segment, self.sample_rate)

                # Update the GUI
                if probability >= THRESHOLD:
                    print(f"Snoring detected with probability: {probability:.2f}")
                    self.frame.configure(bg='green')
                    self.label_text.set("Snoring Detected!")
                else:
                    print(f"No snoring detected with probability: {probability:.2f}")
                    self.frame.configure(bg='grey')
                    self.label_text.set("")

                # Update label background to match frame background
                self.label.configure(bg=self.frame['bg'])

        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.is_running = False
            self.start_button.state(['!disabled'])  # Enable start button
            self.stop_button.state(['disabled'])    # Disable stop button
            # Reset background to grey and clear text
            self.frame.configure(bg='grey')
            self.label_text.set("")
            print("Exiting...")

if __name__ == "__main__":
    root = tk.Tk()
    app = SnoreDetectorGUI(root)
    root.mainloop()
