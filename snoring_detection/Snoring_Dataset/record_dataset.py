import sounddevice as sd
import numpy as np
import wave
import keyboard  # You need to install the keyboard module

# Parameters
sample_rate = 44100  # Hz
channels = 1
device_index = 1  # Replace with the index of your microphone
filename = 'recorded_audio.wav'


def record_audio():
    print("Press Enter to start recording...")
    keyboard.wait('enter')
    print("Recording... Press 'q' to stop.")

    audio_data = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_data.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', callback=callback, device=device_index):
        keyboard.wait('q')

    print("Recording stopped. Saving file...")

    audio_data = np.concatenate(audio_data, axis=0)
    wave_file = wave.open(filename, 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(2)  # 2 bytes per sample
    wave_file.setframerate(sample_rate)
    wave_file.writeframes(audio_data.tobytes())
    wave_file.close()

    print(f"Audio recorded and saved to {filename}")


if __name__ == "__main__":
    record_audio()