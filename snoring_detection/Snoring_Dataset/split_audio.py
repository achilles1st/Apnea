import os
from pydub import AudioSegment

def split_audio(input_file, output_folder):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Get the duration of the audio in milliseconds
    duration_ms = len(audio)

    # Set chunk length in milliseconds (1 second = 1000 ms)
    chunk_length_ms = 1000

    # Initialize start position
    start_ms = 0

    # Initialize counter for file naming
    counter = 1000

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over the audio and extract 1-second chunks
    while start_ms < duration_ms:
        end_ms = start_ms + chunk_length_ms
        if end_ms > duration_ms:
            end_ms = duration_ms

        chunk = audio[start_ms:end_ms]

        # Save chunk as .wav file with the naming scheme "0_counter.wav"
        filename = f"0_{counter}.wav"
        output_path = os.path.join(output_folder, filename)
        chunk.export(output_path, format="wav")
        print(f"Saved chunk {counter} as {filename}")
        counter += 1

        # Move to next chunk
        start_ms += chunk_length_ms

    print("Audio splitting completed successfully.")

if __name__ == "__main__":
    input_file = "C:/Users/tosic/Arduino_projects/sensor_com/snoring_detection/Snoring_Dataset/recorded_audio_talk0.wav"
    output_folder = "C:/Users/tosic/Arduino_projects/sensor_com/snoring_detection/Snoring_Dataset/additions/0"
    split_audio(input_file, output_folder)