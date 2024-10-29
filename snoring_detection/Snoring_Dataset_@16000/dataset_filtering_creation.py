import os
import sys
from pydub import AudioSegment
from pydub.playback import play

def split_and_review_audio(input_file, output_folder):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Get the duration of the audio in milliseconds
    duration_ms = len(audio)

    # Set chunk length in milliseconds (1 second = 1000 ms)
    chunk_length_ms = 1000

    # Initialize start position
    start_ms = 0

    # Initialize counter for file naming
    counter = 658  # Starting from 500 as per naming schema

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over the audio and extract 1-second chunks
    while start_ms < duration_ms:
        end_ms = start_ms + chunk_length_ms
        if end_ms > duration_ms:
            end_ms = duration_ms

        chunk = audio[start_ms:end_ms]

        # Play the chunk
        print(f"\nPlaying chunk {counter}...")
        try:
            play(chunk)
        except Exception as e:
            print(f"Error playing chunk {counter}: {e}")
            # Decide whether to skip saving this chunk
            user_input = input("Error playing chunk. Press Enter to skip saving this chunk, or type 'S' to save anyway: ").strip()
            if user_input.lower() != 's':
                # Skip saving
                start_ms += chunk_length_ms
                continue

        # Prompt user for input
        user_input = input("Press Enter to save this sample, or type 'D' to discard this sample: ").strip()

        if user_input.lower() == 'd':
            # Discard the sample
            print(f"Discarded chunk {counter}")
        else:
            # Save chunk as .wav file with the naming scheme "0_counter.wav"
            filename = f"0_{counter}.wav"
            output_path = os.path.join(output_folder, filename)
            chunk.export(output_path, format="wav")
            print(f"Saved chunk {counter} as {filename}")
            counter += 1

        # Move to next chunk
        start_ms += chunk_length_ms

    print("Audio splitting and reviewing completed successfully.")

if __name__ == "__main__":
    input_file = "C:/Users/tosic/Arduino_projects/sensor_com/old_files/strong_breathing_16k.wav"
    #input_file = "C:/Users/tosic/Arduino_projects/sensor_com/old_files/recorded_snoring_woman.wav"
    #input_file = "C:/Users/tosic/Arduino_projects/sensor_com/old_files/recorded_snoring_youtube.wav"
    #input_file = "C:/Users/tosic/Arduino_projects/sensor_com/old_files/recorded_snoring_yoututbe_peter_baeten.wav"

    output_folder = "C:/Users/tosic/Arduino_projects/sensor_com/snoring_detection/Snoring_Dataset_@16000/dataset_16/additions/0"
    split_and_review_audio(input_file, output_folder)
