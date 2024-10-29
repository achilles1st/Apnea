import os
import csv
import sys
from playsound import playsound

def play_and_manage_samples(samples_folder, csv_filename):
    # List all .wav files in the samples folder, sorted alphabetically
    sample_files = sorted([f for f in os.listdir(samples_folder) if f.endswith('.wav')])

    # Initialize a list to store deleted filenames
    deleted_samples = []

    for sample_file in sample_files:
        sample_path = os.path.join(samples_folder, sample_file)

        print(f"\nPlaying {sample_file}...")
        # Play the audio sample
        try:
            playsound(sample_path)
        except Exception as e:
            print(f"Error playing {sample_file}: {e}")
            continue  # Skip to the next sample if there's an error

        # Prompt user for input
        user_input = input("Press Enter to play next sample, or type 'D' to delete this sample: ").strip()

        if user_input.lower() == 'd':
            # Delete the sample file
            try:
                os.remove(sample_path)
                # Save the filename in the list
                deleted_samples.append(sample_file)
                # Print the filename
                print(f"Deleted {sample_file}")
            except Exception as e:
                print(f"Error deleting {sample_file}: {e}")
        else:
            # Proceed to next sample
            print(f"Keeping {sample_file}")

    # After all samples, save the deleted filenames to a CSV file
    if deleted_samples:
        csv_path = os.path.join(samples_folder, csv_filename)
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for filename in deleted_samples:
                    writer.writerow([filename])
            print(f"\nDeleted samples saved in {csv_path}")
        except Exception as e:
            print(f"Error writing to CSV file: {e}")
    else:
        print("\nNo samples were deleted.")

if __name__ == "__main__":

    samples_folder = "C:/Users/tosic/Arduino_projects/sensor_com/snoring_detection/Snoring_Dataset_@16000/44100_additions/1"
    csv_filename = "wrong_samples.csv"
    if not os.path.isdir(samples_folder):
        print(f"Error: The folder '{samples_folder}' does not exist.")
        sys.exit(1)
    play_and_manage_samples(samples_folder, csv_filename)
