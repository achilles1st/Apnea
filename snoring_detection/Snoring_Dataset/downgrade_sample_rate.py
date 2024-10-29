import os   
from pydub import AudioSegment

def main(directory, save_directory):
    extensions = ["/1"]#, "/0"]
    for extension in extensions:
        for filename in os.listdir(directory + extension):
            
            if filename.endswith(".wav"): 
                sound = AudioSegment.from_wav(directory + extension + "/" + filename)

                # Convert to mono if the audio has more than one channel
                if sound.channels > 1:
                    sound = sound.split_to_mono()[0]

                sound = sound.set_frame_rate(16000)
                sound.export(save_directory + extension + "/" + filename, format="wav")                 
    

                

directory = "C:/Users/tosic/Arduino_projects/sensor_com/snoring_detection/Snoring_Dataset_@16000/44100_additions/left_over_data"
save_directory = "C:/Users/tosic/Arduino_projects/sensor_com/snoring_detection"
main(directory, save_directory)