import os
from pydub import AudioSegment

## segments wavs into 5 minute sections

def split_wav_file(file_path, segment_length_ms):
    """Splits a WAV file into segments of a specified length."""
    audio = AudioSegment.from_wav(file_path)
    length_audio = len(audio)
    segments = []

    for i in range(0, length_audio, segment_length_ms):
        segment = audio[i:i + segment_length_ms]
        if len(segment) == segment_length_ms:
            segments.append(segment)
    
    return segments

def process_directory(directory):
    """Recursively processes directories to find and split WAV files."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                file_id = os.path.splitext(file)[0]
                
                # Splitting the file into 5-minute segments
                segments = split_wav_file(file_path, 5 * 60 * 1000) # 5 minutes in milliseconds

                # Saving and sanity-checking segments
                for idx, segment in enumerate(segments):
                    segment_file_name = f"{file_id}_{idx}.wav"
                    segment_file_path = os.path.join(root, segment_file_name)
                    segment.export(segment_file_path, format='wav')

                    # Sanity check: Verify file size and duration
                    if os.path.getsize(segment_file_path) > 0 and len(segment) == 5 * 60 * 1000:
                        print(f"Segment {segment_file_name} created successfully.")
                    else:
                        print(f"Error in creating segment {segment_file_name}.")
                        break
                else:
                    # If all segments are created successfully, delete the original file
                    os.remove(file_path)
                    print(f"Original file {file} deleted.")

# Run the script for the current directory
process_directory('.')

