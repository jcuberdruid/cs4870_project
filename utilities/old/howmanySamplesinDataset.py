import os
import glob
import subprocess
import re

# Function to extract duration and sample rate from a wav file using ffmpeg
def get_file_info_ffmpeg(file_path):
    print(f"trying file {file_path}")
    try:
        # Run the ffmpeg command and extract the output
        result = subprocess.run(['ffmpeg', '-i', file_path], stderr=subprocess.PIPE, text=True)
        output = result.stderr

        # Use regular expressions to find the duration and sample rate
        duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", output)
        sample_rate_match = re.search(r"(\d+) Hz", output)

        if duration_match and sample_rate_match:
            hours, minutes, seconds = map(float, duration_match.groups())
            duration = 3600 * hours + 60 * minutes + seconds
            sample_rate = int(sample_rate_match.group(1))
            return duration, sample_rate
        else:
            return None, None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None

# Function to count total samples in all wav files using ffmpeg to get metadata
def count_total_samples(wav_files):
    total_samples = 0
    for file_path in wav_files:
        duration, sample_rate = get_file_info_ffmpeg(file_path)
        if duration is not None and sample_rate is not None:
            total_samples += int(duration * sample_rate)
    return total_samples

def main():
    # Define the path to your data directory
    new_path = "./data/"

    # Find all .wav files in the directory
    all_files = glob.glob(os.path.join(new_path, '**/*.wav'), recursive=True)

    # Count the total samples using metadata extracted with ffmpeg
    total_samples = count_total_samples(all_files)

    # Save the result to a text file
    with open('total_samples_ffmpeg.txt', 'w') as f:
        f.write(f"Total number of samples in dataset: {total_samples}\n")

    # Also print the result to the console
    print(f"Total number of samples in dataset: {total_samples}")

if __name__ == "__main__":
    main()

