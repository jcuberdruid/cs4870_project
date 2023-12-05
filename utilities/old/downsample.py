import os
import wave
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample, butter, filtfilt
from concurrent.futures import ThreadPoolExecutor
import gc

def get_sample_rate(wav_file_path):
    with wave.open(wav_file_path, 'rb') as wav_file:
        return wav_file.getframerate()

def apply_lowpass_filter(data, sample_rate, cutoff=500, order=5):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def downsample_to_8k(original_file_path):
    original_rate, original_data = wavfile.read(original_file_path)

    # If the audio is stereo (2 channels), pick the left channel
    if len(original_data.shape) == 2 and original_data.shape[1] == 2:
        original_data = original_data[:, 0]

    # Apply low-pass filter
    original_data = apply_lowpass_filter(original_data, original_rate)

    duration = original_data.shape[0] / original_rate
    resampled_sample_count = int(duration * 8000)

    # Resample the mono data
    resampled_data = resample(original_data, resampled_sample_count).astype(np.int16)

    wavfile.write(original_file_path, 8000, resampled_data)

def process_file(filepath):
    try:
        sample_rate = get_sample_rate(filepath)
        if sample_rate != 8000:
            print(f'Downsampling: {filepath}')
            downsample_to_8k(filepath)
        else:
            print(f'Skipping (already 8k): {filepath}')
    except Exception as e:
        print(f"Error processing {filepath}. Error: {e}")

    # Explicitly release memory
    gc.collect()

def process_files_chunk(files_chunk):
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(process_file, files_chunk)

def main():
    # Create a list of all wav files to process
    wav_files = [os.path.join(dirpath, filename) 
                 for dirpath, _, filenames in os.walk('.')
                 for filename in filenames if filename.endswith('.wav')]
    
    # Split files into chunks of 10 (adjust as needed)
    chunk_size = 10
    for i in range(0, len(wav_files), chunk_size):
        process_files_chunk(wav_files[i:i + chunk_size])

if __name__ == '__main__':
    main()

