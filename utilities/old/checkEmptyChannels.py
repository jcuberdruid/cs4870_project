import os
import wave
import numpy as np

def get_empty_channels(wav_file_path):
    """
    Returns a list of empty channels in the given wav file.
    """
    with wave.open(wav_file_path, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        frames = wav_file.readframes(-1)
        wave_data = np.frombuffer(frames, dtype=np.int16)
        wave_data = wave_data.reshape(-1, n_channels)  # Reshape to have one column per channel

        empty_channels = []
        for channel in range(n_channels):
            if np.all(wave_data[:, channel] == 0):  # Check if all values in the channel are zero
                empty_channels.append(channel + 1)  # Channels are 1-indexed

        return empty_channels

def main():
    count_empty_channel_1 = 0
    count_empty_channel_2 = 0

    for dirpath, dirnames, filenames in os.walk('.'):
        for filename in filenames:
            if filename.endswith('.wav'):
                filepath = os.path.join(dirpath, filename)
                try:
                    empty_channels = get_empty_channels(filepath)
                    if 1 in empty_channels:
                        count_empty_channel_1 += 1
                    if 2 in empty_channels:
                        count_empty_channel_2 += 1
                except Exception as e:
                    print(f"Error reading {filepath}. Error: {e}")

    print(f"Number of files with empty Channel 1: {count_empty_channel_1}")
    print(f"Number of files with empty Channel 2: {count_empty_channel_2}")

if __name__ == '__main__':
    main()

