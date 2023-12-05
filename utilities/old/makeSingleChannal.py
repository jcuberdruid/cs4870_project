import os
import wave
import numpy as np

def stereo_to_mono(wav_file_path):
    """
    Convert a stereo WAV file to mono.
    """
    with wave.open(wav_file_path, 'rb') as original_wav:
        n_channels = original_wav.getnchannels()
        frames = original_wav.readframes(-1)
        wave_data = np.frombuffer(frames, dtype=np.int16)
        wave_data = wave_data.reshape(-1, n_channels)

        # If the file is already mono, just return
        if n_channels == 1:
            return
        
        # Take the average of the two channels
        mono_data = np.mean(wave_data, axis=1, dtype=np.int16)

        # Save the mono data back to the file
        with wave.open(wav_file_path, 'wb') as mono_wav:
            mono_wav.setnchannels(1)
            mono_wav.setsampwidth(original_wav.getsampwidth())
            mono_wav.setframerate(original_wav.getframerate())
            mono_wav.writeframes(mono_data.tobytes())

def main():
    for dirpath, dirnames, filenames in os.walk('.'):
        for filename in filenames:
            if filename.endswith('.wav'):
                filepath = os.path.join(dirpath, filename)
                try:
                    stereo_to_mono(filepath)
                except Exception as e:
                    print(f"Error processing {filepath}. Error: {e}")

if __name__ == '__main__':
    main()

