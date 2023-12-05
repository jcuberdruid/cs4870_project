import os
import wave

def get_sample_rate(wav_file_path):
    """
    Returns the sample rate of the given wav file.
    """
    with wave.open(wav_file_path, 'rb') as wav_file:
        print(wav_file.getnchannels())
        print(wav_file.getnframes())
        return wav_file.getframerate()

def main():
    for dirpath, dirnames, filenames in os.walk('.'):
        for filename in filenames:
            if filename.endswith('.wav'):
                filepath = os.path.join(dirpath, filename)
                try:
                    sample_rate = get_sample_rate(filepath)
                    print(f'File: {filepath}, Sample Rate: {sample_rate}')
                except Exception as e:
                    print(f"Error reading {filepath}. Error: {e}")

if __name__ == '__main__':
    main()

