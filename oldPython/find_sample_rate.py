import numpy as np

# Reading the binary data from the file
with open("predicted_output_from_epoch_10_full_recording.wav", "rb") as f:
    data = f.read()

# Skipping the header and converting to numpy array
raw_audio_data = data[44:]
audio_array = np.frombuffer(raw_audio_data, dtype=np.int16)

# Known duration in seconds
duration_seconds = 14 * 60 + 50  # 14 minutes and 50 seconds

# Inferring the sampling rate
samples = len(audio_array)
sampling_rate = samples / duration_seconds
print(f"Inferred sampling rate: {sampling_rate} Hz")

