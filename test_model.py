import numpy as np
import tensorflow as tf
import scipy.io.wavfile
from tensorflow.keras.layers import Dense, LayerNormalization

test_path = "../data/A3CarScene_AudioData/audio-ch2-20221017-2.wav"
#test_path = "./audio-ch2-20221017-2.wav"

def lowpass_filter(data, cutoff_freq, fs=44100, numtaps=101):
    from scipy.signal import firwin, lfilter
    coeffs = firwin(numtaps, cutoff_freq, fs=fs, pass_zero='lowpass')
    filtered_data = lfilter(coeffs, 1.0, data)
    return filtered_data

def downsample_1d(data, factor):
    return np.mean(data.reshape(-1, factor), axis=1)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, num_heads, forward_expansion, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_size)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential(
            [Dense(forward_expansion, activation="relu"), Dense(embed_size),]
        )

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        attn_output = self.attention(x, x, return_attention_scores=False)
        out1 = self.norm1(x + attn_output)
        ffn_output = self.ffn(out1)
        return self.norm2(out1 + ffn_output)


rate, data = scipy.io.wavfile.read(test_path)
data_length = len(data)
data = data[:int(.2 * data_length)]
data = lowpass_filter(data, cutoff_freq=18e3, fs=rate)
data = downsample_1d(data, 4)

input_window_size = 5000
output_window_size = 1125

reconstructed_audio = np.zeros(len(data))
counts = np.zeros(len(data))

model_path = "model_at_epoch_4x_downsample_4_allrecording_3090.h5"
model = tf.keras.models.load_model(model_path, custom_objects={"TransformerBlock": TransformerBlock})

# Generate predictions for overlapping windows of input data
predictions = []

stride = output_window_size
for i in range(0, len(data) - input_window_size, stride):
    input_sequence = data[i:i+input_window_size]
    predicted_sequence = model.predict(input_sequence.reshape(1, input_window_size, 1))
    predictions.append(predicted_sequence)

# Reconstruct the audio using the predictions
for idx, predicted_sequence in enumerate(predictions):
    start_idx = idx * stride
    end_idx = start_idx + output_window_size

    # Ensure that we don't go beyond the length of the reconstructed audio
    if end_idx > len(reconstructed_audio):
        end_idx = len(reconstructed_audio)
        predicted_sequence = predicted_sequence[0][:end_idx-start_idx]

    reconstructed_audio[start_idx:end_idx] += predicted_sequence[0][:end_idx-start_idx]
    counts[start_idx:end_idx] += 1

reconstructed_audio = reconstructed_audio / np.maximum(counts, 1)  # Avoid division by zero
predicted_sound = np.int16(reconstructed_audio/np.max(np.abs(reconstructed_audio)) * 32767)
scipy.io.wavfile.write('predicted_output_from_epoch_10_3080_4x_downsample_70files_5500_1125.wav', rate//4, predicted_sound)

np.save('predicted_sound.npy', reconstructed_audio)

