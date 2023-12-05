import numpy as np
import tensorflow as tf
import scipy.io.wavfile
from tensorflow.keras.layers import Dense, LayerNormalization

test_path = "./b06598b0753e4ec08669995be9c4ba03_mono_2.wav"

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

# Read the 8kHz WAV file
rate, data = scipy.io.wavfile.read(test_path)
data_length = len(data)
data = data[:int(1 * data_length)]

input_window_size = 3000
output_window_size = 500
middle_samples = 250

reconstructed_audio = np.zeros(len(data))
counts = np.zeros(len(data))

model_path = "model_at_epoch_2_allrecording_3090.h5" # this is the took all week model
model = tf.keras.models.load_model(model_path, custom_objects={"TransformerBlock": TransformerBlock})

# Generate predictions for overlapping windows of input data
predictions = []

stride = middle_samples
for i in range(0, len(data) - input_window_size, stride):
	input_sequence = data[i:i+input_window_size]
	predicted_sequence = model.predict(input_sequence.reshape(1, input_window_size, 1))
	predictions.append(predicted_sequence)

# Reconstruct the audio using the middle 250 samples from each prediction
for idx, predicted_sequence in enumerate(predictions):
	start_idx = idx * stride
	middle_start = (output_window_size - middle_samples) // 2
	middle_end = middle_start + middle_samples

	# Ensure that we don't go beyond the length of the reconstructed audio
	end_idx = start_idx + middle_samples
	if end_idx > len(reconstructed_audio):
		end_idx = len(reconstructed_audio)
		middle_end = middle_samples - (end_idx - start_idx)

	# Adjusted indexing for 1-dimensional predicted_sequence
	reconstructed_audio[start_idx:end_idx] += predicted_sequence[0, middle_start:middle_end]
	counts[start_idx:end_idx] += 1

reconstructed_audio = reconstructed_audio / np.maximum(counts, 1)  # Avoid division by zero
predicted_sound = np.int16(reconstructed_audio/np.max(np.abs(reconstructed_audio)) * 32767)

# Write the output as an 8kHz WAV file
scipy.io.wavfile.write('2Khz_recreate_2_mid250.wav', 2000, predicted_sound)

np.save('predicted_sound.npy', reconstructed_audio)

