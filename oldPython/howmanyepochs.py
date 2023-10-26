import numpy as np
import tensorflow as tf
import scipy.io.wavfile
from tensorflow.keras.layers import Input, Dense, LayerNormalization
from tensorflow.keras.models import Model
from scipy.signal import firwin, lfilter
import csv

tf.keras.mixed_precision.set_global_policy('mixed_float16')

test_path = "../data/A3CarScene_AudioData/audio-ch2-20221017-2.wav"

def lowpass_filter(data, cutoff_freq, fs=44100, numtaps=101):
    coeffs = firwin(numtaps, cutoff_freq, fs=fs, pass_zero='lowpass')
    filtered_data = lfilter(coeffs, 1.0, data)
    return filtered_data

def downsample_1d(data, factor):
    return np.mean(data.reshape(-1, factor), axis=1)

rate, data = scipy.io.wavfile.read(test_path)
print(rate)
data = lowpass_filter(data, cutoff_freq=18e3, fs=rate)
data = downsample_1d(data, 2)

def create_dataset_gen(data, input_window_size=50, output_window_size=10):
    for i in range(len(data) - input_window_size - output_window_size + 1):
        X = data[i:i+input_window_size]
        Y = data[i+input_window_size:i+input_window_size+output_window_size]
        yield np.array(X, dtype=np.float32).reshape(input_window_size, 1), np.array(Y, dtype=np.float32)

input_window_size = 4000
output_window_size = 1000

dataset = tf.data.Dataset.from_generator(
    lambda: create_dataset_gen(data, input_window_size, output_window_size),
    (tf.float32, tf.float32),
    (tf.TensorShape([input_window_size, 1]), tf.TensorShape([output_window_size]))
)

train_size = int(0.8 * dataset.cardinality().numpy())
train_dataset = dataset.take(train_size).batch(18).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = dataset.skip(train_size).batch(18).prefetch(tf.data.experimental.AUTOTUNE)


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

embed_size = 112
num_heads = 4
forward_expansion = 242

inputs = Input(shape=(input_window_size, 1))
x = Dense(embed_size)(inputs)
x = TransformerBlock(embed_size, num_heads, forward_expansion)(x)
x = Dense(output_window_size)(x[:, -1, :])

dataset_length = len(data) - input_window_size - output_window_size + 1
train_size = int(0.8 * dataset_length)
total_steps = train_size // 18  # Assuming batch size of 47

print(f"###############################################################")
print(f"Total steps per epoch: {total_steps}")
print(f"###############################################################")

model = Model(inputs=inputs, outputs=x)
model.compile(loss="mse", optimizer="adam")

history = model.fit(train_dataset, validation_data=test_dataset, epochs=3)

model.save("testModel.keras")
predictions = model.predict(test_dataset.unbatch().batch(1))

predicted_sound = np.concatenate(predictions)
predicted_sound = np.int16(predicted_sound/np.max(np.abs(predicted_sound)) * 32767)
scipy.io.wavfile.write('predicted_output.wav', rate//2, predicted_sound)

with open('history.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Training Loss", "Validation Loss"])
    for epoch, train_loss, val_loss in zip(range(len(history.history['loss'])), history.history['loss'], history.history['val_loss']):
        writer.writerow([epoch, train_loss, val_loss])
