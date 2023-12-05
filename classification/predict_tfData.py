import os
import csv
import numpy as np
import tensorflow as tf
import scipy.io.wavfile
from tensorflow.keras.layers import Input, Dense, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint
import random
import csv
import glob

# Human Vars
percentage = 0.10
BATCH_SIZE = 32
input_window_size = 3000
output_window_size = 500

tf.keras.mixed_precision.set_global_policy('mixed_float16')

new_path = "../../data/2Khz_wav/"
all_files = glob.glob(os.path.join(new_path, '**/*.wav'), recursive=True)
if len(all_files) < 10:
    raise ValueError("Not enough .wav files in the directory.")

random.shuffle(all_files)
num_files_to_use = int(len(all_files) * percentage)
selected_files = all_files[:num_files_to_use]

def load_wav(file_path):
    file_path = file_path.numpy().decode()
    rate, data = scipy.io.wavfile.read(file_path)
    data = data / np.iinfo(np.int16).max  # Normalize data
    data = data.astype(np.float32)
    num_samples = len(data)
    input_windows = []
    output_windows = []
    for start in range(0, num_samples - input_window_size - output_window_size, output_window_size):
        input_window = data[start:start + input_window_size]
        output_window = data[start + input_window_size:start + input_window_size + output_window_size]
        input_windows.append(input_window)
        output_windows.append(output_window)
    return np.array(input_windows), np.array(output_windows)

def tf_load_wav(file_path):
    input_windows, output_windows = tf.py_function(load_wav, [file_path], [tf.float32, tf.float32])
    return tf.data.Dataset.from_tensor_slices((input_windows, output_windows))

files_dataset = tf.data.Dataset.from_tensor_slices(selected_files)
dataset = files_dataset.flat_map(tf_load_wav)
dataset = dataset.map(lambda x, y: (tf.expand_dims(x, -1), tf.expand_dims(y, -1)))
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Train/Test Split
train_size = int(1 * len(selected_files))
train_dataset = dataset.take(train_size).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = dataset.skip(train_size).prefetch(tf.data.experimental.AUTOTUNE)


class SaveEpochLossCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        with open(self.filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Training Loss", "Validation Loss"])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, logs.get('loss'), logs.get('val_loss')])

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

embed_size = 32
num_heads = 4
forward_expansion = 64

inputs = Input(shape=(input_window_size, 1))
x = Dense(embed_size)(inputs)
x = TransformerBlock(embed_size, num_heads, forward_expansion)(x)
x = Dense(output_window_size)(x[:, -1, :])

model_save_path = "model_at_epoch_{epoch}_allrecording_3090.h5"
model_checkpoint_cb = ModelCheckpoint(model_save_path, save_best_only=False, save_weights_only=False)

model = Model(inputs=inputs, outputs=x)
model.compile(loss="mse", optimizer="adam")

save_epoch_loss_cb = SaveEpochLossCallback('epoch_loss.csv')
history = model.fit(train_dataset, validation_data=test_dataset, epochs=10, callbacks=[save_epoch_loss_cb, model_checkpoint_cb])


# Save history to CSV
history_file = 'training_history.csv'
with open(history_file, 'w', newline='') as file:
    writer = csv.writer(file)
    # Writing the headers
    headers = ['epoch']
    for key in history.history.keys():
        headers.extend([key])
    writer.writerow(headers)

    # Writing the data
    for epoch in range(len(history.history['loss'])):
        row = [epoch+1]
        for key in history.history:
            row.append(history.history[key][epoch])
        writer.writerow(row)

print(f"Training history saved to {history_file}")
