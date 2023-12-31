import os
import numpy as np
import tensorflow as tf
import scipy.io.wavfile
from tensorflow.keras.layers import Input, Dense, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import csv
import glob

tf.keras.mixed_precision.set_global_policy('mixed_float16')

new_path = "data/"

# Get all .wav files in the directory recursively
all_files = glob.glob(os.path.join(new_path, '**/*.wav'), recursive=True)
if len(all_files) < 10:
    raise ValueError("Not enough .wav files in the directory.")

# Use a percentage of files (e.g., 20%)
percentage = 0.01
num_files_to_use = int(len(all_files) * percentage)
selected_files = all_files[:num_files_to_use]  # Selects the first 'num_files_to_use' files

def create_dataset_gen(selected_files, input_window_size=50, output_window_size=10):
    for idx, file_path in enumerate(selected_files):
        rate, data = scipy.io.wavfile.read(file_path)  # Read each file individually
        data = data.astype(np.float32)  # Ensure consistent datatype

        for i in range(0, len(data) - input_window_size - output_window_size + 1, output_window_size):
            X = data[i:i+input_window_size]
            Y = data[i+input_window_size:i+input_window_size+output_window_size]
            yield X.reshape(input_window_size, 1), Y.reshape(output_window_size, 1)
        
        # Print the loading progress of files
        print(f"Processing file {idx+1}/{len(selected_files)}: {file_path}", end='\r')
    print("\nFinished processing files.")

input_window_size = 15000
output_window_size = 3000

dataset = tf.data.Dataset.from_generator(
    lambda: create_dataset_gen(selected_files, input_window_size, output_window_size),
    output_signature=(
        tf.TensorSpec(shape=(input_window_size, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(output_window_size, 1), dtype=tf.float32),
    )
)

train_size = int(1 * dataset.cardinality().numpy())
BATCH_SIZE = 2
train_dataset = dataset.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

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


exit(0)
reconstructed_audio = np.zeros(len(data))
counts = np.zeros(len(data))

for i in range(len(data) - input_window_size - output_window_size + 1):
    input_sequence = data[i:i+input_window_size]
    predicted_sequence = model.predict(input_sequence.reshape(1, input_window_size, 1))
    start_idx = i + input_window_size
    end_idx = start_idx + output_window_size
    reconstructed_audio[start_idx:end_idx] += predicted_sequence[0]

