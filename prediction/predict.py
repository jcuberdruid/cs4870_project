import os
import re
import numpy as np
import tensorflow as tf
import scipy.io.wavfile
from keras.layers import Input, Dense, LayerNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import random
import csv
import glob

tf.keras.mixed_precision.set_global_policy('mixed_float16')

percentage = .05
BATCH_SIZE = 400
input_window_size = 1000
output_window_size = 20
new_path = "../../data/2Khz_wav"
#new_path = "../../data/wav_files"
EPOCHS = 25
experiment = "6fd4622c-f001-485c-b49b-37ddc57badf1"
model_dir = f"../experiments/{experiment}/"
#epoch = 1
#MODEL_SAVE_PATH_PATTERN = f"../experiments/{experiment}/model_at_epoch_{epoch}.h5"
#MODEL_SAVE_PATH_PATTERN = f"../experiments/7bf64a2e-05fe-4dc7-9f48-23efcdbc1011/model_at_epoch_{epoch}.h5"

def si_snr(y_true, y_pred, epsilon=1e-4):
    """
    Scale-Invariant Signal-to-Noise Ratio (Si-SNR) loss for float16.
    :param y_true: the ground truth audio signal (float16).
    :param y_pred: the predicted audio signal (float16).
    :param epsilon: a small constant to avoid division by zero.
    :return: the Si-SNR value.
    """
    y_true_f32 = tf.cast(y_true, tf.float32)
    y_pred_f32 = tf.cast(y_pred, tf.float32)
    y_true_zm = y_true_f32 - tf.reduce_mean(y_true_f32, axis=-1, keepdims=True)
    y_pred_zm = y_pred_f32 - tf.reduce_mean(y_pred_f32, axis=-1, keepdims=True)
    dot_product = tf.reduce_sum(y_true_zm * y_pred_zm, axis=-1, keepdims=True)
    s_target = dot_product * y_true_zm / (tf.reduce_sum(y_true_zm ** 2, axis=-1, keepdims=True) + epsilon)
    e_noise = y_pred_zm - s_target
    si_snr_value = 10 * tf.math.log(tf.reduce_sum(s_target ** 2, axis=-1) / (tf.reduce_sum(e_noise ** 2, axis=-1) + epsilon) + epsilon) / tf.math.log(10.0)
    return -tf.reduce_mean(si_snr_value) 

def mse_stft_combined_loss(alpha=0.5):
	def loss(y_true, y_pred):
		y_true_float32 = tf.cast(y_true, tf.float32)
		y_pred_float32 = tf.cast(y_pred, tf.float32)
		mse_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true_float32, y_pred_float32))
		stft_true = tf.signal.stft(y_true_float32, frame_length=128, frame_step=64)
		stft_pred = tf.signal.stft(y_pred_float32, frame_length=128, frame_step=64)
		stft_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.abs(stft_true), tf.abs(stft_pred)))
		combined_loss = alpha * mse_loss + (1 - alpha) * stft_loss
		return combined_loss
	return loss

def find_last_saved_model(model_dir):
	max_epoch = -1
	last_model_path = None
	for file in os.listdir(model_dir):
		match = re.search(r"model_at_epoch_(\d+)", file)
		if match:
			epoch = int(match.group(1))
			if epoch > max_epoch:
				max_epoch = epoch
				last_model_path = os.path.join(model_dir, file)
	return last_model_path, max_epoch

all_files = glob.glob(os.path.join(new_path, '**/*.wav'), recursive=True)
if len(all_files) < 10:
	raise ValueError("Not enough .wav files in the directory.")
	
random.shuffle(all_files)

num_files_to_use = int(len(all_files) * percentage)
selected_files = all_files[:num_files_to_use]  # Selects the first 'num_files_to_use' files
#print(selected_files)

def create_dataset_gen(selected_files, input_window_size=5000, output_window_size=1200):
	for idx, file_path in enumerate(selected_files):
		rate, data = scipy.io.wavfile.read(file_path)  
		data = data / np.iinfo(np.int16).max  
		data = data.astype(np.float32) 
		for i in range(0, len(data) - input_window_size - output_window_size + 1, output_window_size):
			X = data[i:i+input_window_size]
			Y = data[i+input_window_size:i+input_window_size+output_window_size]
			yield X.reshape(input_window_size, 1), Y.reshape(output_window_size, 1)
		#print(f"Processing file {idx+1}/{len(selected_files)}: {file_path}", end='\r')
	print("\nFinished processing files.")
	
dataset = tf.data.Dataset.from_generator(
	lambda: create_dataset_gen(selected_files, input_window_size, output_window_size),
	output_signature=(
		tf.TensorSpec(shape=(input_window_size, 1), dtype=tf.float32),
		tf.TensorSpec(shape=(output_window_size, 1), dtype=tf.float32),
	)
)

train_size = int(1 * dataset.cardinality().numpy())
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
		self.ffn = tf.keras.Sequential([
			tf.keras.layers.Dense(forward_expansion * embed_size, activation="gelu"),
			tf.keras.layers.Dense(embed_size),
		])
		self.dropout = tf.keras.layers.Dropout(rate)

	def call(self, x, training):
		attn_output = self.attention(x, x)
		out1 = self.norm1(x + attn_output)
		ffn_output = self.ffn(out1)
		ffn_output = self.dropout(ffn_output, training=training)
		return self.norm2(out1 + ffn_output)

embed_size = 500
num_heads = 4
forward_expansion = 1

inputs = Input(shape=(input_window_size, 1))
x = TransformerBlock(embed_size, num_heads, forward_expansion)(inputs)
x = Dense(output_window_size)(x[:, -1, :])

last_model_path, last_epoch = find_last_saved_model(model_dir)
MODEL_SAVE_PATH_PATTERN = f"{model_dir}model_at_epoch_{last_epoch+1}.h5"
print(MODEL_SAVE_PATH_PATTERN)

opt= Adam(learning_rate=0.001)
if last_model_path and os.path.exists(last_model_path):
	print(f"Resuming from epoch {last_epoch} using model {last_model_path}")
	model = tf.keras.models.load_model(last_model_path, custom_objects={"TransformerBlock": TransformerBlock}, compile=False)
	model.compile(optimizer='adam', loss='mse')
	#model.compile(optimizer=opt, loss=si_snr)
else:
	print("Starting from scratch.")
	model = Model(inputs=inputs, outputs=x)
	model.compile(loss='mse', optimizer='adam')
	#model.compile(optimizer=opt, loss=si_snr)

model_checkpoint_cb = ModelCheckpoint(MODEL_SAVE_PATH_PATTERN, save_best_only=False, save_weights_only=False)
save_epoch_loss_cb = SaveEpochLossCallback('epoch_loss.csv')

model.summary()
initial_epoch = last_epoch if last_epoch != -1 else 0
history = model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, initial_epoch=initial_epoch, callbacks=[save_epoch_loss_cb, model_checkpoint_cb])
