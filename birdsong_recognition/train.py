from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_io as tfio
import tensorflow as tf
import random
from .dataset import *

seed = 42
random.seed(seed)


running_in_colab = False

if running_in_colab:
  !pip install tensorflow_io -q
  dataset_path = 'drive/MyDrive/dataset/'
else:
  dataset_path = ''

ebirds = ['norcar', 'blujay', 'bkcchi']
all_files = []

for ebird in ebirds:
    files = glob(dataset_path + 'dataset/'+ebird+'/*')
    print('Number of files in {}: {}.'.format(ebird, len(files)))
    all_files.extend(files)

random.shuffle(all_files)

train_size = int(np.round(len(all_files) * 0.65))
val_size = int(np.round(len(all_files) * 0.15))
train_files = all_files[:train_size]
val_files = all_files[train_size: train_size+val_size]
test_files = all_files[train_size+val_size:]
print('train_ds: {}. val_ds: {}. test_ds: {}'.format(len(train_files), len(val_files), len(test_files)))

train_ds = tf.data.Dataset.from_tensor_slices(train_files)
val_ds = tf.data.Dataset.from_tensor_slices(val_files)
test_ds = tf.data.Dataset.from_tensor_slices(test_files)

train_ds = train_ds.map(get_sample_label)
val_ds = val_ds.map(get_sample_label)
test_ds = test_ds.map(get_sample_label)

train_ds = train_ds.map(preprocess_file)
val_ds = val_ds.map(preprocess_file)
test_ds = test_ds.map(preprocess_file)

iterator = iter(train_ds)
sample, label = iterator.next()
assert sample.numpy().min() == -1
assert sample.numpy().max() == 1

shapes = []
iterator = iter(train_ds)
while True:
    try:
        sample,label = iterator.next()
        shapes.append(sample.shape[0])
    except:
        break

min_file_size = np.array(shapes).min()
max_file_size = np.array(shapes).max()
print('min file size: {}, max file size: {}'.format(min_file_size, max_file_size))

"""### option 1: Using minimum sized file as the window size"""

iterator = iter(train_ds)
sample, label = iterator.next()

new_sample, new_label = split_file_by_window_size(sample, label)

train_win_ds = train_ds.map(wrapper_split_file_by_window_size)
val_win_ds = val_ds.map(wrapper_split_file_by_window_size)
test_win_ds = test_ds.map(wrapper_split_file_by_window_size)


train_samples_all, train_labels_all = create_dataset_fixed_size(train_win_ds)
val_samples_all, val_labels_all = create_dataset_fixed_size(val_win_ds)

train_ds = tf.data.Dataset.from_tensor_slices((train_samples_all, train_labels_all))
val_ds = tf.data.Dataset.from_tensor_slices((val_samples_all, val_labels_all))

train_ds = train_ds.map(get_spectrogram)
val_ds = val_ds.map(get_spectrogram)

train_ds = train_ds.map(add_channel_dim)
val_ds = val_ds.map(add_channel_dim)

"""### simple model"""

import tensorflow.keras as keras
from tensorflow.keras import layers, losses, optimizers

if running_in_colab:
    !pip install wandb -qqq

import wandb as wb
from wandb.keras import WandbCallback

wb.login()

wb.init(project='bird_ID', config={'lr': 1e-3, 'bs': 32})
config = wb.config

keras.backend.clear_session()

model = tf.keras.Sequential([
            layers.Conv2D(filters=32, kernel_size=(4,4), strides=1, activation='relu', input_shape=(284, 257, 1)),
            layers.MaxPool2D(pool_size=(4,4)),
            layers.Conv2D(filters=64, kernel_size=(4,4), strides=1, activation='relu'),
            layers.MaxPool2D(pool_size=(4,4)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(3)
])

model.summary()
model.compile(optimizer=tf.optimizers.Adam(learning_rate=config.lr), loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics='accuracy')

AUTOTUNE = tf.data.AUTOTUNE

train_ds_ = train_ds.shuffle(500, seed=seed).cache().prefetch(AUTOTUNE).batch(config.bs)
val_ds_ = val_ds.shuffle(500, seed=seed).cache().prefetch(AUTOTUNE).batch(config.bs)

#LargeDataset
model.fit(train_ds_, epochs=2, validation_data=val_ds_, callbacks=[WandbCallback()])

wb.finish()

