import os
import numpy as np

from util.models import AETripletSplitConvModel
from util.data import TFRecordLoader
from util.model_utils import RocAucCallback

import tensorflow as tf
tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0],
    True
)

ids_removed = False

data_base = "/data"
save_dir = data_base + "/models"
save_dir_plots = data_base + "/model-plots"
if ids_removed:
    data_dir = data_base + "/tfrecord-ids-removed/train"
    num_files = 20
else:
    data_dir = data_base + "/tfrecord-shuffled"
    num_files = 21

use_magnitude = False
magnitude_percentage = 30
data_dir_magnitude = data_base + "/noise/tfrecord-magnitude"
num_files_magnitude = {
    10: 3,
    20: 5,
    30: 7,
    40: 9,
    50: 11,
    60: 13,
    70: 15,
    80: 17,
    90: 19,
}

num_samples = 11000 # = 880*12.5
group_window_size = 4 # Number of samples with the same ID to group together
batch_size = 32
num_epochs = 200
seed = 20220615

shuffle_buffer_file = 15
shuffle_buffer_sample = 10000

layers = [
    (64, 2),
    (64, 4),
    (32, 8),
    (32, 16),
    (32, 32),
    (32, 32),
]
num_layers = len(layers)
latent_dim = 512

triplet_margin = 1.0 # Margin for triplet loss function
triplet_distance_metric = 'angular' # L2, squared-L2, angular
normalization = 'L2' # L2, L1, None

learning_rate = 1e-5 #1e-5

if use_magnitude:
    model_name = f'ae-triplet-magnitude-{magnitude_percentage}'
elif ids_removed:
    model_name = 'ae-triplet-ids-removed'
else:
    model_name = 'ae-triplet'

if use_magnitude:
    num_files = num_files_magnitude[magnitude_percentage]
    files_in = [ os.path.join(data_dir_magnitude, str(magnitude_percentage), "data-{}.tfrecord".format(i)) for i in range(num_files) ]
    file_val = files_in[-1]
    file_test = file_val
    files_train = files_in[:-1]
    shuffle_buffer_file = min(shuffle_buffer_file, len(files_train))
else:
    files_in = [ os.path.join(data_dir, "data-{}.tfrecord".format(i)) for i in range(num_files) ]
    if ids_removed:
        file_val = files_in[-1]
        files_train = files_in[:-1]
    else:
        file_val = files_in[4]
        file_test = files_in[8]
        files_train = files_in[0:4] + files_in[5:8] + files_in[9:]

cycle_length = 4 # Number of files to read in parallel

print(f"Training files: {files_train}")
ds_train = TFRecordLoader.from_files(files_train, shuffle_buffer_file, cycle_length, shuffle_buffer_sample, seed)
ds_train = TFRecordLoader.window_batch(ds_train, group_window_size, batch_size)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

print(f"Validation file: {file_val}")
ds_val = TFRecordLoader.from_file(file_val, shuffle_buffer_sample, seed)
ds_val = TFRecordLoader.window_batch(ds_val, group_window_size, batch_size)
ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

print("Initialising model")
model = AETripletSplitConvModel(
    model_name,
    num_samples,
    2,
    layers,
    latent_dim,
    learning_rate,
    triplet_margin=triplet_margin,
    triplet_distance_metric=triplet_distance_metric,
    normalization=normalization,
    save_dir=save_dir
)

print("Loading checkpoint")
try:
    model.load_model(suffix='checkpoint')
except ValueError:
    print("No checkpoint found, model not loaded.")

model.model.summary()

model.fit(
    ds_train,
    validation_data=ds_val,
    validation_steps=200,
    batch_size=batch_size,
    epochs=num_epochs,
    save_epochs=True,
)

model.save_model(
    suffix='final',
)
