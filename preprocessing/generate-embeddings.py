import os
from tqdm import tqdm

from util.data import TFRecordLoaderFull
from util.models import AETripletSplitConvModel
from util.processing import (
    get_embeddings,
)

import numpy as np
import tensorflow as tf

data_dir = '../data/tfrecord'
model_dir = '../data/models'
output_dir = '../data/test'

def load_model(name, checkpoint='final'):
    model = AETripletSplitConvModel(
        name,
        11000,
        2,
        [
            (64, 2),
            (64, 4),
            (32, 8),
            (32, 16),
            (32, 32),
            (32, 32),
        ],
        512,
        1e-5,
        normalization='L2',
        save_dir=model_dir
    )

    model.load_model(checkpoint)

    return model

model_names = [
    #"days-1",
    #"days-2",
    #"days-4",
    #"days-8",
    #"days-16",
    #"days-32",
    #"days-64",
    "days--1",
]
#model_names.extend([
#    f"days-32-noise-{i}"
#    for i in range(-104, -89, 2)
#])
tfrecord_files = [ f for f in os.listdir(data_dir) if f.endswith('_0.tfrecord') ]
tfrecord_files.sort()

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

first = True
for model_name in tqdm(model_names):
    model = load_model(model_name)
    os.makedirs(os.path.join(output_dir, "embeddings", model_name), exist_ok=True)

    for tfrecord_file in tqdm(tfrecord_files, leave=False):
        ds_t = TFRecordLoaderFull.from_file(os.path.join(data_dir, tfrecord_file), all_features=True, shuffle=False)

        samples = np.array([ s['sample'] for s in ds_t ])
        samples = tf.convert_to_tensor(samples)
        labels = tf.convert_to_tensor([ s['id_cell'] for s in ds_t ]).numpy()

        embeddings = get_embeddings(model, samples)

        np.save(os.path.join(output_dir, "embeddings", model_name, tfrecord_file.replace('.tfrecord', '.npy')), embeddings)
        if first:
            np.save(os.path.join(output_dir, "labels", tfrecord_file.replace('.tfrecord', '.npy')), labels)

    first = False
