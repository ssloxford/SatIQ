from tqdm import tqdm
import numpy as np
import os
import tensorflow as tf

path_base = "/data"
path_in = os.path.join(path_base, "filtered")
path_out = os.path.join(path_base, "tfrecord")

suffixes = ["a", "b", "c"]

chunk_size = 50000
shuffle = True

# Whether to save only the most common IDs
by_id = False
# How many IDs to save
id_counts = [
    50,
]

# Get a unique ID for the given id/cell pair
def get_id_cell(sat_id, sat_cell, num_cells=63):
    return (sat_id * num_cells) + sat_cell

def load_dataset(path, suffix):
    file_samples = os.path.join(path, "samples-{}.npy".format(suffix))
    file_ids = os.path.join(path, "ids-{}.npy".format(suffix))
    file_cells = os.path.join(path, "cells-{}.npy".format(suffix))

    samples_array = np.load(file_samples)
    ids_array = np.load(file_ids)
    cells_array = np.load(file_cells)

    return samples_array, ids_array, cells_array

def save_dataset(path, suffix, samples_array, ids_array, cells_array):
    path_tfrecord = os.path.join(path, "data-{}.tfrecord".format(suffix))

    with tf.io.TFRecordWriter(path_tfrecord) as writer:
        for i in range(samples_array.shape[0]):
            sample = samples_array[i]
            sample_list = sample.flatten().tolist()
            id = ids_array[i]
            cell = cells_array[i]
            id_cell = get_id_cell(id, cell)

            example = tf.train.Example(features=tf.train.Features(feature={
                "sample": tf.train.Feature(float_list=tf.train.FloatList(value=sample_list)),
                "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[id])),
                "cell": tf.train.Feature(int64_list=tf.train.Int64List(value=[cell])),
                "id_cell": tf.train.Feature(int64_list=tf.train.Int64List(value=[id_cell])),
            }))
            writer.write(example.SerializeToString())

def save_dataset_batches(path, chunk_size, samples_array, ids_array, cells_array, verbose):
    chunk_count = 0

    while samples_array.shape[0] >= chunk_size:
        if verbose:
            print(f"Saving chunk {chunk_count}...")
            print(f"Samples remaining: {samples_array.shape[0]}")
        # Take first chunk_size samples
        s = samples_array[:chunk_size]
        i = ids_array[:chunk_size]
        c = cells_array[:chunk_size]

        # Remove first chunk_size samples
        samples_array = samples_array[chunk_size:]
        ids_array = ids_array[chunk_size:]
        cells_array = cells_array[chunk_size:]

        save_dataset(path, str(chunk_count), s, i, c)
        chunk_count += 1

    if verbose:
        print(f"Saving chunk {chunk_count}...")
        print(f"Samples remaining: {samples_array.shape[0]}")
    save_dataset(path, str(chunk_count), samples_array, ids_array, cells_array)
    chunk_count += 1

    return chunk_count

def process_all(chunk_size=50000, verbose=False):
    samples_array = None
    ids_array = None
    cells_array = None

    message_count = 0

    for suffix in tqdm(suffixes, disable=not verbose):
        s, i, c = load_dataset(path_in, suffix)
        message_count += s.shape[0]

        if samples_array is None:
            samples_array = s.copy()
        else:
            samples_array = np.append(samples_array, s, axis=0)
        if ids_array is None:
            ids_array = i.copy()
        else:
            ids_array = np.append(ids_array, i, axis=0)
        if cells_array is None:
            cells_array = c.copy()
        else:
            cells_array = np.append(cells_array, c, axis=0)

        del s
        del i
        del c

    if by_id:
        id_cells_array = np.array([get_id_cell(ids_array[i], cells_array[i]) for i in range(ids_array.shape[0])])
        id_cells_unique, id_cells_counts = np.unique(id_cells_array, return_counts=True)

        id_cells_unique = id_cells_unique[np.argsort(id_cells_counts)][::-1]
        id_cells_counts = np.sort(id_cells_counts)[::-1]

        for id_count in id_counts:
            if verbose:
                print(f"Saving {id_count} IDs...")

            path_out_id = os.path.join(path_out, f"{id_count}")

            id_cells_unique_subset = id_cells_unique[:id_count]
            idx = np.isin(id_cells_array, id_cells_unique_subset)
            samples_array_unique_subset = samples_array[idx].copy()
            ids_array_unique_subset = ids_array[idx].copy()
            cells_array_unique_subset = cells_array[idx].copy()

            if verbose:
                print(f"Number of samples: {samples_array_unique_subset.shape[0]}")

            if shuffle:
                if verbose:
                    print("Shuffling data...")
                idx = np.random.permutation(samples_array_unique_subset.shape[0])
                samples_array_unique_subset = samples_array_unique_subset[idx]
                ids_array_unique_subset = ids_array_unique_subset[idx]
                cells_array_unique_subset = cells_array_unique_subset[idx]
                if verbose:
                    print("Done")

            save_dataset_batches(path_out_id, chunk_size, samples_array_unique_subset, ids_array_unique_subset, cells_array_unique_subset, verbose)

            del samples_array_unique_subset
            del ids_array_unique_subset
            del cells_array_unique_subset

            if verbose:
                print(f"Done")

        # Test dataset
        if verbose:
            print("Saving test dataset...")

        path_out_test = os.path.join(path_out, "test")

        id_count_max = id_counts[-1]
        id_cells_unique_subset = id_cells_unique[id_count_max:]
        idx = np.isin(id_cells_array, id_cells_unique_subset)
        samples_array_unique_subset = samples_array[idx].copy()
        ids_array_unique_subset = ids_array[idx].copy()
        cells_array_unique_subset = cells_array[idx].copy()

        if verbose:
            print(f"Number of samples: {samples_array_unique_subset.shape[0]}")

        if shuffle:
            if verbose:
                print("Shuffling data...")
            idx = np.random.permutation(samples_array_unique_subset.shape[0])
            samples_array_unique_subset = samples_array_unique_subset[idx]
            ids_array_unique_subset = ids_array_unique_subset[idx]
            cells_array_unique_subset = cells_array_unique_subset[idx]
            if verbose:
                print("Done")

        save_dataset_batches(path_out_test, chunk_size, samples_array_unique_subset, ids_array_unique_subset, cells_array_unique_subset, verbose)

        del samples_array_unique_subset
        del ids_array_unique_subset
        del cells_array_unique_subset

        if verbose:
            print(f"Done")

    else:
        if shuffle:
            if verbose:
                print("Shuffling data...")
            idx = np.random.permutation(samples_array.shape[0])
            samples_array = samples_array[idx]
            ids_array = ids_array[idx]
            cells_array = cells_array[idx]
            if verbose:
                print("Done")

        chunk_count = save_dataset_batches(path_out, chunk_size, samples_array, ids_array, cells_array, verbose)

        if verbose:
            print("Total messages: {}".format(message_count))
            print("Total chunks: {}".format(chunk_count))

if __name__ == "__main__":
    process_all(chunk_size=chunk_size, verbose=True)
