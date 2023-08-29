from tqdm import tqdm
import numpy as np
import os

from util.data import NoiseArrayDataset

path_base = "/data"
in_dir = os.path.join(path_base, "processed")
out_dir = os.path.join(path_base, "filtered")

suffixes = ["a", "b", "c"]

def save_dataset(path, suffix, samples_array, ids_array, cells_array, magnitudes_array, noises_array, levels_array, confidences_array):
    file_samples = os.path.join(path, "samples-{}.npy".format(suffix))
    file_ids = os.path.join(path, "ids-{}.npy".format(suffix))
    file_cells = os.path.join(path, "cells-{}.npy".format(suffix))
    file_magnitudes = os.path.join(path, "magnitudes-{}.npy".format(suffix))
    file_noises = os.path.join(path, "noises-{}.npy".format(suffix))
    file_levels = os.path.join(path, "levels-{}.npy".format(suffix))
    file_confidences = os.path.join(path, "confidences-{}.npy".format(suffix))

    np.save(file_samples, samples_array)
    np.save(file_ids, ids_array)
    np.save(file_cells, cells_array)
    np.save(file_magnitudes, magnitudes_array)
    np.save(file_noises, noises_array)
    np.save(file_levels, levels_array)
    np.save(file_confidences, confidences_array)


def process(path_in, path_out, suffix):
    print("Processing dataset {}".format(suffix))

    file_samples = os.path.join(path_in, "samples-{}.npy".format(suffix))
    file_ids = os.path.join(path_in, "ids-{}.npy".format(suffix))
    file_cells = os.path.join(path_in, "cells-{}.npy".format(suffix))
    file_magnitudes = os.path.join(path_in, "magnitudes-{}.npy".format(suffix))
    file_noises = os.path.join(path_in, "noises-{}.npy".format(suffix))
    file_levels = os.path.join(path_in, "levels-{}.npy".format(suffix))
    file_confidences = os.path.join(path_in, "confidences-{}.npy".format(suffix))

    print("Loading ArrayDataset")
    ds = NoiseArrayDataset.from_files(
        file_samples,
        file_ids,
        file_cells,
        file_magnitudes,
        file_noises,
        file_levels,
        file_confidences,
    )

    print("Dataset size: {}".format(len(ds.samples_array)))

    print("Filtering dataset")
    ds = ds.filter_zeros()

    print("Dataset size: {}".format(len(ds.samples_array)))

    print("Scaling dataset")
    for s in ds.samples_array:
        s -= np.min(s)
        s /= np.max(s)

    print("Filtering out identical samples")
    ds = ds.filter_identical(distance=1.0)

    print("Dataset size: {}".format(len(ds.samples_array)))

    print("Saving dataset")
    save_dataset(
        path_out,
        suffix,
        ds.samples_array,
        ds.ids_array,
        ds.cells_array,
        ds.magnitudes_array,
        ds.noises_array,
        ds.levels_array,
        ds.confidences_array,
    )

if __name__ == "__main__":
    for suffix in suffixes:
        process(in_dir, out_dir, suffix)
