from tqdm import tqdm
import numpy as np
import os

from util.data import ArrayDataset

path_base = "/data"
in_dir = os.path.join(path_base, "processed")
out_dir = os.path.join(path_base, "filtered")

suffixes = ["a", "b", "c"]

def save_dataset(path, suffix, samples_array, ids_array, cells_array):
    file_samples = os.path.join(path, "samples_{}.npy".format(suffix))
    file_ids = os.path.join(path, "ra_sat_{}.npy".format(suffix))
    file_cells = os.path.join(path, "ra_cell_{}.npy".format(suffix))

    np.save(file_samples, samples_array)
    np.save(file_ids, ids_array)
    np.save(file_cells, cells_array)

def process(path_in, path_out, suffix):
    file_samples = os.path.join(path_in, "samples_{}.npy".format(suffix))
    file_ids = os.path.join(path_in, "ra_sat_{}.npy".format(suffix))
    file_cells = os.path.join(path_in, "ra_cell_{}.npy".format(suffix))

    print("Loading ArrayDataset")
    ds = ArrayDataset.from_files(
        file_samples,
        file_ids,
        file_cells
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
        ds.cells_array
    )

if __name__ == "__main__":
    for suffix in suffixes:
        process(in_dir, out_dir, suffix)
