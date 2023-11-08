from util.data import Database
import os
from tqdm import tqdm

num_samples = int(880*12.5)
path_base = "/data"
out_dir = os.path.join(path_base, "noise", "processed")

db_indices = ["a", "b", "c"]

for db_index in db_indices:
    print("Processing database", db_index)

    file_db = f"db-{db_index}.sqlite3"

    file_samples = f"samples_{db_index}.npy"
    file_ids = f"ra_sat_{db_index}.npy"
    file_cells = f"ra_cell_{db_index}.npy"
    file_magnitudes = f"magnitudes_{db_index}.npy"
    file_noises = f"noises_{db_index}.npy"
    file_levels = f"levels_{db_index}.npy"
    file_confidences = f"confidences_{db_index}.npy"

    db = Database(os.path.join(path_base, file_db), num_samples)

    db.save_arrays_noise(
        os.path.join(out_dir, file_samples),
        os.path.join(out_dir, file_ids),
        os.path.join(out_dir, file_cells),
        os.path.join(out_dir, file_magnitudes),
        os.path.join(out_dir, file_noises),
        os.path.join(out_dir, file_levels),
        os.path.join(out_dir, file_confidences),
    )
