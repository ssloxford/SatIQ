from util.data import Database
import os

num_samples = int(880*12.5)
path_base = "/data"

db_indices = ["a", "b", "c"]

for db_index in db_indices:
    file_db = f"db-{db_index}.sqlite3"

    out_dir = os.path.join(path_base, "processed")
    file_samples = f"samples_{db_index}.npy"
    file_ids = f"ra_sat_{db_index}.npy"
    file_cells = f"ra_cell_{db_index}.npy"

    db = Database(os.path.join(path_base, file_db), num_samples)

    db.save_arrays(
        os.path.join(out_dir, file_samples),
        os.path.join(out_dir, file_ids),
        os.path.join(out_dir, file_cells),
        ira_only=True,
        message_only=False
    )
