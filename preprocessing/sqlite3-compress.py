import sqlite3
import argparse
import os
import numpy as np
import base64

tables = dict(
    iq_samples = ["run_id", "magnitude", "noise", "msg_id", "timestamp", "timestamp_global", "uw_start", "direction", "center_frequency", "sample_rate", "sample_count", "samples"],
    ira_messages = ["msg_type", "msg", "ra_lat", "ra_lon", "ra_alt", "ra_sat", "ra_cell"],
    bytes = ["n_symbols", "level", "confidence", "bytes"],
)
rows = [ column for table in tables for column in tables[table] ]
id_column = "id"
samples_column = "samples"
waveform_len = 11000
bytes_column = "bytes"
max_bits = 888

rows_per_file = 10000


def process_data_lists(data_lists):
    """
    Process the data lists into a set of numpy arrays, reformatting the samples and bytes.
    """

    data_arrays = []

    for i, row in enumerate(rows):
        if row == samples_column:
            arr = np.array([ np.frombuffer(base64.b64decode(samples), dtype=np.complex64)[:waveform_len].view(np.float32).reshape((waveform_len, 2,)) for samples in data_lists[i] ])
        elif row == bytes_column:
            for j in range(len(data_lists[i])):
                # Pad to max_bits
                data_lists[i][j] = data_lists[i][j].ljust(max_bits, '0')
            arr = np.array([ np.array([ int(d[i:i+8], 2) for i in range(0, len(d), 8) ], dtype=np.uint8) for d in data_lists[i] ])
        else:
            arr = np.array(data_lists[i])

        data_arrays.append(arr)

    return data_arrays

def load_databases(db_files, out_dir):
    """
    Load the databases from the given files, and extract the data into array files.
    """
    data_lists = [ [] for _ in range(len(rows)) ]
    file_count = 0
    rows_count = 0

    for db_file in db_files:
        print(f"Processing {db_file}...")

        conn = sqlite3.connect(db_file)

        # Compose query to get all the data from the database using the "tables" dictionary
        # Join the tables using the "id_column" variable
        query = f"SELECT {', '.join([f'{table}.{column}' for table in tables for column in tables[table]])} FROM {list(tables.keys())[0]}"
        for i, table in enumerate(tables):
            if i > 0:
                query += f" JOIN {table} ON {table}.{id_column} = {list(tables.keys())[0]}.{id_column}"

        cur = conn.execute(query)

        while (elem := cur.fetchone()) is not None:
            for i, e in enumerate(elem):
                data_lists[i].append(e)

            rows_count += 1

            if rows_count % rows_per_file == 0:
                # Save the data to a file
                print(f"Saving file {file_count}...")

                data_arrays = process_data_lists(data_lists)
                #np.savez_compressed(f"{args.out_dir}/data-{file_count:03d}.npz", *data_arrays)
                for i, row in enumerate(rows):
                    np.save(f"{out_dir}/{row}_{file_count:03d}.npy", data_arrays[i])
                file_count += 1
                data_lists = [ [] for _ in range(len(rows)) ]

    # Save the remaining data to a file
    print(f"Saving file {file_count}...")

    data_arrays = process_data_lists(data_lists)
    for i, row in enumerate(rows):
        np.save(f"{out_dir}/{row}_{file_count:03d}.npy", data_arrays[i])
    file_count += 1
    data_lists = [ [] for _ in range(len(rows)) ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", help="Input directory")
    parser.add_argument("out_dir", help="Output directory")

    args = parser.parse_args()

    files_in = [ f for f in os.listdir(args.in_dir) if f.endswith(".sqlite3") ]
    files_in.sort(key=lambda f: int(f.split("-")[1].split(".")[0])) # Sort by their number
    files_in = [ os.path.join(args.in_dir, f) for f in files_in ]

    os.makedirs(args.out_dir, exist_ok=True)

    load_databases(files_in, args.out_dir)