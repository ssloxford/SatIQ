from util.data import Database

from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import geopy.distance

import argparse

num_samples = int(880*12.5)


# Get a unique ID for the given id/cell pair
def get_id_cell(sat_id, sat_cell, num_cells=63):
    return (sat_id * num_cells) + sat_cell


def load_weather(path, timestamp_base):
    df_weather = pd.read_csv(path)
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    df_weather['hours'] = ((df_weather['datetime'] - timestamp_base) / pd.Timedelta(1, 'h')).astype(int)
    df_weather.set_index('hours', inplace=True)
    df_weather = df_weather[~df_weather.index.duplicated(keep='first')]
    return df_weather


def process_dataset(db_path, path_out, prefix_out, df_weather, timestamp_base, distance_to, scale=True, chunk_size=50000, keep_extras=False, verbose=False):
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    db = Database(db_path, num_samples)

    if db.num_samples == 0:
        return

    db.generate_arrays_noise()

    num_entries = db.samples_array.shape[0]
    num_batches = num_entries // chunk_size
    if keep_extras and num_entries % chunk_size != 0:
        num_batches += 1

    for i in tqdm(range(num_batches), disable=not verbose, leave=False):
        with tf.io.TFRecordWriter(os.path.join(path_out, f"{prefix_out}_{i}.tfrecord")) as writer:
            for j in tqdm(range(i * chunk_size, min((i + 1) * chunk_size, num_entries)), disable=not verbose, leave=False):
                sample = db.samples_array[j]
                if scale:
                    scale_1 = np.min(sample)
                    sample -= scale_1
                    scale_2 = np.max(sample)
                    sample /= scale_2
                else:
                    scale_1 = 0
                    scale_2 = 1
                sample = sample.flatten().tolist()
                timestamp = db.timestamps_array[j]
                if df_weather is not None:
                    timestamp_index = int((pd.Timestamp(timestamp) - timestamp_base) / pd.Timedelta(1, 'h'))
                    ti = timestamp_index
                    while ti not in df_weather.index:
                        ti -= 1
                        if timestamp_index - ti > 24:
                            raise ValueError(f"No weather data available for timestamp {timestamp_index}.")
                    weather = df_weather.loc[ti]
                else:
                    weather = dict(temp=0.0, humidity=0.0, precip=0.0, cloudcover=0.0, solarradiation=0.0)
                position = db.positions_array[j]
                distance = geopy.distance.distance(position[:2], distance_to).km

                example = tf.train.Example(features=tf.train.Features(feature={
                    "sample": tf.train.Feature(float_list=tf.train.FloatList(value=sample)),
                    "scale": tf.train.Feature(float_list=tf.train.FloatList(value=[scale_1, scale_2])),
                    "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[db.ids_array[j]])),
                    "cell": tf.train.Feature(int64_list=tf.train.Int64List(value=[db.cells_array[j]])),
                    "id_cell": tf.train.Feature(int64_list=tf.train.Int64List(value=[get_id_cell(db.ids_array[j], db.cells_array[j])])),
                    "timestamp": tf.train.Feature(int64_list=tf.train.Int64List(value=[timestamp])),
                    "position": tf.train.Feature(float_list=tf.train.FloatList(value=position)),
                    "distance": tf.train.Feature(float_list=tf.train.FloatList(value=[distance])),
                    "magnitude": tf.train.Feature(float_list=tf.train.FloatList(value=[db.magnitudes_array[j]])),
                    "noise": tf.train.Feature(float_list=tf.train.FloatList(value=[db.noises_array[j]])),
                    "level": tf.train.Feature(float_list=tf.train.FloatList(value=[db.levels_array[j]])),
                    "confidence": tf.train.Feature(int64_list=tf.train.Int64List(value=[db.confidences_array[j]])),
                    "temp": tf.train.Feature(float_list=tf.train.FloatList(value=[weather['temp']])),
                    "humidity": tf.train.Feature(float_list=tf.train.FloatList(value=[weather['humidity']])),
                    "precip": tf.train.Feature(float_list=tf.train.FloatList(value=[weather['precip']])),
                    "cloudcover": tf.train.Feature(float_list=tf.train.FloatList(value=[weather['cloudcover']])),
                    "solarradiation": tf.train.Feature(float_list=tf.train.FloatList(value=[weather['solarradiation']])),
                }))
                writer.write(example.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Sqlite databases into TFRecord datasets.")
    parser.add_argument("--path-in", type=str, required=True, help="Input directory.")
    parser.add_argument("--path-out", type=str, required=True, help="Output directory.")
    parser.add_argument("--weather-file", type=str, default=None, help="Path to weather data.")
    parser.add_argument("--timestamp-base", type=str, default="2000-01-01", help="Base timestamp for weather data.")
    parser.add_argument("--distance-to", type=float, nargs=2, default=(51.759177133203124, -1.256461002739208), help="Coordinates to calculate distance to.")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Number of records in each file.")
    parser.add_argument("--no-scale", dest='scale', action='store_false', help="Do not scale samples.")
    parser.add_argument("-k", "--keep-extras", action='store_true', help="Keep extra records in the last file.")
    parser.add_argument("-v", "--verbose", action='store_true', help="Display progress.")
    args = parser.parse_args()

    if args.weather_file is not None:
        timestamp_base = pd.Timestamp(args.timestamp_base)
        df_weather = load_weather(args.weather_file, timestamp_base)
    else:
        print("WARNING: No weather data provided!")
        timestamp_base = None
        df_weather = None

    for file in tqdm(os.listdir(args.path_in), disable=not args.verbose):
        if file.endswith(".sqlite3"):
            prefix_out = file.split(".")[0].split("-", 1)[1]
            process_dataset(os.path.join(args.path_in, file), args.path_out, prefix_out, df_weather, timestamp_base, args.distance_to, scale=args.scale, chunk_size=args.chunk_size, keep_extras=args.keep_extras, verbose=args.verbose)
