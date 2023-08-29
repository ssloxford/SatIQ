import numpy as np
import sqlite3
import base64
from tqdm import tqdm

import tensorflow as tf

import scipy.signal

class Dataset:
    pass


class Database(Dataset):
    def __init__(self, data_file, waveform_len):
        self.data_file = data_file
        self.waveform_len = waveform_len

        self.conn = sqlite3.connect(data_file)
        self.cur = self.conn.cursor()

        self.samples_array = None
        self.ids_array = None
        self.cells_array = None

        self.magnitudes_array = None
        self.noises_array = None
        self.levels_array = None
        self.confidences_array = None

        self.cur.execute("SELECT count(*) FROM ira_messages")
        self.num_samples = self.cur.fetchall()[0][0]

        self.cur.execute("SELECT DISTINCT ra_sat FROM ira_messages WHERE ra_sat != 0")
        self.sat_ids = [int(row[0]) for row in self.cur.fetchall()]
        self.sat_ids.sort()
        self.num_sats = len(self.sat_ids)

    def __del__(self):
        self.conn.close()

    """
    Populate samples_array and ids_array with data from SQLite database

    Args:
        batch_size: number of samples to load at once (lower to save memory)
    """
    def __generate_arrays(self, batch_size=1000):
        if self.samples_array is not None and self.ids_array is not None:
            return

        print("Generating arrays...")

        self.cur.execute("SELECT iq_samples.samples, ira_messages.ra_sat, ira_messages.ra_cell FROM ira_messages JOIN iq_samples ON ira_messages.id = iq_samples.id WHERE ira_messages.ra_sat != 0")
        num_samples = self.num_samples

        samples_list = []
        samples_arrays = []
        ids_list = []
        cells_list = []

        count = 0
        elem = self.cur.fetchone()
        with tqdm(total=num_samples) as pbar:
            while elem is not None:
                count += 1

                samples_decoded = np.frombuffer(base64.b64decode(elem[0]), dtype=np.complex64)[:self.waveform_len].view(np.float32).reshape((self.waveform_len,2,))
                samples_list.append(samples_decoded)

                ids_list.append(elem[1])
                cells_list.append(elem[2])

                if count % batch_size == 0:
                    samples_arrays.append(np.stack(samples_list, axis=0))
                    samples_list = []

                elem = self.cur.fetchone()

                pbar.update(1)

        if len(samples_list) > 0:
            samples_arrays.append(np.stack(samples_list, axis=0))
            samples_list = []

        self.samples_array = np.concatenate(samples_arrays, axis=0)
        self.ids_array = np.array(ids_list)
        self.cells_array = np.array(cells_list)

        del samples_list, samples_arrays, ids_list

    """
    Populate samples_array and ids_array with data from SQLite database.
    Also populates arrays with magnitude, noise, level, and confidence values.

    Args:
        batch_size: number of samples to load at once (lower to save memory)
    """
    def __generate_arrays_noise(self, batch_size=1000):
        if self.samples_array is not None and self.ids_array is not None:
            return

        print("Generating arrays...")

        # 3 way join iq_samples, ira_messages, bytes on id
        self.cur.execute("SELECT iq_samples.samples, ira_messages.ra_sat, ira_messages.ra_cell, bytes.magnitude, bytes.noise, bytes.level, bytes.confidence FROM ira_messages JOIN iq_samples ON ira_messages.id = iq_samples.id JOIN bytes ON ira_messages.id = bytes.id WHERE ira_messages.ra_sat != 0")
        num_samples = self.num_samples

        samples_list = []
        samples_arrays = []
        ids_list = []
        cells_list = []
        magnitudes_list = []
        noises_list = []
        levels_list = []
        confidences_list = []

        count = 0
        elem = self.cur.fetchone()
        with tqdm(total=num_samples) as pbar:
            while elem is not None:
                count += 1

                samples_decoded = np.frombuffer(base64.b64decode(elem[0]), dtype=np.complex64)[:self.waveform_len].view(np.float32).reshape((self.waveform_len,2,))
                samples_list.append(samples_decoded)

                ids_list.append(elem[1])
                cells_list.append(elem[2])
                magnitudes_list.append(elem[3])
                noises_list.append(elem[4])
                levels_list.append(elem[5])
                confidences_list.append(elem[6])

                if count % batch_size == 0:
                    samples_arrays.append(np.stack(samples_list, axis=0))
                    samples_list = []

                elem = self.cur.fetchone()

                pbar.update(1)

        if len(samples_list) > 0:
            samples_arrays.append(np.stack(samples_list, axis=0))
            samples_list = []

        self.samples_array = np.concatenate(samples_arrays, axis=0)
        self.ids_array = np.array(ids_list)
        self.cells_array = np.array(cells_list)
        self.magnitudes_array = np.array(magnitudes_list)
        self.noises_array = np.array(noises_list)
        self.levels_array = np.array(levels_list)
        self.confidences_array = np.array(confidences_list)

        del samples_list, samples_arrays, ids_list, magnitudes_list, noises_list, levels_list, confidences_list

    """
    Populate samples_array and ids_array, and cells_array with data from SQLite database
    Loads all samples, including non-IRA messages; these will have ID and cell ID set to 0

    Args:
        batch_size: number of samples to load at once (lower to save memory)
    """
    def __generate_arrays_new(self, batch_size=1000, limit=None):
        if self.samples_array is not None and self.ids_array is not None and self.cells_array is not None:
            return

        print("Generating arrays...")

        self.cur.execute("SELECT iq_samples.id, iq_samples.samples, ira_messages.ra_sat, ira_messages.ra_cell FROM iq_samples LEFT JOIN ira_messages ON ira_messages.id = iq_samples.id")
        #num_samples = self.num_samples
        num_samples = 1

        samples_list = []
        samples_arrays = []
        ids_list = []
        cells_list = []

        count = 0
        elem = self.cur.fetchone()
        with tqdm(total=num_samples) as pbar:
            while elem is not None and (limit is None or count < limit):
                count += 1

                samples_decoded = np.frombuffer(base64.b64decode(elem[1]), dtype=np.complex64)[:self.waveform_len].view(np.float32).reshape((self.waveform_len,2,))
                samples_list.append(samples_decoded)

                ids_list.append(elem[2] if elem[2] is not None else 0)
                cells_list.append(elem[3] if elem[3] is not None else 0)

                if count % batch_size == 0:
                    samples_arrays.append(np.stack(samples_list, axis=0))
                    samples_list = []

                elem = self.cur.fetchone()

                pbar.update(1)

        if len(samples_list) > 0:
            samples_arrays.append(np.stack(samples_list, axis=0))
            samples_list = []

        self.samples_array = np.concatenate(samples_arrays, axis=0)
        self.ids_array = np.array(ids_list)
        self.cells_array = np.array(cells_list)

        del samples_list, samples_arrays, ids_list

    def __decode_samples(self, samples, num_samples):
        """
        Decode samples from base64 to numpy array

        Args:
            samples: base64 encoded samples
            num_samples: number of samples to decode

        Returns:
            decoded samples
        """
        b = samples.numpy()
        samples_decoded = np.frombuffer(base64.b64decode(b), dtype=np.complex64)[:num_samples].view(np.float32).reshape((num_samples,2,))
        return samples_decoded

    def __make_symmetric(self, samples, ra_sat):
        """
        Make samples symmetric (i.e. mapping input to itself)
        """
        samples.set_shape((None, self.waveform_len, 2))

        return (samples, samples)

    def to_generator(self):
        query = "SELECT ira_messages.ra_sat, iq_samples.samples FROM ira_messages JOIN iq_samples ON ira_messages.id = iq_samples.id WHERE ira_messages.ra_sat != 0"

        for row in self.cur.execute(query):
            ra_sat = int(row[0])
            ra_sat = self.sat_ids.index(ra_sat)
            samples = np.frombuffer(base64.b64decode(row[1].encode('UTF-8')), dtype=np.complex64)[:self.waveform_len].view(np.float32).reshape((1,self.waveform_len,2,))
            # Convert to float32

            yield samples, ra_sat

    def to_dataset(self, shuffle=10000, reshuffle_each_iteration=True):
        """
        Create dataset from SQLite database for tensorflow

        Args:
            shuffle: size of shuffle buffer

        Returns:
            dataset: tf.data.experimental.SqlDataset
        """
        query = "SELECT iq_samples.samples, ira_messages.ra_sat FROM ira_messages JOIN iq_samples ON ira_messages.id = iq_samples.id WHERE ira_messages.ra_sat != 0"
        ds_sqlite = tf.data.experimental.SqlDataset(
            'sqlite',
            self.data_file,
            query,
            (tf.string, tf.int32)
        )

        # Index lookup ra_sat according to sat_ids, decode samples and convert to float32
        ds_sqlite = ds_sqlite.map(lambda samples, ra_sat: (
            tf.py_function(func=self.__decode_samples, inp=[samples, self.waveform_len], Tout=[tf.float32]),
            tf.one_hot(tf.where(tf.equal(self.sat_ids, ra_sat)), self.num_sats, axis=-1)
        ))

        # Flatten ra_sat to (None, num_sats)
        ds_sqlite = ds_sqlite.map(lambda samples, ra_sat: (samples, tf.reshape(ra_sat, [-1, self.num_sats])))

        ds_sqlite = ds_sqlite.shuffle(shuffle, reshuffle_each_iteration=reshuffle_each_iteration)

        return ds_sqlite

    def to_symmetric_dataset(self, shuffle=10000, reshuffle_each_iteration=True):
        """
        Create dataset from SQLite database for reconstructing the same input

        Args:
            shuffle: size of shuffle buffer

        Returns:
            dataset: tf.data.experimental.SqlDataset
        """
        ds_sqlite = self.to_dataset(shuffle, reshuffle_each_iteration)

        ds_sqlite = ds_sqlite.map(self.__make_symmetric)

        return ds_sqlite

    """
    Create a batch of pairwise samples, where half are the same class left-and-right and half are different.
    """
    def __select_half_half(self, batch_size=10, with_raw_vals=False):
        self.__generate_arrays()

        ids_reshape = self.ids_array.reshape((-1,1))

        start = np.random.randint(0, len(self.samples_array))
        end = (start + batch_size)
        batchi = np.arange(start, end)

        in_l = np.take(self.samples_array, batchi, axis=0, mode="wrap")
        out_l = np.take(ids_reshape, batchi, axis=0, mode="wrap")

        in_r = np.empty_like(in_l)
        out_r = np.empty_like(out_l)

        #first half of right-hand side is matching (ideally not same entry, but ignore here as there are some singular entries)
        for i in range(batch_size // 2):
            same = np.argwhere(ids_reshape.reshape(-1) == out_l[i]).flatten()
            righti = np.random.randint(0, len(same))
            in_r[i,:] = self.samples_array[same[righti]]
            out_r[i,:] = ids_reshape[same[righti]]

        #second half is not matching
        for i in range(batch_size // 2, batch_size, 1):
            diff = np.argwhere(ids_reshape.reshape(-1) != out_l[i]).flatten()
            righti = np.random.randint(0, len(diff))
            in_r[i,:] = self.samples_array[diff[righti]]
            out_r[i,:] = ids_reshape[diff[righti]]

        if with_raw_vals:
            return in_l, in_r, (out_l == out_r).astype(int), out_l, out_r
        else:
            return in_l, in_r, (out_l == out_r).astype(int)

    def halfhalf_generator(self, batch_size=10, with_raw_vals=False):
        while True:
            if not with_raw_vals:
                (in_l, in_r, out) = self.__select_half_half(batch_size)
                yield ([in_l, in_r], out)
            else:
                (in_l, in_r, out, out_l, out_r) = self.__select_half_half(batch_size, with_raw_vals=True)
                yield ([in_l, in_r], out, out_l, out_r)

    """
    Save the samples, satellite ids, and cell ids to files.

    Args:
        file_samples: path to samples file
        file_ids: path to satellite ids file
        file_cells: path to cell ids file
        ira_only: if True, only save samples for IRA messages
        message_only: if True, only save samples which have properly decoded messages
    """
    def save_arrays(self, file_samples, file_ids, file_cells=None, ira_only=True, message_only=False):
        if ira_only:
            self.__generate_arrays()
        else:
            self.__generate_arrays_new()

        if message_only:
            self.cur.execute("SELECT iq_samples.id, bytes.msg_type FROM iq_samples LEFT JOIN bytes ON iq_samples.id = bytes.id")
            messages = self.cur.fetchall()
            messages_array = np.array([ m[1] for m in messages ])

            # Only save samples with messages
            samples_array = self.samples_array[messages_array != '']
            ids_array = self.ids_array[messages_array != '']
            cells_array = self.cells_array[messages_array != '']
        else:
            samples_array = self.samples_array
            ids_array = self.ids_array
            cells_array = self.cells_array

        np.save(file_samples, samples_array)
        np.save(file_ids, ids_array)
        if file_cells is not None:
            np.save(file_cells, cells_array)

    """
    Save the samples, satellite ids, and cell ids to files.
    Also save the magnitude, noise, level, and confidence.

    Args:
        file_samples: path to samples file
        file_ids: path to satellite ids file
        file_cells: path to cell ids file
        file_magnitudes: path to magnitude file
        file_noises: path to noise file
        file_levels: path to level file
        file_confidences: path to confidence file
    """
    def save_arrays_noise(self, file_samples, file_ids, file_cells, file_magnitudes, file_noises, file_levels, file_confidences):
        self.__generate_arrays_noise()

        np.save(file_samples, self.samples_array)
        np.save(file_ids, self.ids_array)
        np.save(file_cells, self.cells_array)
        np.save(file_magnitudes, self.magnitudes_array)
        np.save(file_noises, self.noises_array)
        np.save(file_levels, self.levels_array)
        np.save(file_confidences, self.confidences_array)

    """
    Return the data in array form.
    Note: don't use for tensorflow, use ArrayDataset instead.

    Args:
        cell: include the cell id
    """
    def to_arrays(self, cell=False):
        self.__generate_arrays()

        if cell:
            return self.samples_array, self.ids_array, self.cells_array
        else:
            return self.samples_array, self.ids_array


class ArrayDataset(Dataset):
    def __init__(self, samples_array, ids_array, cells_array=None):
        self.samples_array = samples_array
        self.ids_array = ids_array
        self.cells_array = cells_array

        self.has_cells = cells_array is not None

        if self.has_cells:
            self.id_cell = (self.ids_array * (np.max(self.cells_array)+1)) + self.cells_array
            #id_cell = np.concatenate((self.ids_array.reshape(-1,1), self.cells_array.reshape(-1,1)), axis=1)
            # Assign a unique number to each id,cell pair
            #id_cell, id_cell_idx = np.unique(id_cell, return_inverse=True, axis=0)
            #self.id_cell = id_cell_idx
        else:
            self.id_cell = None

        self.waveform_len = self.samples_array.shape[1]
        self.sat_ids = np.unique(self.ids_array)
        self.sat_ids.sort()
        self.num_sats = len(self.sat_ids)
        self.num_samples = len(self.samples_array)

    @classmethod
    def from_files(cls, file_samples, file_ids, file_cells=None):
        samples_array = np.load(file_samples)
        ids_array = np.load(file_ids)
        if file_cells is not None:
            cells_array = np.load(file_cells)
        else:
            cells_array = None

        return cls(samples_array, ids_array, cells_array)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.has_cells:
            return self.samples_array[idx], self.ids_array[idx], self.cells_array[idx]
        else:
            return self.samples_array[idx], self.ids_array[idx]

    def balance(self):
        """
        Balance the dataset by removing samples from all classes with more than the minimum number of samples.

        Returns:
            ArrayDataset: balanced dataset
        """

        bins = np.bincount(self.ids_array)
        bins_with_index = [(i, bins[i]) for i in range(len(bins)) if bins[i] > 0]
        bins_with_index.sort(key=lambda x: x[1], reverse=True)
        (min_index, min_size) = min(bins_with_index, key=lambda x: x[1])

        # Remove samples to bring the size of each class down to min_size
        samples_arrays = []
        ids_arrays = []
        if self.has_cells:
            cells_arrays = []
        for (index, size) in bins_with_index:
            samples_array_i = self.samples_array[self.ids_array == index]
            ids_array_i = self.ids_array[self.ids_array == index]
            if self.has_cells:
                cells_array_i = self.cells_array[self.ids_array == index]
            if size > min_size:
                samples_array_i = samples_array_i[:min_size]
                ids_array_i = ids_array_i[:min_size]
                if self.has_cells:
                    cells_array_i = cells_array_i[:min_size]
            samples_arrays.append(samples_array_i)
            ids_arrays.append(ids_array_i)
            if self.has_cells:
                cells_arrays.append(cells_array_i)

        samples_array = np.concatenate(samples_arrays)
        ids_array = np.concatenate(ids_arrays)
        if self.has_cells:
            cells_array = np.concatenate(cells_arrays)
        else:
            cells_array = None

        return ArrayDataset(samples_array, ids_array, cells_array)

    def filter_zeros(self):
        """
        Filter out samples with satellite or cell ID 0.

        Returns:
            ArrayDataset: filtered dataset
        """
        samples_array = self.samples_array[self.ids_array != 0]
        ids_array = self.ids_array[self.ids_array != 0]
        if self.has_cells:
            cells_array = self.cells_array[self.ids_array != 0]

            samples_array = samples_array[cells_array != 0]
            ids_array = ids_array[cells_array != 0]
            cells_array = cells_array[cells_array != 0]

        return ArrayDataset(samples_array, ids_array, cells_array)

    def filter_most_common(self, count=10, remove=False):
        """
        Filter to contain only the most common satellite-cell pairs.

        Args:
            count: number of satellite-cell ID pairs to keep
            remove: if True, remove the most common satellite-cell pairs instead of keeping them

        Returns:
            ArrayDataset: filtered dataset
        """
        if not self.has_cells:
            raise ValueError("Dataset does not have cell IDs")

        id_cell_unique, id_cell_counts = np.unique(self.id_cell, return_counts=True)
        argsort_most_common = np.argsort(id_cell_counts)[::-1][:count]
        id_cells_most_common = id_cell_unique[argsort_most_common]

        if not remove:
            samples_array = self.samples_array[np.isin(self.id_cell, id_cells_most_common)]
            ids_array = self.ids_array[np.isin(self.id_cell, id_cells_most_common)]
            cells_array = self.cells_array[np.isin(self.id_cell, id_cells_most_common)]
        else:
            samples_array = self.samples_array[~np.isin(self.id_cell, id_cells_most_common)]
            ids_array = self.ids_array[~np.isin(self.id_cell, id_cells_most_common)]
            cells_array = self.cells_array[~np.isin(self.id_cell, id_cells_most_common)]

        return ArrayDataset(samples_array, ids_array, cells_array)

    def filter_non_unique(self):
        """
        Filter out satellite-cell ID pairs that occur only once.

        Returns:
            ArrayDataset: filtered dataset
        """
        if not self.has_cells:
            raise ValueError("Dataset does not have cell IDs")

        id_cell_unique, id_cell_counts = np.unique(self.id_cell, return_counts=True)
        id_cells_non_unique = id_cell_unique[id_cell_counts > 1]

        samples_array = self.samples_array[np.isin(self.id_cell, id_cells_non_unique)]
        ids_array = self.ids_array[np.isin(self.id_cell, id_cells_non_unique)]
        cells_array = self.cells_array[np.isin(self.id_cell, id_cells_non_unique)]

        return ArrayDataset(samples_array, ids_array, cells_array)

    def filter_identical(self, distance=5.0):
        """
        Filter out samples that are identical to other samples.

        Returns:
            ArrayDataset: filtered dataset
        """

        samples_keep = np.ones(self.num_samples, dtype=bool)
        for i in tqdm(range(len(self.samples_array))):
            sample = self.samples_array[i]
            id_cell = self.id_cell[i]

            for j in np.where(self.id_cell == id_cell)[0]:
                if j <= i:
                    continue
                if np.linalg.norm(sample - self.samples_array[j]) < distance:
                    samples_keep[j] = False

        samples_array = self.samples_array[samples_keep]
        ids_array = self.ids_array[samples_keep]
        cells_array = self.cells_array[samples_keep]

        return ArrayDataset(samples_array, ids_array, cells_array)

    def shuffle(self, seed=None):
        """
        Shuffle the dataset

        Args:
            seed: seed for random number generator
        """
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(self.num_samples)
        self.samples_array = self.samples_array[perm]
        self.ids_array = self.ids_array[perm]
        if self.has_cells:
            self.cells_array = self.cells_array[perm]
            self.id_cell = self.id_cell[perm]

    def __decimate_sample(self, sample, decimation, simple=False):
        """
        Decimate a single sample by a factor of decimation.

        Args:
            sample: sample to decimate
            decimation: decimation factor
            simple: if True, use simple decimation, otherwise use scipy decimate with a FIR filter
        """

        if simple:
            return sample[::decimation]
        else:
            sample_complex = sample[:,0] + (1j * sample[:,1])
            sample_decimated = scipy.signal.decimate(sample_complex, decimation, ftype='fir')
            sample_decimated_iq = np.stack((sample_decimated.real, sample_decimated.imag), axis=1)
            return sample_decimated_iq

    def decimate(self, decimation=2, simple=False, progress=False):
        """
        Decimate each sample in the dataset by a factor of decimation.

        Args:
            decimation: decimation factor
            simple: if True, use simple decimation, otherwise use scipy decimate with a FIR filter
            progress: if True, show progress bar

        Returns:
            ArrayDataset: decimated dataset
        """

        if progress:
            samples_decimated = np.array([self.__decimate_sample(sample, decimation=decimation, simple=simple) for sample in tqdm(self.samples_array)])
        else:
            samples_decimated = np.array([self.__decimate_sample(sample, decimation=decimation, simple=simple) for sample in self.samples_array])

        return ArrayDataset(samples_decimated, self.ids_array, self.cells_array)

    def split_train_val_test(self, train_size=0.7, val_size=0.15):
        """
        Split dataset into train, validation and test sets

        Args:
            train_size: size of training set
            val_size: size of validation set
        """
        num_train = int(train_size * self.num_samples)
        num_val = int(val_size * self.num_samples)

        if self.has_cells:
            train_ds = ArrayDataset(self.samples_array[:num_train], self.ids_array[:num_train], self.cells_array[:num_train])
            val_ds = ArrayDataset(self.samples_array[num_train:num_train+num_val], self.ids_array[num_train:num_train+num_val], self.cells_array[num_train:num_train+num_val])
            test_ds = ArrayDataset(self.samples_array[num_train+num_val:], self.ids_array[num_train+num_val:], self.cells_array[num_train+num_val:])
        else:
            train_ds = ArrayDataset(self.samples_array[:num_train], self.ids_array[:num_train])
            val_ds = ArrayDataset(self.samples_array[num_train:num_train+num_val], self.ids_array[num_train:num_train+num_val])
            test_ds = ArrayDataset(self.samples_array[num_train+num_val:], self.ids_array[num_train+num_val:])

        train_ds.num_sats = self.num_sats
        val_ds.num_sats = self.num_sats
        test_ds.num_sats = self.num_sats

        return train_ds, val_ds, test_ds

    def to_dataset(self, batch_size, shuffle=None, reshuffle_each_iteration=True, use_cell_ids=False):
        """
        Create dataset mapping samples to ids

        Args:
            batch_size: batch size to use when training
            shuffle: size of shuffle buffer
            reshuffle_each_iteration: whether to reshuffle each iteration
        """
        if not use_cell_ids:
            ds_samples = tf.data.Dataset.from_tensor_slices(self.samples_array).batch(batch_size)
            ds_ids = tf.data.Dataset.from_tensor_slices(self.ids_array).batch(batch_size)
            ds_array = tf.data.Dataset.zip((ds_samples, ds_ids))
        else:
            ds_samples = tf.data.Dataset.from_tensor_slices(self.samples_array).batch(batch_size)
            ds_id_cell = tf.data.Dataset.from_tensor_slices(self.id_cell).batch(batch_size)
            ds_array = tf.data.Dataset.zip((ds_samples, ds_id_cell))

        if shuffle is not None:
            ds_array = ds_array.shuffle(shuffle, reshuffle_each_iteration)

        return ds_array

    def to_onehot_dataset(self, batch_size, shuffle=None, reshuffle_each_iteration=True):
        """
        Create dataset mapping samples to one-hot ids, using a one-hot encoding
        NOTE: currently does not support cell IDs

        Args:
            batch_size: batch size to use when training
            shuffle: size of shuffle buffer
            reshuffle_each_iteration: whether to reshuffle each iteration
        """
        ds_samples = tf.data.Dataset.from_tensor_slices(self.samples_array).batch(batch_size)

        ids_copy = self.ids_array.copy()
        for i in range(len(self.sat_ids)):
            ids_copy[self.ids_array == self.sat_ids[i]] = i
        ids_copy_onehot = np.zeros((len(ids_copy), self.num_sats))
        ids_copy_onehot[np.arange(len(ids_copy)), ids_copy] = 1

        ds_ids = tf.data.Dataset.from_tensor_slices(ids_copy_onehot).batch(batch_size)
        #ds_ids = tf.data.Dataset.from_tensor_slices(self.ids_array).batch(batch_size)
        ds_array = tf.data.Dataset.zip((ds_samples, ds_ids))

        #ds_array = ds_array.map(lambda samples, id: (
        #    samples,
        #    tf.squeeze(tf.one_hot(tf.where(tf.equal(self.sat_ids, id)), self.num_sats, axis=-1), axis=1)
        #))

        if shuffle is not None:
            ds_array = ds_array.shuffle(shuffle, reshuffle_each_iteration)

        return ds_array

    def to_symmetric_dataset(self, batch_size, shuffle=None, reshuffle_each_iteration=True):
        """
        Create dataset for reconstructing the same input

        Args:
            batch_size: batch size to use when training
            shuffle: size of shuffle buffer
            reshuffle_each_iteration: whether to reshuffle each iteration
        """

        #ds_samples_a = tf.data.Dataset.from_tensor_slices(self.samples_array).batch(batch_size)
        #ds_samples_b = tf.data.Dataset.from_tensor_slices(self.samples_array).batch(batch_size)
        #ds_array = tf.data.Dataset.zip((ds_samples_a, ds_samples_b))
        ds_samples = tf.data.Dataset.from_tensor_slices(self.samples_array)
        ds_array = tf.data.Dataset.zip((ds_samples, ds_samples)).batch(batch_size)

        if shuffle is not None:
            ds_array = ds_array.shuffle(shuffle, reshuffle_each_iteration)

        return ds_array

    def __select_half_half(self, use_cell_ids=False, batch_size=10, same_id=True, with_raw_vals=False, exclude_same_sample=False):
        """
        Create a batch of pairwise samples, where half are the same class left-and-right and half are different.
        """

        if use_cell_ids:
            ids_reshape = self.id_cell.reshape((-1, 1))
        else:
            ids_reshape = self.ids_array.reshape((-1, 1))

        start = np.random.randint(0, len(self.samples_array))
        end = (start + batch_size)
        batchi = np.arange(start, end)

        in_l = np.take(self.samples_array, batchi, axis=0, mode="wrap")
        out_l = np.take(ids_reshape, batchi, axis=0, mode="wrap")

        in_r = np.empty_like(in_l)
        out_r = np.empty_like(out_l)

        if same_id:
            #for i in range(batch_size // 2):
            for i in range(batch_size):
                same = np.argwhere(ids_reshape.reshape(-1) == out_l[i]).flatten()
                if exclude_same_sample:
                    same = same[same != batchi[i]]
                righti = np.random.randint(0, len(same))
                in_r[i,:] = self.samples_array[same[righti]]
                out_r[i,:] = ids_reshape[same[righti]]
        else:
            #for i in range(batch_size // 2, batch_size, 1):
            for i in range(batch_size):
                diff = np.argwhere(ids_reshape.reshape(-1) != out_l[i]).flatten()
                righti = np.random.randint(0, len(diff))
                in_r[i,:] = self.samples_array[diff[righti]]
                out_r[i,:] = ids_reshape[diff[righti]]

        if with_raw_vals:
            return in_l, in_r, (out_l != out_r).astype(np.float32), out_l, out_r
        else:
            return in_l, in_r, (out_l != out_r).astype(np.float32)

    def halfhalf_generator(self, use_cell_ids=False, batch_size=10, with_raw_vals=False, exclude_same_sample=False):
        # Alternate between batches with the same and different IDs
        same_id = True
        while True:
            if not with_raw_vals:
                (in_l, in_r, out) = self.__select_half_half(use_cell_ids, batch_size, same_id, exclude_same_sample=exclude_same_sample)
                yield ([in_l, in_r], out)
            else:
                (in_l, in_r, out, out_l, out_r) = self.__select_half_half(use_cell_ids, batch_size, same_id, with_raw_vals=True, exclude_same_sample=exclude_same_sample)
                yield ([in_l, in_r], out, out_l, out_r)
            same_id = not same_id

    def __select_triplet(self, use_cell_ids=False, batch_size=10, exclude_same_sample=False):
        """
        Create a batch of samples suitable for triplet loss, where a third of the samples are anchors, a third are
        positive examples, and a third are negative examples.
        """

        if use_cell_ids:
            ids_reshape = self.id_cell.reshape((-1, 1))
        else:
            ids_reshape = self.ids_array.reshape((-1, 1))

        batch_size_anchor = batch_size // 3
        batch_size_pos = batch_size // 3
        batch_size_neg = batch_size - batch_size_anchor - batch_size_pos

        start = np.random.randint(0, len(self.samples_array))
        end = (start + batch_size_anchor)
        batchi = np.arange(start, end)

        in_anchor = np.take(self.samples_array, batchi, axis=0, mode="wrap")
        out_anchor = np.take(ids_reshape, batchi, axis=0, mode="wrap")

        in_pos = np.empty((batch_size_pos,) + in_anchor.shape[1:], dtype=in_anchor.dtype)
        in_neg = np.empty((batch_size_neg,) + in_anchor.shape[1:], dtype=in_anchor.dtype)

        out_pos = np.empty((batch_size_pos,) + out_anchor.shape[1:], dtype=out_anchor.dtype)
        out_neg = np.empty((batch_size_neg,) + out_anchor.shape[1:], dtype=out_anchor.dtype)

        for i in range(batch_size_pos):
            same = np.argwhere(ids_reshape.reshape(-1) == out_anchor[i % len(out_anchor)]).flatten()
            if exclude_same_sample:
                same = same[same != batchi[i]]
            righti = np.random.randint(0, len(same))
            in_pos[i,:] = self.samples_array[same[righti]]
            out_pos[i,:] = ids_reshape[same[righti]]

        for i in range(batch_size_neg):
            diff = np.argwhere(ids_reshape.reshape(-1) != out_anchor[i % len(out_anchor)]).flatten()
            righti = np.random.randint(0, len(diff))
            in_neg[i,:] = self.samples_array[diff[righti]]
            out_neg[i,:] = ids_reshape[diff[righti]]

        return np.concatenate([in_anchor, in_pos, in_neg], axis=0), np.concatenate([out_anchor, out_pos, out_neg], axis=0)

    # Generate batches of samples suitable for triplet loss.
    # Each batch contains a set of anchor samples, a set of positive samples, and a set of negative samples.
    def triplet_generator(self, use_cell_ids=False, batch_size=10, exclude_same_sample=False):
        while True:
            yield self.__select_triplet(use_cell_ids, batch_size, exclude_same_sample=exclude_same_sample)


class NoiseArrayDataset(Dataset):
    def __init__(self, samples_array, ids_array, cells_array, magnitudes_array, noises_array, levels_array, confidences_array):
        self.samples_array = samples_array
        self.ids_array = ids_array
        self.cells_array = cells_array
        self.magnitudes_array = magnitudes_array
        self.noises_array = noises_array
        self.levels_array = levels_array
        self.confidences_array = confidences_array

        self.has_cells = True

        self.id_cell = (self.ids_array * (np.max(self.cells_array)+1)) + self.cells_array

        self.waveform_len = self.samples_array.shape[1]
        self.sat_ids = np.unique(self.ids_array)
        self.sat_ids.sort()
        self.num_sats = len(self.sat_ids)
        self.num_samples = len(self.samples_array)

    @classmethod
    def from_files(cls, file_samples, file_ids, file_cells, file_magnitudes, file_noises, file_levels, file_confidences):
        samples_array = np.load(file_samples)
        ids_array = np.load(file_ids)
        cells_array = np.load(file_cells)
        magnitudes_array = np.load(file_magnitudes)
        noises_array = np.load(file_noises)
        levels_array = np.load(file_levels)
        confidences_array = np.load(file_confidences)

        return cls(samples_array, ids_array, cells_array, magnitudes_array, noises_array, levels_array, confidences_array)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples_array[idx], self.ids_array[idx], self.cells_array[idx], self.magnitudes_array[idx], self.noises_array[idx], self.levels_array[idx], self.confidences_array[idx]

    def filter_zeros(self):
        """
        Filter out samples with satellite or cell ID 0.

        Returns:
            ArrayDataset: filtered dataset
        """
        samples_array = self.samples_array[self.ids_array != 0]
        ids_array = self.ids_array[self.ids_array != 0]
        cells_array = self.cells_array[self.ids_array != 0]
        magnitudes_array = self.magnitudes_array[self.ids_array != 0]
        noises_array = self.noises_array[self.ids_array != 0]
        levels_array = self.levels_array[self.ids_array != 0]
        confidences_array = self.confidences_array[self.ids_array != 0]

        samples_array_2 = samples_array[cells_array != 0]
        ids_array_2 = ids_array[cells_array != 0]
        cells_array_2 = cells_array[cells_array != 0]
        magnitudes_array_2 = magnitudes_array[cells_array != 0]
        noises_array_2 = noises_array[cells_array != 0]
        levels_array_2 = levels_array[cells_array != 0]
        confidences_array_2 = confidences_array[cells_array != 0]

        return NoiseArrayDataset(samples_array_2, ids_array_2, cells_array_2, magnitudes_array_2, noises_array_2, levels_array_2, confidences_array_2)

    def filter_identical(self, distance=5.0):
        """
        Filter out samples that are identical to other samples.

        Returns:
            ArrayDataset: filtered dataset
        """

        samples_keep = np.ones(self.num_samples, dtype=bool)
        for i in tqdm(range(len(self.samples_array))):
            sample = self.samples_array[i]
            id_cell = self.id_cell[i]

            for j in np.where(self.id_cell == id_cell)[0]:
                if j <= i:
                    continue
                if np.linalg.norm(sample - self.samples_array[j]) < distance:
                    samples_keep[j] = False

        samples_array = self.samples_array[samples_keep]
        ids_array = self.ids_array[samples_keep]
        cells_array = self.cells_array[samples_keep]
        magnitudes_array = self.magnitudes_array[samples_keep]
        noises_array = self.noises_array[samples_keep]
        levels_array = self.levels_array[samples_keep]
        confidences_array = self.confidences_array[samples_keep]

        return NoiseArrayDataset(samples_array, ids_array, cells_array, magnitudes_array, noises_array, levels_array, confidences_array)


class SymmetricArrayDataset():
    def __init__(self, samples_array):
        self.samples_array = samples_array

        #self.waveform_len = self.samples_array.shape[1]
        #self.num_samples = len(self.samples_array)

    @classmethod
    def from_files(cls, file_samples):
        samples_array = np.load(file_samples)

        return cls(samples_array)

    def __len__(self):
        return self.num_samples

    def get_dataset(self, batch_size):
        ds_samples = tf.data.Dataset.from_tensor_slices(self.samples_array)
        ds_array = tf.data.Dataset.zip((ds_samples, ds_samples))
        return ds_array.batch(batch_size)


class TFRecordLoader():
    @staticmethod
    def _parse_example(example):
        """
        Parse a single example from a TFRecord file.
        """
        features = {
            "sample": tf.io.FixedLenFeature([11000, 2], tf.float32),
            "id": tf.io.FixedLenFeature([], tf.int64),
            "cell": tf.io.FixedLenFeature([], tf.int64),
            "id_cell": tf.io.FixedLenFeature([], tf.int64),
        }
        return tf.io.parse_single_example(example, features)

    @staticmethod
    def _features_to_input_output(features):
        """
        Convert the features dictionary to the input and output tensors.
        """
        return features['sample'], features['id_cell']

    @staticmethod
    def _features_to_input_output_id_only(features):
        """
        Convert the features dictionary to the input and output tensors, using only the ID.
        """
        return features['sample'], features['id']

    @classmethod
    def _process_dataset(self, ds, shuffle_buffer_sample, seed, use_id_only, shuffle=True):
        """
        Process a TFRecord dataset by parsing the examples, shuffling, and converting to input/output tensors.
        """
        ds = ds.map(self._parse_example)
        #ds = ds.repeat()
        if use_id_only:
            ds = ds.map(self._features_to_input_output_id_only)
        else:
            ds = ds.map(self._features_to_input_output)
        if shuffle:
            ds = ds.shuffle(shuffle_buffer_sample, seed=seed, reshuffle_each_iteration=True)

        return ds

    @classmethod
    def from_file_no_shuffle(cls, file_in, use_id_only=False):
        """
        Load a single TFRecord file without shuffling.

        Args:
            file_in (str): Path to the TFRecord file.
            use_id_only (bool): If True, use only the satellite ID as the output tensor, rather than the combined satellite and cell ID.
        """
        ds = tf.data.TFRecordDataset(file_in)
        return cls._process_dataset(ds, 0, 0, use_id_only, shuffle=False)

    @classmethod
    def from_file(cls, file_in, shuffle_buffer_sample, seed, use_id_only=False):
        """
        Load a single TFRecord file.

        Args:
            file_in (str): Path to the TFRecord file.
            shuffle_buffer_sample (int): Size of the shuffle buffer for the samples.
            seed (int): Seed for the random number generator.
            use_id_only (bool): If True, use only the satellite ID as the output tensor, rather than the combined satellite and cell ID.
        """
        ds = tf.data.TFRecordDataset(file_in)
        return cls._process_dataset(ds, shuffle_buffer_sample, seed, use_id_only)

    @classmethod
    def from_files(cls, files_in, shuffle_buffer_file, cycle_length, shuffle_buffer_sample, seed, use_id_only=False):
        """
        Load a list of TFRecord files.

        Args:
            files_in (list): List of paths to the TFRecord files.
            shuffle_buffer_file (int): Size of the shuffle buffer for the files.
            cycle_length (int): Number of files to read in parallel.
            shuffle_buffer_sample (int): Size of the shuffle buffer for the samples.
            seed (int): Seed for the random number generator.
            use_id_only (bool): If True, use only the satellite ID as the output tensor, rather than the combined satellite and cell ID.
        """
        ds = tf.data.Dataset.from_tensor_slices(files_in).shuffle(shuffle_buffer_file, seed=seed).interleave(
            lambda x: tf.data.TFRecordDataset(x),
            cycle_length=cycle_length,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        return cls._process_dataset(ds, shuffle_buffer_sample, seed, use_id_only)

    @staticmethod
    def window_batch(ds, window_size, batch_size):
        """
        Group samples by their ID then further group them into batches.

        Args:
            ds (tf.data.Dataset): Dataset to group.
            window_size (int): Size of the window to group samples by.
            batch_size (int): Size of the batches to group samples into.
        """
        return ds.apply(tf.data.experimental.group_by_window(
            key_func=lambda x, y: y,
            reduce_func=lambda key, ds: ds.batch(window_size),
            window_size=window_size,
        )).unbatch().batch(batch_size)

    @staticmethod
    def get_test_batch(ds, num_batches):
        """
        Get a batch of samples and labels from a dataset.

        Args:
            ds (tf.data.Dataset): Dataset to get the batch from.
            num_batches (int): Number of batches to get.

        Returns:
            samples_test (np.array): Array of samples.
            labels_test (np.array): Array of labels.
        """
        samples_test = []
        labels_test = []
        for i, (x, y) in enumerate(ds.take(num_batches)):
            if i == num_batches:
                break
            samples_test.append(x.numpy())
            labels_test.append(y.numpy())

        samples_test = np.concatenate(samples_test)
        labels_test = np.concatenate(labels_test)

        return samples_test, labels_test