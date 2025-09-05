import os
import numpy as np
import datetime
import tensorflow as tf
import argparse

from tensorflow.keras.callbacks import EarlyStopping

from util.models import AETripletSplitConvModel, VAETripletSplitConvModel, TripletSplitConvModel
from util.model_utils import AUCCallback, LossHistoryCallback
from util.data import TFRecordLoaderFull

tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0],
    True
)

data_base = "/data"
save_dir = data_base + "/models"
data_dir = data_base + "/tfrecord"

start_date = "2023-06-12"
num_days = 0

num_epochs = 200

model_name = 'ae-triplet'

filter_ids = np.array([
    81, 119, 134, 142, 144, 155, 156, 196, 204, 207, 215,
    219, 220, 221, 255, 257, 258, 263, 452, 510, 552, 560,
    570, 572, 577, 578, 589, 662, 665, 758, 799, 821, 828,
    831, 833, 834, 845, 849, 854, 924, 1011, 1061, 1074, 1082,
    1134, 1146, 1147, 1149, 1150, 1152, 1166, 1170, 1219, 1325, 1389,
    1450, 1457, 1463, 1464, 1465, 1467, 1475, 1479, 1481, 1516, 1576,
    1583, 1584, 1587, 1588, 1591, 1606, 1607, 1643, 1831, 1861, 1863,
    1869, 1870, 2026, 2060, 2113, 2114, 2119, 2122, 2270, 2276, 2277,
    2278, 2282, 2283, 2288, 2301, 2302, 2303, 2306, 2307, 2309, 2311,
    2312, 2313, 2314, 2315, 2316, 2413, 2562, 2646, 2651, 2680, 2681,
    2683, 2686, 2688, 2689, 2694, 2704, 2775, 2780, 2781, 2783, 2784,
    2787, 2788, 2791, 2792, 2794, 2795, 2796, 2798, 2800, 2801, 2806,
    2807, 2808, 2810, 2812, 2814, 2816, 2817, 2819, 2820, 2906, 2907,
    2913, 2916, 2924, 2956, 3066, 3096, 3099, 3100, 3103, 3113, 3126,
    3184, 3185, 3190, 3192, 3193, 3505, 3574, 3593, 3594, 3595, 3602,
    3603, 3604, 3605, 3606, 3607, 3609, 3612, 3617, 3618, 3624, 3626,
    3629, 3631, 3635, 3636, 3637, 3638, 3704, 4091, 4128, 4130, 4131,
    4132, 4135, 4138, 4139, 4252, 4256, 4257, 4258, 4259, 4267, 4319,
    4321, 4324, 4326, 4327, 4361, 4362, 4363, 4367, 4368, 4369, 4370,
    4371, 4373, 4375, 4378, 4380, 4381, 4383, 4384, 4386, 4388, 4391,
    4395, 4435, 4510, 4513, 4515, 4517, 4572, 4576, 4578, 4579, 4580,
    4600, 4603, 4608, 4609, 4610, 4611, 4613, 4614, 4617, 4618, 4621,
    4623, 4625, 4627, 4630, 4631, 4633, 4636, 4638, 4639, 4640, 4642,
    4645, 4646, 4647, 4922, 4923, 4924, 4932, 4935, 4938, 4944, 4947,
    4948, 4950, 4958, 4959, 4961, 4985, 4986, 4988, 4991, 4992, 4993,
    4994, 4995, 4997, 5004, 5008, 5011, 5012, 5014, 5015, 5016, 5020,
    5022, 5024, 5047, 5122, 5137, 5138, 5140, 5141, 5143, 5146, 5171,
    5172, 5360, 5494, 5501, 5534, 5546, 5548, 5552, 5556, 5557, 5558,
    5560, 5565, 5566, 5568, 5570, 5571, 5572, 5573, 5576, 5577, 5582,
    5584, 5587, 5596, 5608, 5609, 5610, 5616, 5617, 5618, 5619, 5620,
    5625, 5633, 5637, 5639, 5641, 5642, 5643, 5647, 5650, 5653, 5671,
    5682, 5683, 5684, 5686, 5688, 5705, 5765, 5905, 5923, 5930, 5931,
    5934, 5936, 5937, 5938, 5948, 5953, 5961, 6036, 6062, 6063, 6077,
    6080, 6238, 6241, 6245, 6246, 6247, 6248, 6250, 6251, 6255, 6256,
    6257, 6258, 6264, 6265, 6267, 6270, 6272, 6273, 6274, 6278, 6280,
    6281, 6282, 6283, 6494, 6552, 6840, 6935, 6940, 7059, 7067, 7193,
    7246, 7247, 7249, 7253, 7254, 7263, 7264, 7265, 7267, 7268, 7271,
    7273, 7274, 7278, 7279, 7280, 7284, 7286, 7287, 7288, 7289, 7290,
    7291, 7292, 7547, 7981])

# Downsample the dataset by a specified factor, replacing all other samples with zeros
def downsample_data(x, y, factor):
    indices = tf.range(tf.shape(x)[0])
    condition = tf.equal(tf.math.floormod(indices, factor), 0)
    condition = tf.stack([condition, condition], axis=1)
    return tf.where(condition, x, tf.zeros_like(x)), y

def main(
        save_dir,
        data_dir,
        start_date,
        num_days,
        num_epochs,
        model_name,
        save_epochs,
        save_best_epoch,
        best_stop_after,
        filter_predicate=None,
        no_ae=False,
        vae=False,
        triplet_margin=1.0,
        downsample=1,
        save_loss=False,
    ):
    num_samples = 11000 # = 880*12.5
    #if downsample > 1:
    #    num_samples = (num_samples + downsample - 1) // downsample
    group_window_size = 4 # Number of samples with the same ID to group together
    batch_size = 32
    seed = 20220615

    validation_steps = 200

    # shuffle_buffer_file = 32
    cycle_length = 32 # Number of files to read in parallel
    shuffle_buffer_sample = 10000

    layers = [
        (64, 2),
        (64, 4),
        (32, 8),
        (32, 16),
        (32, 32),
        (32, 32),
    ]
    latent_dim = 512

    triplet_distance_metric = 'angular' # L2, squared-L2, angular
    normalization = 'L2' # L2, L1, None

    learning_rate = 1e-5 #1e-5

    # Get the data files for the given date range
    files_in = []
    if num_days > 0:
        for i in range(num_days):
            date = datetime.datetime.strptime(start_date, '%Y-%m-%d') + datetime.timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            files_in.extend([
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith('.tfrecord') and f.startswith(date_str)
            ])
    else:
        files_in.extend([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.tfrecord')
        ])
    files_in.sort()

    files_val = [ f for f in files_in if f.endswith('_1.tfrecord') ]
    files_test = [ f for f in files_in if f.endswith('_0.tfrecord') ]
    files_train = [ f for f in files_in if f not in files_val and f not in files_test ]

    print(f"Training files: {files_train}")
    ds_train = TFRecordLoaderFull.from_files(
        files_train,
        len(files_train),
        min(cycle_length, len(files_train)),
        seed=seed,
        shuffle_buffer_sample=shuffle_buffer_sample,
        filter_predicate=filter_predicate,
    )
    if downsample > 1:
        #ds_train = ds_train.map(lambda x, y: (x[::downsample], y))
        ds_train = ds_train.map(lambda x, y: downsample_data(x, y, downsample))
    ds_train = TFRecordLoaderFull.window_batch(ds_train, group_window_size, batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    print(f"Validation files: {files_val}")
    ds_val = TFRecordLoaderFull.from_files(
        files_val,
        len(files_val),
        min(cycle_length, len(files_val)),
        seed=seed,
        shuffle_buffer_sample=shuffle_buffer_sample,
        filter_predicate=filter_predicate,
    )
    if downsample > 1:
        #ds_val = ds_val.map(lambda x, y: (x[::downsample], y))
        ds_val = ds_val.map(lambda x, y: downsample_data(x, y, downsample))
    ds_val = TFRecordLoaderFull.window_batch(ds_val, group_window_size, batch_size)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)
    val_size = sum(1 for _ in ds_val) - 1

    # Turn the validation dataset into numpy arrays for the AUCCallback
    val_data = [ d for d in ds_val.take(min(val_size, validation_steps)) ]
    val_data_samples = np.array([ d[0] for d in val_data ])
    val_data_labels = np.concatenate([ d[1] for d in val_data ])

    print("Initialising model")
    if no_ae:
        model = TripletSplitConvModel(
            model_name,
            num_samples,
            2,
            layers,
            latent_dim,
            learning_rate,
            triplet_margin=triplet_margin,
            triplet_distance_metric=triplet_distance_metric,
            save_dir=save_dir
        )
    elif not vae:
        model = AETripletSplitConvModel(
            model_name,
            num_samples,
            2,
            layers,
            latent_dim,
            learning_rate,
            triplet_margin=triplet_margin,
            triplet_distance_metric=triplet_distance_metric,
            save_dir=save_dir
        )
    else:
        model = VAETripletSplitConvModel(
            model_name,
            num_samples,
            2,
            layers,
            latent_dim,
            learning_rate,
            triplet_margin=triplet_margin,
            triplet_distance_metric=triplet_distance_metric,
            #normalization=normalization,
            save_dir=save_dir
        )

    print("Loading checkpoint")
    try:
        model.load_model(suffix='checkpoint')
    except ValueError:
        print("No checkpoint found, model not loaded.")

    model.model.summary()

    callbacks = [
        AUCCallback(
            val_data_samples,
            val_data_labels,
            batch_size,
            no_ae=no_ae,
        ),
    ]
    if best_stop_after is not None:
        callbacks.append(EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=best_stop_after,
            restore_best_weights=True,
        ))
    if save_loss:
        callbacks.append(LossHistoryCallback(
            os.path.join(save_dir, model_name + '_loss.csv'),
            ['loss', 'embedding_loss', 'reconstruction_loss', 'val_loss', 'val_embedding_loss', 'val_reconstruction_loss', 'val_auc'],
        ))

    model.fit(
        ds_train,
        validation_data=ds_val,
        validation_steps=min(val_size, validation_steps),
        batch_size=batch_size,
        epochs=num_epochs,
        save_epochs=save_epochs,
        save_best_epoch=save_best_epoch,
        callbacks=callbacks,
    )

    model.save_model(
        suffix='final',
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an autoencoder with triplet loss.")
    parser.add_argument("--save-dir", type=str, default=save_dir, help="Directory to save models.")
    parser.add_argument("--data-dir", type=str, default=data_dir, help="Directory containing the TFRecord files.")
    parser.add_argument("--start-date", type=str, default=start_date, help="Start date for the dataset.")
    parser.add_argument("--num-days", type=int, default=num_days, help="Number of days to include in the dataset.")
    parser.add_argument("--num-epochs", type=int, default=num_epochs, help="Number of epochs to train for.")
    parser.add_argument("--model-name", type=str, default=model_name, help="Name of the model.")
    parser.add_argument("--save-epochs", dest='save_epochs', action='store_true', help="Save the model at the end of each epoch.")
    parser.add_argument("--save-best-epoch", dest='save_best_epoch', action='store_true', help="Save the best model based on validation loss.")
    parser.add_argument("--stop-after", type=int, default=None, help="Stop after this many epochs of no improvement.")
    parser.add_argument("--filter-property", type=str, default=None, help="Property to filter the dataset on.")
    parser.add_argument("--filter-value", type=float, default=None, help="Value to filter the dataset on.")
    parser.add_argument("--filter-gt", dest='filter_gt', action='store_true', help="Filter the dataset to values greater than the filter value.")
    parser.add_argument("--filter-labels", dest='filter_labels', action='store_true', help="Filter the dataset to remove certain labels.")
    parser.add_argument("--no-ae", dest='no_ae', action='store_true', help="Do not use an autoencoder model.")
    parser.add_argument("--vae", dest='vae', action='store_true', help="Use a VAE model instead of an AE model.")
    parser.add_argument("--triplet-margin", type=float, default=1.0, help="Triplet margin for the triplet loss function.")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample the dataset by this factor.")
    parser.add_argument("--save-loss", dest='save_loss', action='store_true', help="Save the loss history to a file.")
    args = parser.parse_args()

    if args.filter_property is not None and args.filter_value is not None:
        if args.filter_gt:
            filter_predicate = lambda x: x[args.filter_property] > args.filter_value
        else:
            filter_predicate = lambda x: x[args.filter_property] < args.filter_value
    elif args.filter_labels:
        filter_predicate = lambda x: not tf.reduce_any(tf.equal(x['id_cell'], filter_ids))
    else:
        filter_predicate = None

    main(
        args.save_dir,
        args.data_dir,
        args.start_date,
        args.num_days,
        args.num_epochs,
        args.model_name,
        args.save_epochs,
        args.save_best_epoch,
        args.stop_after,
        filter_predicate=filter_predicate,
        no_ae=args.no_ae,
        vae=args.vae,
        triplet_margin=args.triplet_margin,
        downsample=args.downsample,
        save_loss=args.save_loss,
    )
