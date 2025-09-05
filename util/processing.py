import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
try:
    import tensorflow_addons as tfa
except ImportError:
    tfa = None

from tqdm import tqdm

import functools

# Decorator to add optional output_pdf and show arguments to plot functions
def plot_d(func, save_dir):
    @functools.wraps(func)
    def wrapper(*args, output_pdf=None, dpi=None, show=True, **kwargs):
        res = func(*args, **kwargs)

        if output_pdf is not None:
            if dpi is None:
                plt.savefig(os.path.join(save_dir, output_pdf), bbox_inches='tight')
            else:
                plt.savefig(os.path.join(save_dir, output_pdf), bbox_inches='tight', dpi=dpi)
        if show:
            plt.show()
        plt.close()

        return res

    return wrapper

def get_embeddings(model, samples, autoencoder=True):
    """
    Encode the given samples using the given model.

    Args:
        model (tf.keras.Model): Model to use for encoding.
        samples (np.ndarray): Samples to encode.
        autoencoder (bool, optional): Whether the model has an autoencder, defaults to True.
    """
    embeddings = model.model.predict(samples)
    if autoencoder:
        embeddings = embeddings[0]
    return embeddings

def get_distances(embeddings):
    """
    Get the pairwise distances between the given embeddings, using angular distance.

    Args:
        embeddings (np.ndarray): Embeddings to get distances for.
    """
    if tfa is not None:
        distances = tfa.losses.metric_learning.angular_distance(embeddings).numpy()
        return distances
    else:
        return get_distances_numpy(embeddings)

def get_distances_numpy(embeddings):
    """
    Get the pairwise distances between the given embeddings, using angular distance.
    Uses numpy functions rather than tensorflow - should be faster outside training, but can be slightly less accurate.

    Args:
        embeddings (np.ndarray): Embeddings to get distances for.
    """
    feature = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    angular_distances = 1 - np.matmul(feature, feature.T)
    angular_distances = np.maximum(angular_distances, 0.0)
    return angular_distances

def get_distances_same_diff(distances, labels):
    """
    Split the given distances into same and different classes.

    Args:
        distances (np.ndarray): Distances to split.
        labels (np.ndarray): Labels for the distances.
    """
    mask = np.zeros(distances.shape, dtype=bool)
    np.fill_diagonal(mask, True)
    mask = mask | np.tri(*mask.shape, dtype=bool)
    mask = ~mask

    distances_same = distances[(labels[:, None] == labels) & mask]
    distances_diff = distances[(labels[:, None] != labels) & mask]

    return distances_same, distances_diff

def get_distances_same_diff_a_b(distances, labels_a, labels_b, same=False):
    """
    Split the given distances into same and different classes, for two different sets of labels.
    Distances are assumed to be constructed from the concatenation of the two sets of embeddings,
        and should therefore be a square array with size len(labels_a) + len(labels_b).

    Args:
        distances (np.ndarray): Distances to split.
        labels_a (np.ndarray): Labels for the first set of embeddings.
        labels_b (np.ndarray): Labels for the second set of embeddings.
        same (bool, optional): Excludes the diagonal.
    """
    # Construct mask to get only [0..len(labels_a)] x [len(labels_a)..len(labels_a) + len(labels_b)]
    mask = np.zeros(distances.shape, dtype=bool)
    mask[:len(labels_a), len(labels_a):] = True
    if same:
        if len(labels_a) != len(labels_b):
            raise ValueError("Same class comparison requires equal length labels.")
        for i in range(len(labels_a)):
            mask[i, i + len(labels_a)] = False

    labels = np.concatenate([labels_a, labels_b])

    distances_same = distances[(labels[:, None] == labels) & mask]
    distances_diff = distances[(labels[:, None] != labels) & mask]

    return distances_same, distances_diff

def true_false_positive_negative(distances_same, distances_diff, threshold):
    """
    Get the number of true/false positives/negatives for the given distances and threshold.
    """
    tp = len(distances_same[distances_same < threshold])
    fp = len(distances_diff[distances_diff < threshold])
    tn = len(distances_diff[distances_diff >= threshold])
    fn = len(distances_same[distances_same >= threshold])
    return (tp, fp, tn, fn)

def get_statistical_data(distances_same, distances_diff, resolution=1000, verbose=True):
    """
    Get statistical data for the given distances.

    Args:
        distances_same (np.ndarray): Distances between same class samples.
        distances_diff (np.ndarray): Distances between different class samples.
        resolution (int, optional): Number of thresholds to use, defaults to 1000.

    Returns:
        df_statistical (pd.DataFrame): Statistical data for the given distances.
    """
    max_threshold = max(distances_same.max(), distances_diff.max())
    thresholds = np.linspace(0, max_threshold, resolution)
    data = []
    for threshold in tqdm(thresholds, disable=not verbose):
        tp, fp, tn, fn = true_false_positive_negative(distances_same, distances_diff, threshold)
        data.append(dict(threshold=threshold, tp=tp, fp=fp, tn=tn, fn=fn))

    df_statistical = pd.DataFrame(data)
    df_statistical['tpr'] = df_statistical['tp'] / (df_statistical['tp'] + df_statistical['fn'])
    df_statistical['fpr'] = df_statistical['fp'] / (df_statistical['fp'] + df_statistical['tn'])
    df_statistical['fnr'] = df_statistical['fn'] / (df_statistical['tp'] + df_statistical['fn'])
    df_statistical['tnr'] = df_statistical['tn'] / (df_statistical['fp'] + df_statistical['tn'])
    df_statistical['precision'] = df_statistical['tp'] / (df_statistical['tp'] + df_statistical['fp'])
    df_statistical['recall'] = df_statistical['tp'] / (df_statistical['tp'] + df_statistical['fn'])
    df_statistical['f1'] = 2 * df_statistical['precision'] * df_statistical['recall'] / (df_statistical['precision'] + df_statistical['recall'])

    return df_statistical

# Get the AUC (Area Under Curve) of the ROC curve
def roc_auc(df_statistical):
    auc = np.trapz(df_statistical['tpr'], df_statistical['fpr'])

    return auc

def find_eer(distances_same, distances_diff, epsilon=0.00001):
    """
    Find the Equal Error Rate (EER) for the given distances.
    """
    max_threshold = max(distances_same.max(), distances_diff.max())

    threshold_l = 0.0
    threshold_r = max_threshold
    while threshold_r - threshold_l > epsilon:
        threshold = (threshold_l + threshold_r) / 2

        tp, fp, tn, fn = true_false_positive_negative(distances_same, distances_diff, threshold)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        fnr = 1 - tpr

        if fpr >= fnr:
            threshold_r = threshold
        else:
            threshold_l = threshold

    threshold = (threshold_l + threshold_r) / 2

    tp, fp, tn, fn = true_false_positive_negative(distances_same, distances_diff, threshold)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = 1 - tpr

    eer = (fpr + fnr) / 2

    return eer

# Edd's FFT function
def calculate_frequency_domain(time_domain_signal, sample_rate):
    fourier = np.abs(np.fft.fft(time_domain_signal))
    fourier /= time_domain_signal.size # Normalise by signal length
    timestep = 1/sample_rate
    n = time_domain_signal.size
    freq = np.fft.fftfreq(n, d=timestep)

    return (freq[np.argsort(freq)], fourier[np.argsort(freq)]) # Sort freq, fftfreq doesn't do this

def angular_distance(y_true, y_pred):
    """
    Calculate the angular distance between the given true and predicted values.
    """
    y_true = tf.math.l2_normalize(y_true, axis=-1)
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)

    return 1 - tf.reduce_sum(y_true * y_pred, axis=-1)