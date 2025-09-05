import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks, losses
from tensorflow.keras.callbacks import LambdaCallback

from scipy import signal

from .processing import get_distances, get_distances_same_diff, get_statistical_data, roc_auc

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

palette = 'colorblind'

sns.set_theme(palette=palette, color_codes=True)
sns.set_style('ticks')
sns.set_context('notebook')
sns.color_palette(palette)

plt.rcParams['figure.dpi'] = 72


class SamplingLayer(layers.Layer):
    """
    Sampling layer for a variational autoencoder.
    Uses (z_mean, z_log_var) to sample z.
    See https://keras.io/examples/generative/vae/.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = tf.random.Generator.from_seed(42)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        #epsilon = tf.random.normal(shape=(batch, dim), seed=self.seed_generator.make_seeds(2)[0])
        epsilon = self.seed_generator.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class MeanLoss(losses.Loss):
    """
    Computes the mean loss over a batch.
    """

    def __init__(self, weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight

    def call(self, y_true, y_pred):
        return self.weight * tf.reduce_mean(y_pred)


class AUCCallback(callbacks.Callback):
    def __init__(self, samples, labels, batch_size, resolution=100, no_ae=False, **kwargs):
        super(AUCCallback, self).__init__(**kwargs)
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size
        self.resolution = resolution
        self.no_ae = no_ae
        self.auc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        embeddings = []
        if self.no_ae:
            for i in range(len(self.samples)):
                embeddings.append(self.model.predict_on_batch(self.samples[i]))
        else:
            for i in range(len(self.samples)):
                embeddings.append(self.model.predict_on_batch(self.samples[i])[0])
        embeddings = np.concatenate(embeddings)
        distances = get_distances(embeddings)
        distances_same, distances_diff = get_distances_same_diff(distances, self.labels)
        self.auc = roc_auc(get_statistical_data(distances_same, distances_diff, resolution=self.resolution, verbose=False))

        logs = logs or {}
        logs['val_auc'] = self.auc


class Model(object):
    def __init__(self, model, name, save_dir=None):
        self.model = model
        self.name = name
        self.save_dir = save_dir or './models'

        self.best_loss = float('inf')
        self.best_epoch = -1

    def model_path(self, suffix=None):
        if suffix is None:
            suffix = ''
        else:
            suffix = '-' + suffix

        return os.path.join(self.save_dir, "{}{}.h5".format(self.name, suffix))

    def save_model(self, suffix=None):
        save_path = self.model_path(suffix)
        self.model.save(save_path)

    def _save_best_epoch(self, epoch, logs):
        if logs['val_loss'] < self.best_loss:
            self.best_loss = logs['val_loss']
            self.best_epoch = epoch
            self.save_model("best")

    def load_model(self, suffix=None):
        load_path = self.model_path(suffix)
        if tf.io.gfile.exists(load_path):
            self.model.load_weights(load_path)
        else:
            raise ValueError("Model {} not found".format(load_path))

    def fit(self, dataset, epochs=10, batch_size=32, save_epochs=False, save_best_epoch=False, callbacks=[], **kwargs):
        if save_epochs:
            checkpoint_callback = LambdaCallback(
                on_epoch_end=lambda epoch, logs: self.save_model("checkpoint-{:04d}".format(epoch))
            )
            callbacks.append(checkpoint_callback)

        if save_best_epoch:
            best_epoch_callback = LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._save_best_epoch(epoch, logs)
            )
            callbacks.append(best_epoch_callback)

            train_end_callback = LambdaCallback(
                on_train_end=lambda logs: print(f"Best epoch: {self.best_epoch} (loss: {self.best_loss})")
            )
            callbacks.append(train_end_callback)

        self.model.fit(
            dataset,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            **kwargs
        )

class AutoencoderModel(Model):
    def __init__(self, model, name, encoding_layer, generator_layer, save_dir=None):
        self.encoding_layer = encoding_layer
        self.generator_layer = generator_layer

        self.encoding_model = models.Model(
            inputs=model.input,
            outputs=model.get_layer(self.encoding_layer).output
        )
        #self.generator_model = models.Model(
        #    inputs=model.get_layer(self.generator_layer).input,
        #    outputs=model.output
        #)

        super().__init__(model, name, save_dir)

    def encode(self, input):
        return self.encoding_model.predict(input)

    #def generate(self, input):
    #    return self.generator_model.predict(input)

class WeightedLayer(layers.Layer):
    def __init__(self, kernel_regularizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=(int(input_shape[-1]),), regularizer=self.kernel_regularizer)

    def call(self, inputs):
        return tf.multiply(inputs, self.kernel)

class SiameseAccuracy(tf.keras.metrics.Metric):
    """
    Computes an accuracy metric for a siamese network.
    """

    def __init__(self, name='siamese_accuracy', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.embeddings = []
        self.labels = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.embeddings.append(y_pred)
        self.labels.append(y_true)

    def result(self):
        embeddings = tf.concat(self.embeddings, axis=0)
        labels = tf.concat(self.labels, axis=0)

        label_equality = tf.equal(labels[:, None], labels[None, :])

        distances = tf.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=-1)
        predicted_matches = distances < self.threshold

        correct_matches = tf.logical_and(label_equality, predicted_matches)

        accuracy = tf.reduce_mean(tf.cast(correct_matches, tf.float32))
        return accuracy

    def reset_states(self):
        self.embeddings = []
        self.labels = []

class RocAucCallback(tf.keras.callbacks.Callback):
    """
    Callback that plots the ROC curve and computes the AUC score after each epoch.
    """

    def __init__(self, validation_data, has_autoencoder=False, eer=False, resolution=1000, distance_metric='L2', save_dir='./plots', model_name='roc_auc'):
        """
        Args:
            validation_data: Validation data to use for the ROC curve.
            has_autoencoder: Set to True if the model has an attached decoder.
            eer: Compute and plot the equal error rate alongside the ROC curve.
            max_threshold: Maximum threshold to use.
            resolution: Number of points to use for the ROC curve.
            distance_metric: Distance metric to use for the ROC curve.
            save_dir: Directory to save the plots after each epoch.
            model_name: Name of the model.
        """
        self.validation_data = validation_data
        self.has_autoencoder = has_autoencoder
        self.eer = eer
        self.resolution = resolution
        self.distance_metric = distance_metric
        self.save_dir = save_dir
        self.model_name = model_name

    def _pairwise_distances(self, embeddings):
        """
        Computes the pairwise distances between the embeddings.
        """
        if self.distance_metric == 'L2':
            distances = np.zeros((len(embeddings), len(embeddings)))
            for i in range(len(embeddings)):
                for j in range(len(embeddings)):
                    distances[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])
            return distances
        elif self.distance_metric == 'squared-L2':
            distances = np.zeros((len(embeddings), len(embeddings)))
            for i in range(len(embeddings)):
                for j in range(len(embeddings)):
                    distances[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])**2
            return distances
        elif self.distance_metric == 'angular':
            distances = get_distances(embeddings)
            return distances

        else:
            raise ValueError('Invalid distance metric.')

    def _tpr_fpr(self, distances_same, distances_diff, threshold):
        """
        Computes the true positive rate and false positive rate for a given threshold.
        """
        tpr = len(distances_same[distances_same < threshold]) / len(distances_same)
        fpr = len(distances_diff[distances_diff < threshold]) / len(distances_diff)
        return (tpr, fpr)

    def _plot_roc_auc(self, df_tpr_fpr, output_pdf):
        """
        Plots the ROC curve and computes the AUC score.

        Args:
            df_tpr_fpr: Dataframe with the true positive rate and false positive rate for each threshold.
            output_pdf: Path to the output PDF file.

        Returns:
            The AUC score.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        #g = sns.relplot(x="fpr", y="tpr", kind='line', data=df_tpr_fpr)
        sns.lineplot(x="fpr", y="tpr", data=df_tpr_fpr, ax=ax)
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--') # Draw diagonal line
        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_box_aspect(1)

        # Display AUC (Area Under Curve)
        auc = np.trapz(df_tpr_fpr['tpr'], df_tpr_fpr['fpr'])
        ax.text(0.99, 0.01, f'AUC: {auc:.3f}', horizontalalignment='right', verticalalignment='bottom')

        plt.savefig(os.path.join(self.save_dir, output_pdf), bbox_inches='tight')

        plt.close()

        return auc

    # Find EER (crossing point of FPR and FNR) using a binary search
    def _find_eer(self, distances_same, distances_diff, max_threshold):
        threshold_l = 0.0
        threshold_r = max_threshold
        while threshold_r - threshold_l > 0.00001:
            threshold = (threshold_l + threshold_r) / 2
            tpr, fpr = self._tpr_fpr(distances_same, distances_diff, threshold)
            fnr = 1 - tpr

            if fpr >= fnr:
                threshold_r = threshold
            else:
                threshold_l = threshold

        threshold = (threshold_l + threshold_r) / 2
        tpr, fpr = self._tpr_fpr(distances_same, distances_diff, threshold)
        fnr = 1 - tpr

        eer = (fpr + fnr) / 2
        return eer

    def _plot_eer(self, df_tpr_fpr, eer, max_threshold, output_pdf):
        df = df_tpr_fpr.copy()

        df['FNR'] = 1 - df['tpr']
        df['FPR'] = df['fpr']
        del df['tpr']
        del df['fpr']

        df = df.melt(id_vars=['threshold'], value_vars=['FPR', 'FNR'], var_name='Rate', value_name='value')

        df['value'] *= 0.998 # Hack to make sure the line is not drawn on the border

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.lineplot(x="threshold", y="value", hue="Rate", data=df, ax=ax)
        #sns.despine()
        ax.set(xlabel='Threshold', ylabel='Rate')
        ax.set(xlim=(0, max_threshold), ylim=(0, 1))
        sns.move_legend(ax, 'upper right')
        ax.set_box_aspect(1)

        ax.plot([0, max_threshold], [eer, eer], color='gray', linestyle='--')
        ax.text(0.01, eer - 0.01, f'EER: {eer:.3f}', horizontalalignment='left', verticalalignment='top')

        plt.savefig(os.path.join(self.save_dir, output_pdf), bbox_inches='tight')

        plt.close()

    def on_epoch_end(self, epoch, logs={}):
        x_val, y_val = self.validation_data
        embeddings = None
        if not self.has_autoencoder:
            embeddings = self.model.predict(x_val)
        else:
            embeddings, _ = self.model.predict(x_val)

        # Pairwise distances
        distances = self._pairwise_distances(embeddings)

        mask = np.zeros((embeddings.shape[0], embeddings.shape[0]), dtype=bool)
        np.fill_diagonal(mask, True)
        mask = ~mask

        distances_same = distances[(y_val[:, None] == y_val) & mask]
        distances_diff = distances[y_val[:, None] != y_val]

        max_threshold = np.max(distances)

        thresholds = np.linspace(0, max_threshold, self.resolution)
        data = []
        for threshold in thresholds:
            tpr, fpr = self._tpr_fpr(distances_same, distances_diff, threshold)
            data.append(dict(threshold=threshold, tpr=tpr, fpr=fpr))

        df_tpr_fpr = pd.DataFrame(data)

        auc = self._plot_roc_auc(df_tpr_fpr, "{}-{:04d}.pdf".format(self.model_name, epoch))
        print("AUC: {}".format(auc))

        if self.eer:
            eer = self._find_eer(distances_same, distances_diff, max_threshold)
            self._plot_eer(df_tpr_fpr, eer, max_threshold, "{}-eer-{:04d}.pdf".format(self.model_name, epoch))
            print("EER: {}".format(eer))

class LearnableConstant(layers.Layer):
    """
    A layer that learns a constant vector of the specified shape.
    """
    def __init__(self, shape):
        super(LearnableConstant, self).__init__()
        self.learnable_constant = tf.Variable(tf.random.normal((1, *shape)), trainable=True)

    def get_config(self):
        config = super().get_config()
        config.update({"shape": self.learnable_constant.shape})
        return config

    def call(self, inputs=None):
        return tf.tile(self.learnable_constant, (tf.shape(inputs)[0], 1, 1))

class EnergyNormalise(layers.Layer):
    """
    Normalise the energy of the input by dividing by the total energy.
    """
    def __init__(self, shape, scale=1.0, scale_raw=None, polar=False):
        """
        Initialise the layer.

        Args:
            shape: The shape of the input tensor.
            scale: The scale factor to apply to the normalised energy.
            scale_raw: "Raw" scale factor (not scaled by a magic number).
            polar: Whether the input is in polar (magnitude, phase) or cartesian (real, imaginary) form.
        """
        super(EnergyNormalise, self).__init__()
        self.shape = shape
        self.scale = scale_raw if scale_raw is not None else scale * 0.008
        self.polar = polar

        # Ensure the shape is of the form (batch, samples, features)
        assert(len(shape) == 3)

    def get_config(self):
        config = super().get_config()
        config.update({"shape": self.shape, "scale": self.scale, "polar": self.polar})
        return config

    def call(self, inputs):
        if not self.polar:
            energy = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.square(inputs), axis=2, keepdims=False), axis=1, keepdims=False))
            energy = energy[:, tf.newaxis, tf.newaxis]
            # Clamp the energy to prevent division by zero
            energy = tf.maximum(energy, 1e-6)
            # Normalise the input by the total energy
            inputs_scaled = inputs / energy * self.scale * self.shape[1]

        else:
            # Split the input into magnitude and phase
            magnitude = inputs[:, :, 0]
            phase = inputs[:, :, 1]

            energy = tf.sqrt(tf.reduce_sum(tf.square(magnitude), axis=1, keepdims=False))
            energy = energy[:, tf.newaxis]
            # Clamp the energy to prevent division by zero
            energy = tf.maximum(energy, 1e-6)
            # Normalise the magnitude by the total energy
            magnitude_scaled = magnitude / energy * self.scale * self.shape[1]
            # Combine the magnitude and phase
            inputs_scaled = tf.stack([magnitude_scaled, phase], axis=-1)

        return inputs_scaled

class PhaseRotation(layers.Layer):
    """
    Apply a random phase rotation to the input.
    """
    def __init__(self, polar=False):
        """
        Initialise the layer.

        Args:
            polar: Whether the input is in polar (magnitude, phase) or cartesian (real, imaginary) form.
        """
        super(PhaseRotation, self).__init__()
        self.polar = polar

    def get_config(self):
        config = super().get_config()
        config.update({"polar": self.polar})
        return config

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Generate a random phase angle for each batch
        phases = tf.random.uniform(shape=(batch_size,), minval=0, maxval=2*np.pi)

        if not self.polar:
            # Convert to complex numbers
            complex_phases = tf.complex(tf.cos(phases), tf.sin(phases))
            # Reshape to match input
            complex_phases = tf.expand_dims(complex_phases, axis=1)
            # Convert input to complex numbers
            inputs_complex = tf.complex(inputs[:, :, 0], inputs[:, :, 1])
            # Apply the phase rotation
            inputs_rotated = inputs_complex * complex_phases
            # Convert back to real numbers
            inputs_rotated = tf.stack([tf.math.real(inputs_rotated), tf.math.imag(inputs_rotated)], axis=-1)
        else:
            # Reshape to match input
            phases = tf.expand_dims(phases, axis=1)

            magnitude = inputs[:, :, 0]
            phase = inputs[:, :, 1]

            # Apply the phase rotation
            phase_rotated = phase + phases

            inputs_rotated = tf.stack([magnitude, phase_rotated], axis=-1)

        return inputs_rotated

class PolarToCartesian(layers.Layer):
    """
    Convert polar coordinates to cartesian coordinates.
    """
    def call(self, inputs):
        magnitude = inputs[:, :, 0]
        phase = inputs[:, :, 1]

        real = magnitude * tf.cos(phase)
        imag = magnitude * tf.sin(phase)

        return tf.stack([real, imag], axis=-1)

class CartesianToPolar(layers.Layer):
    """
    Convert cartesian coordinates to polar coordinates.
    """
    def call(self, inputs):
        real = inputs[:, :, 0]
        imag = inputs[:, :, 1]

        magnitude = tf.sqrt(tf.square(real) + tf.square(imag))
        phase = tf.math.atan2(imag, real)

        return tf.stack([magnitude, phase], axis=-1)

class LossHistoryCallback(tf.keras.callbacks.Callback):
    """
    Callback that writes the loss history to a CSV file.
    """
    def __init__(self, filename='loss_history.csv', columns=['loss', 'val_loss']):
        """
        Initialise the callback and write the header to the file.

        Args:
            filename: Name of the CSV file to write to, relative to the working directory.
            columns: Columns to write to the file.
        """
        super().__init__()
        self.filename = filename
        self.columns = columns
        with open(self.filename, 'w') as f:
            f.write("epoch,{}\n".format(','.join(columns)))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        data = [ logs.get(column) for column in self.columns ]
        with open(self.filename, 'a') as f:
            f.write("{},{}\n".format(epoch, ','.join(map(str, data))))

class ScalingLayer(layers.Layer):
    """
    Single learnable scale factor, by which the input is multiplied.
    """
    def __init__(self):
        super(ScalingLayer, self).__init__()

    def build(self, input_shape):
        self.scale = self.add_weight("scale", shape=(1,), initializer='ones', trainable=True)

    def call(self, inputs):
        return inputs * self.scale


class ShiftScalingLayer(layers.Layer):
    """
    Learnable shift and scale factors, by which the input is shifted and multiplied.
    """
    def __init__(self):
        super(ShiftScalingLayer, self).__init__()

    def build(self, input_shape):
        self.shift = self.add_weight("shift", shape=(1,), initializer='zeros', trainable=True)
        self.scale = self.add_weight("scale", shape=(1,), initializer='ones', trainable=True)

    def call(self, inputs):
        return (inputs + self.shift) * self.scale


class AffineTransformLayer(layers.Layer):
    """
    Learnable affine transform (per-element shift and scale), by which the input is shifted and multiplied.
    """
    def __init__(self):
        super(AffineTransformLayer, self).__init__()

    def build(self, input_shape):
        self.scale = self.add_weight("scale", shape=input_shape[1:], initializer='ones', trainable=True)
        self.shift = self.add_weight("shift", shape=input_shape[1:], initializer='zeros', trainable=True)

    def call(self, inputs):
        return (inputs * self.scale) + self.shift


class PhaseRotationLayer(layers.Layer):
    """
    Single learnable phase rotation factor, by which the input is rotated.
    """
    def __init__(self):
        super(PhaseRotationLayer, self).__init__()

    def build(self, input_shape):
        self.phase = self.add_weight("phase", shape=(1,), initializer='zeros', trainable=True)

    def call(self, inputs):
        cos_phase = tf.cos(self.phase)
        sin_phase = tf.sin(self.phase)

        return tf.stack([
            inputs[..., 0] * tf.cos(self.phase) - inputs[..., 1] * tf.sin(self.phase),
            inputs[..., 0] * tf.sin(self.phase) + inputs[..., 1] * tf.cos(self.phase)
        ], axis=-1)


class CosineDistance(layers.Layer):
    """
    Compute the (pairwise) cosine distance between two embedding tensors.
    """
    def __init__(self, min_value=-1024.0, max_value=1024.0):
        super(CosineDistance, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def get_config(self):
        config = super().get_config()
        config.update({"min_value": self.min_value, "max_value": self.max_value})
        return config

    def call(self, inputs):
        inputs_l, inputs_r = inputs

        prox = tf.matmul(
            tf.linalg.normalize(inputs_l, axis=1)[0],
            tf.linalg.normalize(inputs_r, axis=1)[0],
            transpose_b=True,
        )
        dist = 1 - prox
        dist = tf.maximum(dist, self.min_value)
        dist = tf.minimum(dist, self.max_value)

        return dist

def butter_coefficients(order, normalized_cutoff, btype):
    b, a = signal.butter(order, normalized_cutoff, btype=btype)
    b = tf.constant(b, dtype=tf.float32)
    a = tf.constant(a, dtype=tf.float32)
    return b, a

def low_pass_filter(signal_in, cutoff=1e6, fs=25e6, order=8):
    nyq = fs / 2
    normalized_cutoff = cutoff / nyq

    b, a = signal.butter(order, normalized_cutoff, btype='low')

    filtered_i = signal.filtfilt(b, a, signal_in[...,0])
    filtered_q = signal.filtfilt(b, a, signal_in[...,1])

    return tf.cast(tf.stack([filtered_i, filtered_q], axis=-1), tf.float32)

@tf.function
def tf_low_pass_filter(signal_in, cutoff=1e6, fs=25e6, order=8):
    return tf.map_fn(lambda x: tf.numpy_function(low_pass_filter, [x, cutoff, fs, order], tf.float32), signal_in)

class LowPassFilterLayer(layers.Layer):
    def __init__(self, cutoff=1e6, fs=25e6, order=8):
        super(LowPassFilterLayer, self).__init__()
        self.cutoff = cutoff
        self.fs = fs
        self.order = order

    def get_config(self):
        config = super().get_config()
        config.update({"cutoff": self.cutoff, "fs": self.fs, "order": self.order})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        outputs = tf_low_pass_filter(inputs, cutoff=self.cutoff, fs=self.fs, order=self.order)
        outputs.set_shape(inputs.shape)
        return outputs

@tf.function
def angular_distance(a, b):
    a_norm = tf.nn.l2_normalize(a, axis=0)
    b_norm = tf.nn.l2_normalize(b, axis=0)
    dot_product = tf.tensordot(a_norm, b_norm, axes=1)
    angular_distance = 1 - dot_product
    angular_distance = tf.maximum(angular_distance, 0.0)
    return angular_distance

@tf.function
def angular_distances_pairwise(a, b):
    a_norm = tf.nn.l2_normalize(a, axis=1)
    b_norm = tf.nn.l2_normalize(b, axis=1)
    dot_product = tf.matmul(a_norm, b_norm, transpose_b=True)
    angular_distance = 1 - dot_product
    angular_distance = tf.maximum(angular_distance, 0.0)
    return angular_distance
