import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import LambdaCallback

import tensorflow_addons as tfa

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


class Model(object):
    def __init__(self, model, name, save_dir=None):
        self.model = model
        self.name = name
        self.save_dir = save_dir or './models'

    def model_path(self, suffix=None):
        if suffix is None:
            suffix = ''
        else:
            suffix = '-' + suffix

        return os.path.join(self.save_dir, "{}{}.h5".format(self.name, suffix))

    def save_model(self, suffix=None):
        save_path = self.model_path(suffix)
        self.model.save(save_path)

    def load_model(self, suffix=None):
        load_path = self.model_path(suffix)
        if tf.io.gfile.exists(load_path):
            self.model.load_weights(load_path)
        else:
            raise ValueError("Model {} not found".format(load_path))

    def fit(self, dataset, epochs=10, batch_size=32, save_epochs=False, callbacks=[], **kwargs):
        if save_epochs:
            checkpoint_callback = LambdaCallback(
                on_epoch_end=lambda epoch, logs: self.save_model("checkpoint-{:04d}".format(epoch))
            )
            #checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            #    filepath=self.model_path('checkpoint'),
            #    verbose=1,
            #    save_freq='epoch',
            #    #save_weights_only=True,
            #)
            callbacks.append(checkpoint_callback)

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
            distances = tfa.losses.metric_learning.angular_distance(embeddings).numpy()
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
