from .model_utils import Model, SiameseAccuracy

import tensorflow as tf
from tensorflow.keras import layers, models, metrics, losses
from tensorflow.keras.callbacks import LambdaCallback

import tensorflow_addons as tfa

class UpscalingLayer(layers.Layer):
    def __init__(self, out_shape):
        super(UpscalingLayer, self).__init__()
        self.out_shape = out_shape

    def call(self, inputs, **kwargs):
        output = tf.repeat(inputs, 2, axis=1)
        return tf.ensure_shape(output, self.out_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'out_shape': self.out_shape,
        })
        return config


"""
A simple siamese model with convolutional layers, followed by a dense layer and a difference layer.
Model architecture based on https://github.com/ssloxford/auto-phy-fingerprint/blob/main/code/fingerprinting/adsb-siamese/train.py
"""
class SimpleSiameseConvModel(Model):
    def build_model(self, input_len, feature_count, conv_layers, dense_size, learning_rate):
        """
        Build the siamese model.

        Args:
            input_len (int): Length of the input vector.
            feature_count (int): Number of features in the input vector.
            conv_layers (list): List of tuples (filters, kernel_size) for each convolutional layer.
            dense_size (int): Size of the dense layer.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
            tf.keras.Model: The siamese model.
        """

        input_left = layers.Input(shape=(input_len, feature_count), name="input_left")
        input_right = layers.Input(shape=(input_len, feature_count), name="input_right")

        input_dummy = layers.Input(shape=(input_len, feature_count), name="input_dummy")
        input_norm = layers.BatchNormalization()(input_dummy)
        input_padded = layers.ZeroPadding1D(padding=2)(input_norm)

        conv = [input_padded]
        for i, (filters, kernel_size) in enumerate(conv_layers):
            conv.append(layers.Conv1D(filters, kernel_size, activation='relu', name="conv_{}".format(i))(conv[-1]))
            conv.append(layers.MaxPooling1D()(conv[-1]))

        flat = layers.Flatten()(conv[-1])
        dense = layers.Dense(dense_size, activation='sigmoid', name="dense")(flat)

        extractor = models.Model(inputs=[input_dummy], outputs=[dense], name="extractor")

        encoded_left = models.Model(inputs=[input_left], outputs=[extractor.call(input_left)], name="encoded_left")
        encoded_right = models.Model(inputs=[input_right], outputs=[extractor.call(input_right)], name="encoded_right")

        difference = layers.Subtract(name="difference")([encoded_left.output, encoded_right.output])
        difference_abs = layers.Lambda(lambda x: tf.abs(x), name="difference_abs")(difference)

        prediction = layers.Dense(1, activation='sigmoid', name="output")(difference_abs)

        model = models.Model(inputs=[input_left, input_right], outputs=[prediction], name="siamese")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def __init__(self, name, input_len, feature_count, conv_layers, dense_size, learning_rate, save_dir=None):
        """
        Initialize the siamese model.

        Args:
            name (str): Name of the model.
            input_len (int): Length of the input vector.
            feature_count (int): Number of features in the input vector.
            conv_layers (list): List of tuples (filters, kernel_size) for each convolutional layer.
            dense_size (int): Size of the dense layer.
            learning_rate (float): Learning rate for the optimizer.
            save_dir (str): Directory to save the model to.
        """

        self.model = self.build_model(input_len, feature_count, conv_layers, dense_size, learning_rate)

        super().__init__(self.model, name, save_dir=save_dir)


"""
A simple siamese model with convolutional layers (split into separate IQ layers), followed by a dense layer and a difference layer.
Model architecture based on https://github.com/ssloxford/auto-phy-fingerprint/blob/main/code/fingerprinting/adsb-siamese/train.py
"""
class SimpleSiameseSplitConvModel(Model):
    def build_model(self, input_len, feature_count, conv_layers, dense_size, learning_rate):
        """
        Build the siamese model.

        Args:
            input_len (int): Length of the input vector.
            feature_count (int): Number of features in the input vector.
            conv_layers (list): List of tuples (filters, kernel_size) for each convolutional layer.
            dense_size (int): Size of the dense layer.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
            tf.keras.Model: The siamese model.
        """

        input_left = layers.Input(shape=(input_len, feature_count), name="input_left")
        input_right = layers.Input(shape=(input_len, feature_count), name="input_right")

        input_dummy = layers.Input(shape=(input_len, feature_count), name="input_dummy")
        input_norm = layers.BatchNormalization()(input_dummy)
        input_padded = layers.ZeroPadding1D(padding=2)(input_norm)

        input_i, input_q = tf.split(input_padded, feature_count, axis=2)

        conv_i = [input_i]
        conv_q = [input_q]
        for i, (filters, kernel_size) in enumerate(conv_layers):
            conv_i.append(layers.Conv1D(filters, kernel_size, activation='relu', name="conv_i_{}".format(i))(conv_i[-1]))
            conv_i.append(layers.MaxPooling1D()(conv_i[-1]))

            conv_q.append(layers.Conv1D(filters, kernel_size, activation='relu', name="conv_{}".format(i))(conv_q[-1]))
            conv_q.append(layers.MaxPooling1D()(conv_q[-1]))

        conv_out = layers.Concatenate(axis=2, name="conv_out")([conv_i[-1], conv_q[-1]])

        flat = layers.Flatten()(conv_out)
        dense = layers.Dense(dense_size, activation='sigmoid', name="dense")(flat)

        extractor = models.Model(inputs=[input_dummy], outputs=[dense], name="extractor")

        encoded_left = models.Model(inputs=[input_left], outputs=[extractor.call(input_left)], name="encoded_left")
        encoded_right = models.Model(inputs=[input_right], outputs=[extractor.call(input_right)], name="encoded_right")

        difference = layers.Subtract(name="difference")([encoded_left.output, encoded_right.output])
        difference_abs = layers.Lambda(lambda x: tf.abs(x), name="difference_abs")(difference)

        prediction = layers.Dense(1, activation='sigmoid', name="output")(difference_abs)

        model = models.Model(inputs=[input_left, input_right], outputs=[prediction], name="siamese")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def __init__(self, name, input_len, feature_count, conv_layers, dense_size, learning_rate, save_dir=None):
        """
        Initialize the siamese model.

        Args:
            name (str): Name of the model.
            input_len (int): Length of the input vector.
            feature_count (int): Number of features in the input vector.
            conv_layers (list): List of tuples (filters, kernel_size) for each convolutional layer.
            dense_size (int): Size of the dense layer.
            learning_rate (float): Learning rate for the optimizer.
            save_dir (str): Directory to save the model to.
        """

        if feature_count != 2:
            raise ValueError("This model currently only supports 2D input vectors (IQ samples).")

        self.model = self.build_model(input_len, feature_count, conv_layers, dense_size, learning_rate)

        super().__init__(self.model, name, save_dir=save_dir)


"""
A simple siamese model with convolutional layers (split into separate IQ layers), followed by a dense layer.
Triplet loss is used.
Model architecture based on https://github.com/ssloxford/auto-phy-fingerprint/blob/main/code/fingerprinting/adsb-siamese/train.py
"""
class SimpleTripletSplitConvModel(Model):
    def siamese_accuracy(self, y_true, y_pred):
        """
        Calculate the accuracy of the siamese model.

        Args:
            y_true (tf.Tensor): True labels.
            y_pred (tf.Tensor): Embeddings.
        """

        label_equality = tf.equal(y_true[:, None], y_true[None, :])

        distances = tf.linalg.norm(y_pred[:, None, :] - y_pred[None, :, :], axis=-1)
        predicted_matches = distances < self.accuracy_threshold

        correct_matches = tf.logical_and(label_equality, predicted_matches)

        accuracy = tf.reduce_mean(tf.cast(correct_matches, tf.float32))
        return accuracy

    def siamese_tpr(self, y_true, y_pred):
        """
        Calculate the TPR of the siamese model.

        Args:
            y_true (tf.Tensor): True labels.
            y_pred (tf.Tensor): Embeddings.
        """

        label_equality = tf.equal(y_true[:, None], y_true[None, :])

        distances = tf.linalg.norm(y_pred[:, None, :] - y_pred[None, :, :], axis=-1)
        predicted_matches = distances < self.accuracy_threshold

        #true_positives = tf.logical_and(label_equality, predicted_matches)
        #tpr = tf.reduce_sum(tf.cast(true_positives, tf.float32)) / tf.reduce_sum(tf.cast(label_equality, tf.float32))

        true_positives = tf.logical_and(label_equality, predicted_matches)
        tp = tf.reduce_sum(tf.cast(true_positives, tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(label_equality), predicted_matches), tf.float32))
        tpr = tp / (tp + fn)
        return tpr

    def siamese_fpr(self, y_true, y_pred):
        """
        Calculate the FPR of the siamese model.

        Args:
            y_true (tf.Tensor): True labels.
            y_pred (tf.Tensor): Embeddings.
        """

        label_equality = tf.equal(y_true[:, None], y_true[None, :])

        distances = tf.linalg.norm(y_pred[:, None, :] - y_pred[None, :, :], axis=-1)
        predicted_matches = distances < self.accuracy_threshold

        #false_positives = tf.logical_and(tf.logical_not(label_equality), predicted_matches)
        #fpr = tf.reduce_sum(tf.cast(false_positives, tf.float32)) / tf.reduce_sum(tf.cast(tf.logical_not(label_equality), tf.float32))
        false_positives = tf.logical_and(tf.logical_not(label_equality), predicted_matches)
        fp = tf.reduce_sum(tf.cast(false_positives, tf.float32))
        tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(label_equality), tf.logical_not(predicted_matches)), tf.float32))
        fpr = fp / (fp + tn)
        return fpr

    def siamese_fscore(self, y_true, y_pred):
        """
        Calculate the F-score of the siamese model.

        Args:
            y_true (tf.Tensor): True labels.
            y_pred (tf.Tensor): Embeddings.
        """

        label_equality = tf.equal(y_true[:, None], y_true[None, :])

        distances = tf.linalg.norm(y_pred[:, None, :] - y_pred[None, :, :], axis=-1)
        predicted_matches = distances < self.accuracy_threshold

        true_positives = tf.logical_and(label_equality, predicted_matches)
        false_positives = tf.logical_and(tf.logical_not(label_equality), predicted_matches)
        false_negatives = tf.logical_and(label_equality, tf.logical_not(predicted_matches))

        tp = tf.reduce_sum(tf.cast(true_positives, tf.float32))
        fp = tf.reduce_sum(tf.cast(false_positives, tf.float32))
        fn = tf.reduce_sum(tf.cast(false_negatives, tf.float32))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_score = 2 * precision * recall / (precision + recall)
        return f_score

    def build_model(self, input_len, feature_count, conv_layers, dense_size, learning_rate, triplet_margin, triplet_distance_metric, normalization):
        """
        Build the siamese model.

        Args:
            input_len (int): Length of the input vector.
            feature_count (int): Number of features in the input vector.
            conv_layers (list): List of tuples (filters, kernel_size) for each convolutional layer.
            dense_size (int): Size of the dense layer.
            learning_rate (float): Learning rate for the optimizer.
            triplet_margin (float): Margin for the triplet loss.
            triplet_distance_metric (str): Distance metric to use for the triplet loss. One of 'L2', 'squared-L2', 'angular', or a callable.
            normalization (str): Normalization to use for the embeddings. One of 'L1', 'L2', None.

        Returns:
            tf.keras.Model: The siamese model.
        """

        input = layers.Input(shape=(input_len, feature_count), name="input")
        input_norm = layers.BatchNormalization()(input)
        input_padded = layers.ZeroPadding1D(padding=2)(input_norm)

        input_i, input_q = tf.split(input_padded, feature_count, axis=2)

        conv_i = [input_i]
        conv_q = [input_q]
        for i, (filters, kernel_size) in enumerate(conv_layers):
            conv_i.append(layers.Conv1D(filters, kernel_size, activation='relu', name="conv_i_{}".format(i))(conv_i[-1]))
            conv_i.append(layers.MaxPooling1D()(conv_i[-1]))

            conv_q.append(layers.Conv1D(filters, kernel_size, activation='relu', name="conv_{}".format(i))(conv_q[-1]))
            conv_q.append(layers.MaxPooling1D()(conv_q[-1]))

        conv_out = layers.Concatenate(axis=2, name="conv_out")([conv_i[-1], conv_q[-1]])

        flat = layers.Flatten()(conv_out)
        dense = layers.Dense(dense_size, activation=None, name="dense")(flat)
        norm = None
        if normalization == 'l1':
            norm = layers.Lambda(lambda x: tf.linalg.normalize(x, ord=1, axis=1)[0], name="l1_norm")(dense)
        elif normalization == 'l2':
            norm = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="l2_norm")(dense)
        elif normalization is None:
            norm = dense

        model = models.Model(inputs=[input], outputs=[norm], name="model")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tfa.losses.TripletSemiHardLoss(
                margin=triplet_margin,
                distance_metric=triplet_distance_metric,
            ),
            metrics=[
                #self.siamese_accuracy,
                #self.siamese_tpr,
                #self.siamese_fpr,
                #self.siamese_fscore,
            ]
        )

        return model

    def __init__(self, name, input_len, feature_count, conv_layers, dense_size, learning_rate, triplet_margin=1.0, triplet_distance_metric='L2', normalization='L2', accuracy_threshold=0.5, save_dir=None):
        """
        Initialize the siamese model.

        Args:
            name (str): Name of the model.
            input_len (int): Length of the input vector.
            feature_count (int): Number of features in the input vector.
            conv_layers (list): List of tuples (filters, kernel_size) for each convolutional layer.
            dense_size (int): Size of the dense layer.
            learning_rate (float): Learning rate for the optimizer.
            triplet_margin (float): Margin for the triplet loss.
            triplet_distance_metric (str): Distance metric to use for the triplet loss. One of 'L2', 'squared-L2', 'angular', or a callable.
            normalization (str): Normalization to use for the embeddings. One of 'L1', 'L2', None.
            accuracy_threshold (float): Accuracy threshold for the model's SiameseAccuracy metric.
            save_dir (str): Directory to save the model to.
        """

        self.accuracy_threshold = accuracy_threshold

        if feature_count != 2:
            raise ValueError("This model currently only supports 2D input vectors (IQ samples).")

        if normalization is not None:
            normalization = normalization.lower()
            if normalization not in ['l1', 'l2']:
                raise ValueError("Invalid normalization: {}".format(normalization))

        self.model = self.build_model(input_len, feature_count, conv_layers, dense_size, learning_rate, triplet_margin, triplet_distance_metric, normalization)

        super().__init__(self.model, name, save_dir=save_dir)


"""
A simple siamese model with convolutional layers (split into separate IQ layers), followed by a dense layer.
Triplet loss is used.
"""
class AETripletSplitConvModel(Model):
    def build_model(self, input_len, feature_count, conv_layers, dense_size, learning_rate, triplet_margin, triplet_distance_metric, normalization):
        """
        Build the siamese model.

        Args:
            input_len (int): Length of the input vector.
            feature_count (int): Number of features in the input vector.
            conv_layers (list): List of tuples (filters, kernel_size) for each convolutional layer.
            dense_size (int): Size of the dense layer.
            learning_rate (float): Learning rate for the optimizer.
            triplet_margin (float): Margin for the triplet loss.
            triplet_distance_metric (str): Distance metric to use for the triplet loss. One of 'L2', 'squared-L2', 'angular', or a callable.
            normalization (str): Normalization to use for the embeddings. One of 'L1', 'L2', None.

        Returns:
            tf.keras.Model: The siamese model.
        """

        input = layers.Input(shape=(input_len, feature_count), name="input")
        input_norm = layers.BatchNormalization()(input)
        input_padded = layers.ZeroPadding1D(padding=2)(input_norm)

        input_i, input_q = tf.split(input_padded, feature_count, axis=2)

        conv_shapes = []
        conv_i = [input_i]
        conv_q = [input_q]
        for i, (filters, kernel_size) in enumerate(conv_layers):
            conv_shapes.append(conv_i[-1].get_shape().as_list()[1])

            conv_i.append(layers.Conv1D(filters, kernel_size, activation='relu', name="conv_i_{}".format(i))(conv_i[-1]))
            conv_i.append(layers.MaxPooling1D()(conv_i[-1]))

            conv_q.append(layers.Conv1D(filters, kernel_size, activation='relu', name="conv_q_{}".format(i))(conv_q[-1]))
            conv_q.append(layers.MaxPooling1D()(conv_q[-1]))

        conv_out = layers.Concatenate(axis=2, name="conv_out")([conv_i[-1], conv_q[-1]])

        flat = layers.Flatten()(conv_out)
        dense = layers.Dense(dense_size, activation=None, name="dense")(flat)
        norm = None
        if normalization == 'l1':
            norm = layers.Lambda(lambda x: tf.linalg.normalize(x, ord=1, axis=1)[0], name="embedding")(dense)
        elif normalization == 'l2':
            norm = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="embedding")(dense)
        elif normalization is None:
            norm = layers.Identity(name="embedding")(dense)

        # Build the decoder
        decoder_dense = layers.Dense(flat.get_shape().as_list()[1], activation='relu', name="decoder_dense")(norm)
        deconv_in = layers.Reshape(conv_out.get_shape().as_list()[1:])(decoder_dense)
        deconv_i, deconv_q = tf.split(deconv_in, feature_count, axis=2)

        deconv_i = [deconv_i]
        deconv_q = [deconv_q]
        for i, (filters, kernel_size) in enumerate(reversed(conv_layers)):
            upscaling_shape = deconv_i[-1].get_shape().as_list()
            upscaling_shape[1] *= 2

            #deconv_i.append(layers.UpSampling1D()(deconv_i[-1]))
            #deconv_i.append(layers.Conv1DTranspose(filters, kernel_size, activation='relu', name="deconv_i_{}".format(i))(deconv_i[-1]))
            deconv_i.append(UpscalingLayer(upscaling_shape)(deconv_i[-1]))
            deconv_i.append(layers.Conv1D(filters, kernel_size, activation='relu', name="deconv_i_{}".format(i))(deconv_i[-1]))

            #deconv_q.append(layers.UpSampling1D()(deconv_q[-1]))
            #deconv_q.append(layers.Conv1DTranspose(filters, kernel_size, activation='relu', name="deconv_{}".format(i))(deconv_q[-1]))
            deconv_q.append(UpscalingLayer(upscaling_shape)(deconv_q[-1]))
            deconv_q.append(layers.Conv1D(filters, kernel_size, activation='relu', name="deconv_q_{}".format(i))(deconv_q[-1]))

            conv_shape = list(reversed(conv_shapes))[i]
            if deconv_i[-1].get_shape().as_list()[1] != conv_shape:
                deconv_i.append(layers.ZeroPadding1D(padding=(0, conv_shape - deconv_i[-1].get_shape().as_list()[1]))(deconv_i[-1]))
                deconv_q.append(layers.ZeroPadding1D(padding=(0, conv_shape - deconv_q[-1].get_shape().as_list()[1]))(deconv_q[-1]))

        output_i = layers.Conv1DTranspose(1, 5, activation='linear', padding='valid', name="output_i")(deconv_i[-1])
        output_q = layers.Conv1DTranspose(1, 5, activation='linear', padding='valid', name="output_q")(deconv_q[-1])

        output_concatenate = layers.Concatenate(axis=2, name="output_concatenate")([output_i, output_q])
        cropping = (output_concatenate.get_shape().as_list()[1] - input_len) // 2
        output = layers.Cropping1D(cropping=cropping, name="output")(output_concatenate)


        model = models.Model(inputs=[input], outputs=[norm, output], name="model")

        #model.add_loss(tfa.losses.TripletSemiHardLoss(
        #    margin=triplet_margin,
        #    distance_metric=triplet_distance_metric,
        #)(norm), name="embedding_loss")
        mse = tf.keras.losses.MeanSquaredError()(input, output)
        model.add_loss(mse)
        model.add_metric(mse, name="reconstruction_loss")

        losses = {
            "embedding": tfa.losses.TripletSemiHardLoss(
                margin=triplet_margin,
                distance_metric=triplet_distance_metric,
            ),
            # MSE between input and output
            #"output": tf.keras.losses.MeanSquaredError()(input, output),
        }
        loss_weights = {
            "embedding": 1.0,
            #"output": 1.0,
        }

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=losses,
            loss_weights=loss_weights,
            #loss=None,
            metrics=[
            ]
        )

        return model

    def __init__(self, name, input_len, feature_count, conv_layers, dense_size, learning_rate, triplet_margin=1.0, triplet_distance_metric='L2', normalization='L2', accuracy_threshold=0.5, save_dir=None):
        """
        Initialize the siamese model.

        Args:
            name (str): Name of the model.
            input_len (int): Length of the input vector.
            feature_count (int): Number of features in the input vector.
            conv_layers (list): List of tuples (filters, kernel_size) for each convolutional layer.
            dense_size (int): Size of the dense layer.
            learning_rate (float): Learning rate for the optimizer.
            triplet_margin (float): Margin for the triplet loss.
            triplet_distance_metric (str): Distance metric to use for the triplet loss. One of 'L2', 'squared-L2', 'angular', or a callable.
            normalization (str): Normalization to use for the embeddings. One of 'L1', 'L2', None.
            accuracy_threshold (float): Accuracy threshold for the model's SiameseAccuracy metric.
            save_dir (str): Directory to save the model to.
        """

        self.accuracy_threshold = accuracy_threshold

        if feature_count != 2:
            raise ValueError("This model currently only supports 2D input vectors (IQ samples).")

        if normalization is not None:
            normalization = normalization.lower()
            if normalization not in ['l1', 'l2']:
                raise ValueError("Invalid normalization: {}".format(normalization))

        self.model = self.build_model(input_len, feature_count, conv_layers, dense_size, learning_rate, triplet_margin, triplet_distance_metric, normalization)

        super().__init__(self.model, name, save_dir=save_dir)
