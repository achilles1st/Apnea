import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from IPython import display
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras import regularizers
from sklearn.model_selection import train_test_split


class SnoringDetector:
    def __init__(self, dataset_path, batch_size=150, validation_split=0.2, seed=0, output_sequence_length=44100, sample_rate=44100):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.seed = seed
        self.output_sequence_length = output_sequence_length
        self.label_names = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.model = None
        self.norm_layer = None
        self.history = None  # Store training history
        self.test_log_mel_ds = None  # Ensure test dataset is accessible
        self.sample_rate = sample_rate

    def load_data(self):
        data_dir = pathlib.Path(self.dataset_path)
        dataset = keras.utils.audio_dataset_from_directory(
            directory=data_dir,
            batch_size=self.batch_size,
            output_sequence_length=self.output_sequence_length
        )

        self.label_names = np.array(dataset.class_names)
        print("\nLabel names:", self.label_names)

        # Convert dataset to numpy arrays
        audio_data = []
        labels = []
        for audio, label in dataset:
            audio_data.extend(audio.numpy())
            labels.extend(label.numpy())

        audio_data = np.array(audio_data)
        labels = np.array(labels)

        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(audio_data, labels, test_size=self.validation_split, random_state=self.seed, stratify=labels)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=self.seed, stratify=y_temp)

        # Convert back to tf.data.Dataset
        self.train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(self.batch_size)
        self.val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(self.batch_size)
        self.test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(self.batch_size)

        # Squeeze the audio data
        self.train_ds = self.train_ds.map(self._squeeze, tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.map(self._squeeze, tf.data.AUTOTUNE)
        self.test_ds = self.test_ds.map(self._squeeze, tf.data.AUTOTUNE)

    @staticmethod
    def _squeeze(audio, labels):
        audio = tf.squeeze(audio, axis=-1)
        return audio, labels
    @staticmethod
    def _squeeze(audio, labels):
        audio = tf.squeeze(audio, axis=-1)
        return audio, labels

    def preprocess_data(self):
        # Create log-mel spectrogram datasets
        self.train_log_mel_ds = self._make_log_mel_ds(self.train_ds)
        self.val_log_mel_ds = self._make_log_mel_ds(self.val_ds)
        self.test_log_mel_ds = self._make_log_mel_ds(self.test_ds)

        self.train_log_mel_ds = self.train_log_mel_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
        self.val_log_mel_ds = self.val_log_mel_ds.cache().prefetch(tf.data.AUTOTUNE)
        self.test_log_mel_ds = self.test_log_mel_ds.cache().prefetch(tf.data.AUTOTUNE)

        self.train_log_mel_not_flat_ds = self._make_log_mel_not_flat_ds(self.train_ds)


    def _get_log_mel(self, waveform):
        frame_length = int(0.032 * self.sample_rate)  # Approximately 32ms
        frame_step = int(0.016 * self.sample_rate)  # Approximately 16ms
        stfts = tf.signal.stft(waveform, frame_length=frame_length, frame_step=frame_step)

        # Obtain the magnitude of the STFT.
        spectrograms = tf.abs(stfts)
        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 40.0, 6000.0, 30
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, self.sample_rate, lower_edge_hertz,
            upper_edge_hertz)
        mel_spectrograms = tf.tensordot(
            spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        log_mel_spectrograms = tf.reshape(log_mel_spectrograms, [-1, 1830])
        return log_mel_spectrograms

    def _get_log_mel_not_flat(self, waveform):
        frame_length = int(0.032 * self.sample_rate)
        frame_step = int(0.016 * self.sample_rate)
        stfts = tf.signal.stft(waveform, frame_length=frame_length, frame_step=frame_step)
        # Obtain the magnitude of the STFT.
        spectrograms = tf.abs(stfts)
        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 40.0, 6000.0, 30
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, self.sample_rate, lower_edge_hertz,
            upper_edge_hertz)
        mel_spectrograms = tf.tensordot(
            spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        log_mel_spectrograms = log_mel_spectrograms[..., tf.newaxis]
        return log_mel_spectrograms

    def _make_log_mel_ds(self, ds):
        return ds.map(
            map_func=lambda audio, label: (self._get_log_mel(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE)

    def _make_log_mel_not_flat_ds(self, ds):
        return ds.map(
            map_func=lambda audio, label: (self._get_log_mel_not_flat(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE)

    def build_model(self, l1_value=0.0001):
        for example_log_mels, example_lm_labels in self.train_log_mel_ds.take(1):
            break
        input_shape = example_log_mels.shape[1:]

        # Instantiate the `tf.keras.layers.Normalization` layer.
        self.norm_layer = layers.Normalization()
        # Fit the state of the layer to the spectrograms
        # with `Normalization.adapt`.
        self.norm_layer.adapt(data=self.train_log_mel_not_flat_ds.map(map_func=lambda spec, label: spec))

        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Reshape((61, 30, 1)),
            # Downsample the input.
            layers.Resizing(32, 30),
            # Normalize.
            self.norm_layer,
            layers.Conv2D(32, (3, 3), padding='same', activation="relu",
                          kernel_regularizer=regularizers.l1(l1_value)),
            layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=regularizers.l1(l1_value)),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), padding='same', activation="relu",
                          kernel_regularizer=regularizers.l1(l1_value)),
            layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers.l1(l1_value)),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l1(l1_value)),
            layers.Dropout(0.5),
            layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l1(l1_value)),
            layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l1(l1_value))
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy'],
        )

        self.model.summary()

    def train_model(self, epochs=100, patience=10):
        checkpoint = keras.callbacks.ModelCheckpoint(
            'models/cnn_new_big_data_deviation_44k.keras',
            verbose=1,
            monitor='val_loss',
            save_best_only=True,
            mode='auto'
        )

        self.history = self.model.fit(
            self.train_log_mel_ds,
            validation_data=self.val_log_mel_ds,
            epochs=epochs,
            callbacks=[keras.callbacks.EarlyStopping(verbose=1, patience=patience), checkpoint],
        )

        return self.history

    def evaluate_model(self):
        test_loss, test_acc = self.model.evaluate(self.test_log_mel_ds)
        print(f"Test accuracy: {test_acc:.2f}")

    def plot_training_history(self):
        if self.history is None:
            print("No training history available. Train the model first.")
            return

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.show()

    def plot_confusion_matrix(self):
        # Get true labels and predictions
        y_true = []
        y_pred = []
        for x, y in self.test_log_mel_ds:
            preds = self.model.predict(x)
            y_true.extend(y.numpy())
            y_pred.extend(preds.squeeze())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_labels = (y_pred > 0.5).astype(int)

        cm = confusion_matrix(y_true, y_pred_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.label_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

    def plot_sample_waveforms(self):
        # Plot sample waveforms
        for example_audio, example_labels in self.train_ds.take(1):
            break

        plt.figure(figsize=(16, 10))
        rows, cols = 3, 3
        n = rows * cols
        for i in range(n):
            plt.subplot(rows, cols, i + 1)
            audio_signal = example_audio[i]
            plt.plot(audio_signal)
            plt.title(self.label_names[example_labels[i]])
            plt.yticks(np.arange(-1.2, 1.2, 0.2))
            plt.ylim([-1.1, 1.1])
        plt.show()

    def plot_log_mel_spectrogram(self, waveform, label):
        log_mel = self._get_log_mel(waveform)

        print('Label:', label)
        print('Waveform shape:', waveform.shape)
        print('Log-mel spectrogram shape:', log_mel.shape)
        plt.imshow(tf.squeeze(log_mel).numpy().T, aspect='auto', origin='lower')
        plt.title('Log-Mel Spectrogram')
        plt.show()

        display.display(display.Audio(waveform, rate=self.sample_rate))


if __name__ == "__main__":
    detector = SnoringDetector('C:/Users/tosic/Arduino_projects/sensor_com/snoring_detection/Snoring_Dataset_@16000/44100_additions/44_clean')
    detector.load_data()
    detector.preprocess_data()
    detector.build_model()
    history = detector.train_model(epochs=64, patience=12)
    # plots
    detector.plot_training_history()
    detector.evaluate_model()
    detector.plot_confusion_matrix()
    #detector.plot_sample_waveforms()

