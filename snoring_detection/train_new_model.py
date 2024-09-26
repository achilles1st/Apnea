import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from IPython import display
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class SnoringDetector:
    def __init__(self, dataset_path, batch_size=150, validation_split=0.2, seed=0, output_sequence_length=16000):
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

    def load_data(self):
        data_dir = pathlib.Path(self.dataset_path)
        self.train_ds, self.val_ds = keras.utils.audio_dataset_from_directory(
            directory=data_dir,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            seed=self.seed,
            output_sequence_length=self.output_sequence_length,
            subset='both'
        )

        self.label_names = np.array(self.train_ds.class_names)
        print("\nLabel names:", self.label_names)

        # Squeeze the audio data
        self.train_ds = self.train_ds.map(self._squeeze, tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.map(self._squeeze, tf.data.AUTOTUNE)

        # Split validation dataset into validation and test datasets
        self.test_ds = self.val_ds.shard(num_shards=2, index=0)
        self.val_ds = self.val_ds.shard(num_shards=2, index=1)

    @staticmethod
    def _squeeze(audio, labels):
        audio = tf.squeeze(audio, axis=-1)
        return audio, labels

    def preprocess_data(self):
        # Create log-mel spectrogram datasets
        self.train_log_mel_ds = self._make_log_mel_ds(self.train_ds)
        self.val_log_mel_ds = self._make_log_mel_ds(self.val_ds)
        self.test_log_mel_ds = self._make_log_mel_ds(self.test_ds)

        # Normalize the data
        self.norm_layer = layers.Normalization()
        self.norm_layer.adapt(data=self.train_log_mel_ds.map(lambda spec, label: spec))

    @staticmethod
    def _get_log_mel(waveform):
        stfts = tf.signal.stft(
            waveform, frame_length=512, frame_step=256)
        spectrograms = tf.abs(stfts)
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 40.0, 6000.0, 30
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, 16000, lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(
            spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))

        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        log_mel_spectrograms = log_mel_spectrograms[..., tf.newaxis]  # Add channel dimension
        return log_mel_spectrograms

    def _make_log_mel_ds(self, ds):
        return ds.map(
            map_func=lambda audio, label: (self._get_log_mel(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    def build_model(self):
        # Get input shape
        for example_log_mels, _ in self.train_log_mel_ds.take(1):
            input_shape = example_log_mels.shape[1:]
            break

        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            self.norm_layer,
            layers.Conv2D(32, 3, activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid'),
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy'],
        )

        self.model.summary()

    def train_model(self, epochs=100, patience=10):
        checkpoint = keras.callbacks.ModelCheckpoint(
            'models/cnn_new.keras',
            verbose=1,
            monitor='val_loss',
            save_best_only=True,
            mode='auto'
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=1,
            restore_best_weights=True
        )

        self.history = self.model.fit(
            self.train_log_mel_ds,
            validation_data=self.val_log_mel_ds,
            epochs=epochs,
            callbacks=[early_stopping, checkpoint],
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

        display.display(display.Audio(waveform, rate=44100))


if __name__ == "__main__":
    detector = SnoringDetector(dataset_path='C:/Users/tosic/Arduino_projects/sensor_com/snoring_detection/Snoring_Dataset_@16000/44100_additions',
                               output_sequence_length=44100
                               )
    detector.load_data()
    detector.preprocess_data()
    detector.build_model()
    history = detector.train_model()
    detector.plot_training_history()
    detector.evaluate_model()
    detector.plot_confusion_matrix()
    detector.plot_sample_waveforms()

