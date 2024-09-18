import tensorflow as tf
import sys

sys.path.append('./')
from utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold


class CNNModelTrainer:
    def __init__(self, learning_rate=0.001, batch_size=None, epochs=500):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = self._build_model()
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'models/cnn.keras',
            verbose=1,
            monitor='val_loss',
            save_best_only=True,
            mode='auto'
        )
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=12,
            verbose=1,
            restore_best_weights=True
        )

    def _build_model(self):
        # Define the CNN model structure
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation="relu", input_shape=(32, 32, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation="relu"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        model.summary()
        return model

    def compile_model(self):
        # Compile the model with the optimizer, loss function, and metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

    def train_model(self, train_x, train_y, val_x, val_y):
        # Train the model with the provided dataset
        history = self.model.fit(
            train_x,
            train_y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
            validation_data=(val_x, val_y),
            callbacks=[self.checkpoint, self.early_stopping]
        )
        return history

    def evaluate_model(self, test_x):
        # Make predictions and evaluate the model
        predictions = self.model.predict(test_x)
        return predictions

    @staticmethod
    def plot_confusion_matrix(test_y, predictions):
        # Plot a confusion matrix for model predictions
        rounded_predictions = np.round(predictions).astype(int)
        confusion = tf.math.confusion_matrix(
            labels=tf.constant(test_y.flatten()),
            predictions=tf.constant(rounded_predictions.flatten()),
            num_classes=2
        )
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion, xticklabels=['0', '1'], yticklabels=['0', '1'],
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()

    @staticmethod
    def plot_metrics(history):
        # Plot the training and validation loss and accuracy during training
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        epochs = range(len(loss))

        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, 'b', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()


def main():
    # Load dataset
    train_x, test_x, val_x, train_y, test_y, val_y, x_mean, x_std = load_dataset()
    # N = train_x.shape[0]
    N = 500

    # Combine train and validation sets for cross-validation
    x_data = np.concatenate((train_x, val_x), axis=0)
    y_data = np.concatenate((train_y, val_y), axis=0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_no = 1
    for train_index, val_index in skf.split(x_data, y_data):
        print(f'Fold {fold_no}')

        # Split data
        train_x_fold, val_x_fold = x_data[train_index], x_data[val_index]
        train_y_fold, val_y_fold = y_data[train_index], y_data[val_index]

        # Initialize and train the model
        trainer = CNNModelTrainer(learning_rate=0.001, batch_size=N, epochs=200)
        trainer.compile_model()
        history = trainer.train_model(train_x_fold, train_y_fold, val_x_fold, val_y_fold)

        # Evaluate the model
        predictions = trainer.evaluate_model(test_x)

        # Print training results
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        print(f'Training accuracy for fold {fold_no}: ', acc[-1])
        print(f'Validation accuracy for fold {fold_no}: ', val_acc[-1])

        # Plot confusion matrix
        trainer.plot_confusion_matrix(test_y, predictions)

        # Plot loss and accuracy functions
        CNNModelTrainer.plot_metrics(history)

        fold_no += 1


if __name__ == "__main__":
    #main()

    # Load dataset
    N = [400]  # 32 is best
    for n in N:
        train_x, test_x, val_x, train_y, test_y, val_y, x_mean, x_std = load_dataset()
        # N = train_x.shape[0]
        # N = 32
        # Initiali+ze and train the model
        trainer = CNNModelTrainer(learning_rate=0.0001, batch_size=n, epochs=200)
        trainer.compile_model()
        history = trainer.train_model(train_x, train_y, val_x, val_y)

        # Evaluate the model
        predictions = trainer.evaluate_model(test_x)

        # Print training results
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        print('Training accuracy: ', acc[-1])
        print('Validation accuracy: ', val_acc[-1])

        # Plot confusion matrix
        trainer.plot_confusion_matrix(test_y, predictions)
        # Plot loss functions
        CNNModelTrainer.plot_metrics(history)