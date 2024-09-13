import tensorflow as tf
import sys
sys.path.append('./')
from utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
            tf.keras.layers.Dense(2)
        ])
        model.summary()
        return model

    def compile_model(self):
        # Compile the model with the optimizer, loss function, and metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def train_model(self, train_x, train_y_one_hot, val_x, val_y_one_hot):
        # Train the model with the provided dataset
        history = self.model.fit(
            train_x,
            train_y_one_hot,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
            validation_data=(val_x, val_y_one_hot),
            callbacks=[self.checkpoint]
        )
        return history

    def evaluate_model(self, test_x, test_y):
        # Make predictions and evaluate the model
        predictions = self.model.predict(test_x)
        return predictions

    @staticmethod
    def index_of_max(output_list):
        # Helper function to get indices of max values in a list of lists
        return [np.argmax(sub_list) for sub_list in output_list]

    @staticmethod
    def plot_confusion_matrix(test_y, predictions):
        # Plot a confusion matrix for model predictions
        confusion = tf.math.confusion_matrix(
            labels=tf.constant(test_y.flatten()),
            predictions=tf.constant(CNNModelTrainer.index_of_max(predictions)),
            num_classes=2
        )
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion, xticklabels=['0', '1'], yticklabels=['0', '1'],
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()

def main():
    # Load dataset
    train_x, test_x, val_x, train_y, test_y, val_y, x_mean, x_std = load_dataset()
    N = train_x.shape[0]

    # Prepare labels in one-hot encoding format
    train_y_one_hot = tf.one_hot(train_y, depth=2)
    val_y_one_hot = tf.one_hot(val_y, depth=2)

    # Initialize and train the model
    trainer = CNNModelTrainer(learning_rate=0.001, batch_size=N, epochs=500)
    trainer.compile_model()
    history = trainer.train_model(train_x, train_y_one_hot, val_x, val_y_one_hot)

    # Evaluate the model
    predictions = trainer.evaluate_model(test_x, test_y)

    # Print training results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    print('Training accuracy: ', acc[-1])
    print('Validation accuracy: ', val_acc[-1])

    # Plot confusion matrix
    trainer.plot_confusion_matrix(test_y, predictions)

if __name__ == "__main__":
    main()
