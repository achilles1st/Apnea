import tensorflow as tf
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from train_model import CNNModelTrainer


class CNNHyperModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=hp.Choice('conv_1_filter', values=[32, 64], default=32),
                kernel_size=hp.Choice('conv_1_kernel', values=[3, 5], default=3),
                activation='relu',
                input_shape=self.input_shape
            ),
            tf.keras.layers.Conv2D(
                filters=hp.Choice('conv_2_filter', values=[32, 64], default=32),
                kernel_size=hp.Choice('conv_2_kernel', values=[3, 5], default=3),
                activation='relu'
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=hp.Int('dense_units', min_value=32, max_value=512, step=32),
                activation='relu'
            ),
            tf.keras.layers.Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model


def run_hyperparameter_search():
    # Load dataset
    train_x, test_x, val_x, train_y, test_y, val_y, x_mean, x_std = load_dataset()
    train_x = np.expand_dims(train_x, -1)
    test_x = np.expand_dims(test_x, -1)
    val_x = np.expand_dims(val_x, -1)

    # Define the input shape
    input_shape = train_x.shape[1:]

    # Initialize the hypermodel
    hypermodel = CNNHyperModel(input_shape)

    # Initialize the tuner
    tuner = RandomSearch(
        hypermodel.build,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='hyperparam_tuning',
        project_name='cnn_tuning'
    )

    # Display search space summary
    tuner.search_space_summary()

    # Perform hyperparameter search
    tuner.search(
        train_x, train_y,
        epochs=50,
        validation_data=(val_x, val_y),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
    )

    # Retrieve the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Evaluate the best model
    test_loss, test_acc = best_model.evaluate(test_x, test_y)
    print(f'Best model test accuracy: {test_acc}')

    # Display the search results summary
    tuner.results_summary()

    # Plot confusion matrix using the best model predictions
    predictions = best_model.predict(test_x)
    CNNModelTrainer.plot_confusion_matrix(test_y, predictions)


if __name__ == "__main__":
    run_hyperparameter_search()
