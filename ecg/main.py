"""
The code structures and ideas were taken from the GitHub repository (with small modifications):
https://github.com/JackAndCole/Detection-of-sleep-apnea-from-single-lead-ECG-signal-using-a-time-window-artificial-neural-network/blob/master/main.py
"""

import os
import pickle

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

np.random.seed(0)

base_dir = "dataset"

with open(os.path.join(base_dir, "apnea-ecg.pkl"), "rb") as f:
    apnea_ecg = pickle.load(f)

print(apnea_ecg.keys())


def plot_ecg_recording(ecg_signal, sampling_rate=128, title="ECG Recording", save_path=None):
    """
    Plots the ECG recording and optionally saves the plot to a file.

    Parameters:
    - ecg_signal (list or numpy array): The ECG signal data to be plotted.
    - sampling_rate (int): The sampling rate of the ECG signal in Hz. Default is 100 Hz.
    - title (str): The title of the plot.
    - save_path (str): The path to save the plot. If None, the plot will not be saved.
    """
    # Calculate the time axis based on the sampling rate
    time_axis = np.linspace(0, len(ecg_signal) / sampling_rate, len(ecg_signal))

    # Create the plot
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, ecg_signal, color='blue')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (mV)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# Plot and save all the ECG recordings
# for name, ecg_signals in apnea_ecg.items():
#     if name == "a01":
#         continue
#     num = 0
#     for row, ecg_signal in enumerate(ecg_signals):
#     #for ecg_signal in ecg_signals.T:
#         save_path = os.path.join("C:\\Users\\tosic\\Arduino_projects\\sensor_com\\ecg\\plots\\" f"{name}_{num}.png")
#         plot_ecg_recording(ecg_signal, sampling_rate=100, title=f"ECG Recording - {name}", save_path=save_path)
#         num += 1
#     break

def shift(xs, n):
    e = np.empty_like(xs)
    if n > 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    elif n < 0:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    else:
        e[:] = xs[:]
    return e


def acquisition_features(recordings, time_window_size):
    features = []
    labels = []
    groups = []
    num = 0
    for recording in recordings:
        data = apnea_ecg[recording]
        temp = []
        for w in range(time_window_size + 1):
            temp.append(shift(data[:, :-1], w))
        temp = np.concatenate(temp, axis=1)
        mask = ~np.isnan(temp).any(axis=1)
        features.append(temp[mask])
        labels.append(data[mask, -1])
        groups.append([recording] * sum(mask))

        # for row, ecg_signal in enumerate(temp):
        #     save_path = os.path.join("C:\\Users\\tosic\\Arduino_projects\\sensor_com\\ecg\\plots\\" f"{num}.png")
        #     plot_ecg_recording(ecg_signal, sampling_rate=100, title=f"ECG Recording - {num}", save_path=save_path)
        #     num += 1

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    groups = np.concatenate(groups, axis=0)
    return features, labels, groups


x_train, y_train, groups_train = acquisition_features(list(apnea_ecg.keys())[:35], 5)
x_test, y_test, groups_test = acquisition_features(list(apnea_ecg.keys())[35:], 5)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
#
# # Fit and transform the x_train data using t-SNE
# tsne = TSNE(n_components=2, random_state=0)
# x_train_tsne = tsne.fit_transform(x_train)
#
# # Plot the t-SNE results
# plt.figure(figsize=(10, 6))
# plt.scatter(x_train_tsne[:, 0], x_train_tsne[:, 1], c=y_train, cmap='viridis', s=5)
# #plt.colorbar(label='Label')
# plt.title('t-SNE plot of x_train data')
# plt.xlabel('t-SNE component 1')
# plt.ylabel('t-SNE component 2')
# plt.grid(True)
# plt.show()


# Save the scaler to a file
scaler_filename = 'scaler_base.pkl'
joblib.dump(scaler, scaler_filename)

clf = MLPClassifier(hidden_layer_sizes=(x_train.shape[1] * 2 + 1,), alpha=1, max_iter=1000)
clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))

y_pred = clf.predict(x_test)
C = confusion_matrix(y_test, y_pred, labels=(1, 0))
TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
print("acc: {}, sn: {}, sp: {}".format(acc, sn, sp))

# Save the model to a file
model_filename = 'mlp_classifier_model_base.pkl'
joblib.dump(clf, model_filename)