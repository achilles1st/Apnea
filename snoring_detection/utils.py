import numpy as np
import csv
import os


def load_dataset(main_path = './',preprocess = 'mfcc'):
    train_y = []
    test_y = []
    val_y  = []
    train_x = []
    test_x = []
    val_x = []
    with open(main_path+'processed_data/training.csv', 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            train_y.append(int(row[0]))
            print( " ******* " )
            print('{}processed_data/{}/{}_{}{}.npy'.format(main_path,row[0],row[0],row[1],preprocess))
            train_x.append(np.load('{}processed_data/{}/{}_{}{}.npy'.format(main_path,row[0],row[0],row[1],preprocess)))
    with open(main_path+'processed_data/testing.csv', 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            test_y.append(int(row[0]))
            test_x.append(np.load('{}processed_data/{}/{}_{}{}.npy'.format(main_path,row[0],row[0],row[1],preprocess)))
    if os.path.exists(main_path+'processed_data/validation.csv'):
        with open(main_path+'processed_data/validation.csv', 'r') as file:
            reader = csv.reader(file, delimiter = ',')
            for row in reader:
                val_y.append(int(row[0]))
                val_x.append(np.load('{}processed_data/{}/{}_{}{}.npy'.format(main_path,row[0],row[0],row[1],preprocess)))
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    train_x = np.array(train_x)
    test_x = np.array(test_x)
    val_x = np.array(val_x)
    val_y = np.array(val_y)

    
    # standard normalization
    x_mean = np.mean(train_x,axis=0)
    x_std = np.std(train_x,axis=0)
    train_x = (train_x-x_mean)/x_std
    test_x = (test_x-x_mean)/x_std

    # also normalize validation dataset 
    if len(val_y)>0:
        if preprocess=='psd': val_x = np.log(val_x)
        val_x = (val_x-x_mean)/x_std
    return train_x, test_x, val_x, train_y, test_y, val_y, x_mean, x_std

