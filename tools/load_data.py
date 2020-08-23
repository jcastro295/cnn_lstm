# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 08:18:47 2020

@author: Jhon
"""

import hdf5storage

def load_training_data(path):
    """
    Load training data
    This function allow to import hdf5 files from matlab for training

    Parameters:
    -----------
    path: String
        Path where the training data is located.

    Returns:
    --------
    data: array
        Array with spectrograms for training
    labels: array
        Array with labels for training
    """

    try:
        data = hdf5storage.loadmat(path)['XTrain']
        labels = hdf5storage.loadmat(path)['YTrain']
    except NotImplementedError:
        ValueError('No possible to read file...')

    return data, labels


def load_testing_data(path):
    """
    Load testing data
    This function allow to import hdf5 files from matlab for testing

    Parameters:
    -----------
    path: String
        Path where the testing data is located.

    Returns:
    --------
    data: array
        Array with spectrograms for testing
    """

    try:
        data = hdf5storage.loadmat(path)['XTest']
    except NotImplementedError:
        ValueError('No possible to read file...')

    return data
