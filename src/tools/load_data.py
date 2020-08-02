# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 08:18:47 2020

@author: Jhon
"""

import hdf5storage

def load_data(path): 
    try:
        #train_x = train_x.reshape(train_x.shape[0], -1, 27,  train_x.shape[2], train_x.shape[3])
        #train_x = train_x.transpose([4, 1, 3, 0, 2])
        train_x = hdf5storage.loadmat(path)['XTrain']
        train_x = train_x.reshape(train_x.shape[0], 27, -1,  train_x.shape[2], train_x.shape[3])
        train_x = train_x.transpose([4, 2, 3, 0, 1])
        train_y = hdf5storage.loadmat(path)['YTrain']

    except NotImplementedError:
        ValueError('No possible to read file...')
    
    train_y[:,0] = train_y[:,0]/0.65 
    train_y[:,1] = train_y[:,1]/11.9536171
    train_y[:,2] = train_y[:,2]*20
    train_y[:,3] = train_y[:,3]*25
    
    return train_x, train_y


path = 'C:/Users/Jhon/Documents/Udel/Research/Data/TrainingData.mat'
train_x, train_y = load_data(path)