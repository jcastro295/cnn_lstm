# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 08:47:17 2020

@author: Jhon
"""

import os
import os.path as osp
import torch
import hdf5storage

def save_model(model_dict, train_loss, val_loss, filename="default_ouput"):
    """
    definition
    """
    if not osp.exists('output'):
        os.makedirs('output')

    hdf5storage.savemat('output/' + filename,
                        mdict={'training_loss' : train_loss, 'validation_loss' : val_loss})
    torch.save(model_dict, 'output/' + filename + '.pt')

    print('File has been saved succesfully')


def save_predictions(predictions, filename="default_prediction"):
    """
    Definition
    """
    if not osp.exists('tests'):
        os.makedirs('tests')

    hdf5storage.savemat('tests/' + filename, mdict={'predictions' : predictions})

    print('File has been saved succesfully')
