# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 08:47:17 2020

@author: Jhon
"""

import os
import os.path as osp
import torch
import hdf5storage

def save_model(model_dict, loss_dict, filename="default_ouput"):
    """
    definition
    """
    if not osp.exists('../output'):
        os.makedirs('../output')

    hdf5storage.savemat('...output/' + filename, mdict=loss_dict)
    torch.save(model_dict(), '../output/' + filename + '.pt')

    print('File has been saved succesfully')


def save_predictions(prediction_dict, filename="default_prediction"):
    """
    Definition
    """
    if not osp.exists('../output'):
        os.makedirs('../output')

    hdf5storage.savemat('../output' + filename, mdict=prediction_dict)

    print('File has been saved succesfully')
