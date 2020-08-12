# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 08:47:17 2020

@author: Jhon
"""
import torch 
import scipy.io as sio
import os
import os.path as osp
from pathlib import Path

def save_model(dictionary, train, val, filename):
    path = osp.abspath(osp.join('..', os.getcwd()))
    path = Path(path).parents[1]
    path = osp.join(path, 'output')
    error = False
    try:
        int(filename)
    except ValueError:
        error = True
    if not error:
        print('File name is not String, saving with a default name "CNN_LSTM"...\n')
        filename = 'CNN_LSTM'
    sio.savemat(osp.join(path ,filename +'.mat'), mdict={'train_loss': train, 'val_loss': val})
    torch.save(dictionary, osp.join(path , filename + '.ckpt'))
    print('Files .mat and .ckpt saved succesfully in output directory')