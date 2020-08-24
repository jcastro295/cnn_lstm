# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:47:48 2020

@author: Jhon
"""
# pytorch mnist cnn + lstm

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models.cnn_lstm import ConvNetLSTM
from tools.load_data import load_training_data
from tools.save_data import save_model
from tools.useful_functions import set_optimizer
from tools.useful_functions import set_criterion
from tools.useful_functions import set_learning_rate_scheduler
from tools.early_stopping import EarlyStopping

args = dict(cuda=True,
            no_cuda=False,
            batch_size=33,
            val_batch_size=33,
            epochs=1,
            learning_rate=0.0001,
            seed=42,
            patience=20,
            verbose=True)

def train(current_epoch, num_epochs):
    """
    Training function for ResNet

    Function for doing the training of the network

    Parameters:
    -----------
    epoc: integer
        Current epoc for training
    num_epocs: integer
        Total number of epocs for training

    Returns:
    --------
    loss: list
        List with loss obtained in epoc
    """
    training_loss = []
    model.train()
    for i, (data, target) in enumerate(train_loader):

        if args['cuda']:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        training_loss.append(loss.item())
        print('Train epoch [{}/{}], Iter [{}/{}], Loss: {:.4f}'
              .format(current_epoch, num_epochs, i+1, len(train_loader), loss.item()))
    return training_loss


def validation(current_epoch, num_epochs):
    """
    Validation function for ResNet

    Function for doing the training of the network

    Parameters:
    -----------
    epoc: integer
        Current epoc for training
    num_epocs: integer
        Total number of epocs for training

    Returns:
    --------
    loss: list
        List with loss obtained in epoc
    """

    #let's play with this
    valid_losses = []
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:

            if args['cuda']:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(output, target)
            valid_losses.append(loss.item())
        validation_loss = np.average(valid_losses)
        print('\nValidation Loss: {:.4f}, epoch [{}/{}]\n'
              .format(validation_loss, current_epoch, num_epochs))
    return validation_loss


#Calling function of arguments

args['cuda'] = not args['no_cuda'] and torch.cuda.is_available()

#Let's load the data
PATH = 'D:/Desktop/Networks/TrainingData.mat'

train_x, train_y = load_training_data(PATH)

train_y[:, 0] = train_y[:, 0]/0.65
train_y[:, 1] = train_y[:, 1]/11.9536171
train_y[:, 2] = train_y[:, 2]*20
train_y[:, 3] = train_y[:, 3]*25

#80 1107 1 6600  -> (6600, 41, 1, 80, 27)
#changing dimensions x train
train_x = train_x.reshape(train_x.shape[0], -1, 27, train_x.shape[2], train_x.shape[3])
train_x = train_x.transpose([4, 1, 3, 0, 2])

#Creating validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y,
                                                  test_size=0.05, random_state=args['seed'])

#Turning data into tensors
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)

val_x = torch.from_numpy(val_x)
val_y = torch.from_numpy(val_y)

loader = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = DataLoader(dataset=loader, batch_size=args['batch_size'], shuffle=True)
loader = torch.utils.data.TensorDataset(val_x, val_y)
val_loader = DataLoader(dataset=loader, batch_size=args['val_batch_size'], shuffle=False)

#Model
model = ConvNetLSTM()

if args['cuda']:
    model.cuda()

#Optimizer and training criterion
optimizer = set_optimizer('adam')(model.parameters(), lr=args['learning_rate'])
criterion = set_criterion('mse')

#Running training and validation
train_loss = []
val_loss = []

#initialize the early_stopping object
early_stopping = EarlyStopping(patience=args['patience'], verbose=args['verbose'])
scheduler = set_learning_rate_scheduler('cosine', optimizer, T_max=args['epochs']/5, eta_min=0)

for epoch in range(1, args['epochs'] + 1):
    train_loss += train(epoch, args['epochs'])
    val_losss = validation(epoch, args['epochs'])
    val_loss.append(val_losss)
    scheduler.step()
    early_stopping(val_losss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

save_model(model.state_dict(), train_loss, val_loss, filename='model_test')
