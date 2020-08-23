# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:47:48 2020

@author: Jhon
"""
# pytorch mnist cnn + lstm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models.cnn_lstm import ConvNetLSTM
from tools.load_data import load_training_data
from tools.save_data import save_model


args = dict(cuda=True,
            no_cuda=False,
            batch_size=33,
            val_batch_size=33,
            epochs=160,
            learning_rate=0.0001,
            seed=42)

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
    loss = []
    model.train()
    for i, (data, target) in enumerate(train_loader):

        if args['cuda']:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss.append(loss.item())
        print('Train epoch [{}/{}], Iter [{}/{}], Loss: {:.4f}'
              .format(current_epoch, num_epochs, i+1, len(train_loader), loss.item()))
    return loss


val_criterion = nn.MSELoss(reduction='none')
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
    model.eval()
    loss = 0
    with torch.no_grad():
        for data, target in val_loader:

            if args['cuda']:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss += val_criterion(output, target)  # sum up batch loss
        loss /= len(val_loader)
        print('\nValidation Loss: {:.4f}, epoch [{}/{}]\n'
              .format(loss.mean(), current_epoch, num_epochs))
    return loss.mean().cpu().numpy()


#Calling function of arguments

args['cuda'] = not args['no_cuda'] and torch.cuda.is_available()

#Let's load the data
PATH = 'C:/Users/Jhon/Documents/Udel/Research/Data/TrainingData.mat'

train_x, train_y = load_training_data(PATH)

train_y[:, 0] = train_y[:, 0]/0.65
train_y[:, 1] = train_y[:, 1]/11.9536171
train_y[:, 2] = train_y[:, 2]*20
train_y[:, 3] = train_y[:, 3]*25

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
optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'])
criterion = nn.MSELoss()

#Running training and validation
train_loss = []
val_loss = []
for epoch in range(1, args['epochs'] + 1):
    train_loss += train(epoch, args['epochs'])
    val_loss.append(validation(epoch, args['epochs']))

save_model(model.state_dict(),
           dict(training_loss=train_loss, validaton_loss=val_loss),
           'model_test')
