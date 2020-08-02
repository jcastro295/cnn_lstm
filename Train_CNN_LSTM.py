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
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import scipy.io as sio

#Defining classes 
class Args:   #Arguments needed 
    def __init__(self):
        self.cuda = True
        self.no_cuda = False
        self.batch_size = 33
        self.val_batch_size = 33
        self.epochs = 160
        self.lr = 0.0001
        self.seed = 42

#Convolutional Neural Network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,10), stride=(1,5), padding=(1,5)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=(3,5), stride=(1,4), padding=(1,2)))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,9), stride=(1,3), padding=(1,4)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=(3,5), stride=(1,2), padding=(1,2)))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,9), stride=(1,2),padding=(1,4)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3,5), stride=(1,2),padding=(1,2)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3,5), stride=(1,1), padding=(1,2)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True))
        self.drop_out = nn.Dropout()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # End of convolution stage
        out = out.reshape(out.size(0), -1) # Reshape the output of the
                                        # Convolution Stage into 1 vector
        out = self.drop_out(out)
        return out


#Adding Long Short Term Memory    
class ConvNet_LSTM(nn.Module):
    def __init__(self):
        super(ConvNet_LSTM, self).__init__()
        self.CNN = ConvNet()
        self.RNN = nn.LSTM(
            input_size = 5120, 
            hidden_size = 400,
            num_layers=1,
            batch_first=True)
        self.linear = nn.Linear(400,4)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.CNN(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n,h_c) = self.RNN(r_in)
        r_out2 = self.linear(r_out[:,-1,:])
        
        return r_out2
    
#Training
def train(epoch, num_epochs):
    train_loss = []
    model.train()
    for i, (data, target) in enumerate(train_loader):
        
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        print('Train epoch [{}/{}], Iter [{}/{}], Loss: {:.4f}'
              .format(epoch, num_epochs, i+1, len(train_loader), loss.item()))
    return train_loss


#Validation  
val_criterion = nn.MSELoss(reduction = 'none')          
def validation(epoch, num_epochs):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            val_loss += val_criterion(output, target)  # sum up batch loss
        val_loss /= len(val_loader)
        print('\nValidation Loss: {:.4f}, epoch [{}/{}]\n'
              .format(val_loss.mean(), epoch, num_epochs))
    return val_loss.mean().cpu().numpy()

if __name__ == "__main__":
    
    #Calling function of arguments
    args = Args()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    #Let's load the data
    path_data = 'C:/Users/Jhon/Documents/Udel/Research/Data/TrainingData.mat'
    
    
    try:
        f = sio.loadmat(path_data) 
        print('Using scipy to open training data')
        train_x = f.get('XTrain')[()]
        train_y = f.get('YTrain')[()]
        #Reshape data in 5 dim
        train_x = train_x.reshape(train_x.shape[0], -1, 27,  train_x.shape[2], train_x.shape[3])
        train_x = train_x.transpose([4, 1, 3, 0, 2])
    except NotImplementedError:
        f = h5py.File(path_data, 'r')
        print('Using h5py to open training data')
        train_x = f.get('XTrain')[()]
        print(train_x.shape)
        train_y = np.transpose(f.get('YTrain')[()])
        #Reshape data in 5 dim
        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1, 27, train_x.shape[3])
        print(train_x.shape)
        train_x = train_x.transpose([0, 2, 1, 4, 3])
        print(train_x.shape)
    except:
        ValueError('No possible to read file...')
    
    train_y[:,0] = train_y[:,0]/0.65 
    train_y[:,1] = train_y[:,1]/11.9536171
    train_y[:,2] = train_y[:,2]*20
    train_y[:,3] = train_y[:,3]*25
    
    #Creating validation set
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.05, random_state = args.seed)
    
    #Turning data into tensors
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    
    val_x = torch.from_numpy(val_x)
    val_y = torch.from_numpy(val_y)
    
    loader = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = DataLoader(dataset=loader, batch_size=args.batch_size, shuffle=True)
    loader = torch.utils.data.TensorDataset(val_x, val_y)
    val_loader = DataLoader(dataset=loader, batch_size=args.val_batch_size, shuffle=False)
      
    #Model 
    model = ConvNet_LSTM()
    
    if args.cuda:
        model.cuda()
    
    #Optimizer and training criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    #Running training and validation
    train_loss = []
    val_loss = []
    for epoch in range(1, args.epochs + 1):
        train_loss += train(epoch, args.epochs)
        val_loss.append(validation(epoch, args.epochs))
    sio.savemat('NEW_CNN_LSTM_run_1.mat', mdict={'train_loss': train_loss, 'val_loss': val_loss})
        
    PATH='NEW_CNN_LSTM_run_1.pt'
    torch.save(model.state_dict(), PATH)