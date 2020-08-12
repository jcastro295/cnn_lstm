# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 08:43:52 2020

@author: Jhon
"""
import torch.nn as nn

#Convolutional Neural Network
class ConvNet(nn.Module):
    """
    definition
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 30), stride=(1, 5), padding=(1, 5)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=(3, 5), stride=(1, 4), padding=(1, 2)))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 9), stride=(1, 3), padding=(1, 4)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2)),
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
class ConvNetLSTM(nn.Module):
    """
    Description
    """
    def __init__(self):
        super(ConvNetLSTM, self).__init__()
        self.conv_net = ConvNet()
        self.rec_net = nn.LSTM(input_size=5120,
                               hidden_size=400,
                               num_layers=1,
                               batch_first=True)
        self.linear = nn.Linear(400, 4)

    def forward(self, x):
        batch_size, timesteps, channels, height, width = x.size()
        c_in = x.view(batch_size * timesteps, channels, height, width)
        c_out = self.conv_net(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (_, _) = self.rec_net(r_in)
        r_out2 = self.linear(r_out[:, -1, :])

        return r_out2
