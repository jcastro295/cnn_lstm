# -*- coding: utf-8 -*-
"""
Created on Sun May 10 20:00:59 2020

@author: Jhon
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from train import args
from tools.load_data import load_testing_data
from tools.save_data import save_predictions


# Load the trained parameters
PATH = 'model_test.pt' # Model path
model = torch.load('output/' + PATH)

OUTPUT_FILENAME = 'CNN_LSTM_predictions_'

# Testing data suffixes
strs = ["" for x in range(5)]
strs[0] = 'A'
strs[1] = 'B'
strs[2] = 'C'
strs[3] = 'D'
strs[4] = 'E'

#Load Test data (matlab files)
# the data path is assigned to the variable full_path
for test_num in range(5):
    path = 'C:/Users/Jhon/Documents/Udel/Research/Data/Testing/Test'\
             + strs[test_num] + '.mat'

    test_x = load_testing_data(path)
    test_x = test_x.reshape(test_x.shape[0], -1, 27, test_x.shape[2], test_x.shape[3])
    test_x = test_x.transpose([4, 1, 3, 0, 2])

    test_x = torch.from_numpy(test_x)
    loader = torch.utils.data.TensorDataset(test_x)
    test_loader = DataLoader(dataset=loader, batch_size=args['batch_size'], shuffle=False)

    args['cuda'] = not args['no_cuda'] and torch.cuda.is_available()
    if args['cuda']:
        model.cuda()

    model.eval()
    predictions = np.array([])
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # Run the forward pass
            data = torch.FloatTensor(data[0].float())

            #Alocate the tensors into the GPU to increase the speed (optional)
            if args['cuda']:
                data = data.cuda()

            output = model(data)

            #Scale the outputs
            output[:, 0] = output[:, 0]*0.65
            output[:, 1] = output[:, 1]*11.9536171
            output[:, 2] = output[:, 2]/20
            output[:, 3] = output[:, 3]/25
            output = output.cpu().detach().numpy()

            predictions = np.vstack([predictions, np.asarray(output)]) \
                if predictions.size else output
    # Save predictions per testing dataset as a matlab file
    name = OUTPUT_FILENAME + 'Test' + strs[test_num] + '.mat'

    save_predictions(predictions, filename=name)
