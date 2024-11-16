#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-11-16 19:32:54 Saturday

@author: Nikhil Kapila
"""

import torch, torchvision, skorch, mlflow, mlflow.sklearn, argparse
import torchvision.transforms as transforms
from skorch import NeuralNetClassifier
from sklearn.model_selection import train_test_split
import models.squeezenet as resnet #TODO: implement resnet_cbam in models/ and change here

RANDOM_VAR = 10

def load_data(name='CIFAR-10'): # default is cifar-10
    transform = transforms.Compose([
        # TODO: Need to add different augmentation from Resnet paper.
        # Can be a tuning hyperparam too?
    ])

    if (name=='CIFAR-10'):
        train_d = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
        test = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform)
    
    train, val = train_test_split(train_d, random_state=RANDOM_VAR, test_size=5000)
    return train, val, test

def train(train, val, lr, batch, epochs):
    # TODO: Other args to be passed, l2 reg, optimizer etc as call back
    # l2 reg: https://skorch.readthedocs.io/en/stable/user/FAQ.html#how-do-i-apply-l2-regularization
    # callback: https://skorch.readthedocs.io/en/stable/user/callbacks.html
    network = NeuralNetClassifier(
        resnet, #TODO: placeholder, import path to be changed in line 13.
        max_epochs=epochs,
        lr=lr,
        batch_size=batch,
        optimizer=torch.nn.optim.Adam, #TODO: Is this the same Opt in the ResNet paper? (Maybe irrelevant question since tuning can have a different optimizer)
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            # TODO: Any other callbacks that may be required for plots
        ]
    )

    network.fit(train)

    for epoch, row in enumerate(network.history):
        train_loss = row['train_loss']
        mlflow.log_metric('loss_train', train_loss, step=epoch)
        # TODO: Need to check if it logs loss (I think skorch does this automatically)

        # TODO: TO CHECK:
        # Resnet paper splits into 45k train, 5k val and 5k test.
        # Tuning: Model is trained on 45k train, validated on 5k val.
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch, shuffle=False)
        val_loss, val_tot = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                val_loss += network.criterion_(network.module_(inputs), targets).item() * inputs.size(0)
                val_tot /= inputs.size(0)
        val_loss /= val_tot
        mlflow.log_metric('loss_val', val_loss, step=epoch)
    
    return network

def eval_model():
    # TODO
    pass

def plot():
    # TODO: Complete this using matplotlib
    # refer here: 
    pass

def main(lr=1e-4, batch=128, epochs=50, dataset='CIFAR-10'): # cmdLine=True):
    # OPTIONAL command line invoke
    # if cmdLine:
    #     parser = argparse.ArgumentParser(description='CNN Att train pipeline')
    #     parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    #     parser.add_argument('--batch', type=int, default=128, help='Batch size')
    #     parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    #     args = parser.parse_args()
    #     lr, batch, epochs = args.lr, args.batch, args.epochs
    with mlflow.start_run():
        mlflow.log_param('lr', lr)
        mlflow.log_param('batch_size', batch),
        mlflow.log_param('max_epochs', epochs)
        
        train, val, test = load_data(dataset)
        # TODO
    
    

if __name__ == '__main__':
    main()
    # main(cmdLine=True)



