#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-11-16 19:32:54 Saturday

@author: Nikhil Kapila
"""
import argparse
from random import random

import torch, torchvision, mlflow, mlflow.sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint, EpochScoring

from experiment_configs import configs, ModelConfig, DataConfig, RANDOM_VAR
import numpy as np

def load_data(config: DataConfig):
    if config.name == 'CIFAR-10':
        train_set = torchvision.datasets.CIFAR10(root='./data',
                                                 train=True,
                                                 download=True,
                                                 transform=config.train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data',
                                                train=False,
                                                download=True,
                                                transform=config.test_transform
                                                )
    elif config.name == 'CIFAR-100':
        train_set = torchvision.datasets.CIFAR100(root='./data',
                                                  train=True,
                                                  download=True,
                                                  transform=config.train_transform)
        test_set = torchvision.datasets.CIFAR100(root='./data',
                                                 train=False,
                                                 download=True,
                                                 transform=config.test_transform
                                                 )
    else:
        raise ValueError('Unknown dataset')

    return train_set, test_set


def train(train_set, model_config: ModelConfig, test_set):
    callbacks = []

    if model_config.scheduler is not None:
        callbacks.append(model_config.scheduler)

    #callbacks.append(
    #    Checkpoint(monitor='valid_acc_best', load_best=True)
    #)

    def valid_err_scoring(net, X, y):
        valid_preds = net.predict(X)
        return 1 - accuracy_score(y, valid_preds)
    callbacks.append(
        ('valid_err', EpochScoring(valid_err_scoring, name='valid_err'))
    )

    # it seems they do show plots of test error vs. iterations in CIFAR-10 paper (does not work yet because net.predict always gives 5k instead of 10k)
    def test_err_scoring(net, X, y):
        test_preds = net.predict(test_set)
        return 100 - accuracy_score(test_set.targets, test_preds) * 100
    callbacks.append(
        ('test_err', EpochScoring(test_err_scoring, name='test_err', use_caching=False))
    )

    network = NeuralNetClassifier(
        model_config.model,
        lr=model_config.lr,
        optimizer=model_config.optimizer,
        batch_size=model_config.batch_size,
        max_epochs=model_config.max_epochs,
        optimizer__weight_decay=model_config.weight_decay,
        optimizer__momentum=model_config.momentum,
        train_split=model_config.train_split,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        criterion=torch.nn.CrossEntropyLoss,
        callbacks=callbacks
    )

    # TODO this may still be wrong! not sure how to call fit correctly with skorch
    return network.fit(train_set, np.array(train_set.targets))


def eval_model(network, test_set):
    train_loss = network.history[:, 'train_loss']
    # mlflow.log_metric('train_loss', train_loss)

    valid_loss = network.history[:, 'valid_loss']
    # mlflow.log_metric('valid_loss', valid_loss)

    valid_err = network.history[:, 'valid_err']

    test_err = network.history[:, 'test_err']
    # calculate error like they do in the original resnet paper
    predictions = network.predict(test_set)
    accuracy = accuracy_score(test_set.targets, predictions)
    error = 100 * (1 - accuracy)
    print(f'Test set accuracy: {accuracy}')
    print(f'Test set error: {error}')

    return train_loss, valid_loss, valid_err, test_err, accuracy, error


def plot(train_loss, valid_loss, valid_err, test_err):
    # Train / Valid Loss plot
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, valid_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./plots/train_val_loss.png')
    plt.show()

    # Valid err plot
    epochs = range(1, len(valid_err) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, valid_err, label='Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Validation Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./plots/valid_err.png')
    plt.show()

    # Valid err plot
    epochs = range(1, len(test_err) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, test_err, label='Test Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Test Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./plots/test_err.png')
    plt.show()


def log_initial_params(config):
    mlflow.log_param('lr', config.model_config.lr)
    mlflow.log_param('batch_size', config.model_config.batch_size),
    mlflow.log_param('max_epochs', config.model_config.max_epochs)


def main(config_id):  # either add default param here or just call main from command line with arg
    config = configs[config_id]

    np.random.seed(RANDOM_VAR)
    torch.manual_seed(RANDOM_VAR)

    with mlflow.start_run():
        log_initial_params(config)

        train_set, test_set = load_data(config.data_config)

        trained_network = train(train_set, config.model_config, test_set)

        train_loss, valid_loss, valid_err, test_err, accuracy, error = eval_model(trained_network, test_set)

        plot(train_loss, valid_loss, valid_err, test_err)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_id", type=str)
    args = parser.parse_args()

    main(args.config_id)
