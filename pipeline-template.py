#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-11-16 19:32:54 Saturday

@author: Nikhil Kapila
"""
import argparse

import torch, torchvision, mlflow, mlflow.sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint

from experiment_configs import configs, ModelConfig, DataConfig
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


def train(train_set, model_config: ModelConfig):
    callbacks = []

    if model_config.scheduler is not None:
        callbacks.append(model_config.scheduler)

    callbacks.append(
        Checkpoint(monitor='valid_loss_best', f_params='best_model_params_valid_loss.pt')
    )

    callbacks.append(
        Checkpoint(monitor='valid_acc_best', f_params='best_model_params_valid_acc.pt')
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
    fitted_net = network.fit(train_set, np.array(train_set.targets))

    best_valid_loss_net = NeuralNetClassifier(model_config.model)
    best_valid_loss_net.load_params(f_params='best_model_params_valid_loss.pt')

    best_valid_acc_net = NeuralNetClassifier(model_config.model)
    best_valid_acc_net.load_params(f_params='best_model_params_valid_acc.pt')

    return fitted_net, best_valid_loss_net, best_valid_acc_net


def eval_model(network, test_set):
    train_loss = network.history[:, 'train_loss']
    # mlflow.log_metric('train_loss', train_loss)

    valid_loss = network.history[:, 'valid_loss']
    # mlflow.log_metric('valid_loss', valid_loss)

    # calculate error like they do in the original resnet paper
    predictions = network.predict(test_set)
    accuracy = accuracy_score(test_set.targets, predictions)
    error = 100 * (1 - accuracy)
    print(f'Test set accuracy: {accuracy}')
    print(f'Test set error: {error}')

    return train_loss, valid_loss, accuracy, error


def plot(train_loss, valid_loss):
    # Train / Valid Loss plot
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label='Training Loss', marker='o')
    plt.plot(epochs, valid_loss, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def log_initial_params(config):
    mlflow.log_param('lr', config.model_config.lr)
    mlflow.log_param('batch_size', config.model_config.batch_size),
    mlflow.log_param('max_epochs', config.model_config.max_epochs)


def main(config_id):  # either add default param here or just call main from command line with arg
    config = configs[config_id]

    with mlflow.start_run():
        log_initial_params(config)

        train_set, test_set = load_data(config.data_config)

        trained_network, best_valid_loss_net, best_valid_acc_net = train(train_set, config.model_config)

        train_loss, valid_loss, accuracy, error = eval_model(trained_network, test_set)
        plot(train_loss, valid_loss)
        train_loss, valid_loss, accuracy, error = eval_model(best_valid_loss_net, test_set)
        plot(train_loss, valid_loss)
        train_loss, valid_loss, accuracy, error = eval_model(best_valid_acc_net, test_set)
        plot(train_loss, valid_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_id", type=str, )
    args = parser.parse_args()

    #main(args.config_id)
    main("cifar10_resnet20_original_paper")
    main("cifar10_resnet32_original_paper")
    main("cifar10_resnet44_original_paper")
    main("cifar10_resnet56_original_paper")

