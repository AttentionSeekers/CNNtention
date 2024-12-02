#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-11-16 19:32:54 Saturday

@author: Nikhil Kapila
"""
import argparse
import random

import mlflow
import torch, torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, MlflowLogger

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



    # for final evaluations, we should use the entire training set and then we cannot track validation
    if model_config.train_split is not None:
        def valid_err_scoring(net, X, y):
            valid_preds = net.predict(X)
            return 1 - accuracy_score(y, valid_preds)
        callbacks.append(
            ('valid_err', EpochScoring(valid_err_scoring, name='valid_err'))
        )

        #def train_err_scoring(net, X, y):
        #    train_preds = net.predict(X)
        #    return 100 - accuracy_score(y, train_preds) * 100

        #callbacks.append(
        #    # would be better to use caching, but this increases memory usage by a lot
        #    ('train_err', EpochScoring(train_err_scoring, name='train_err', on_train=True, use_caching=False))
        #)
    else:
        def train_err_scoring(net, X, y):
            train_preds = net.predict(X)
            return 100 - accuracy_score(X.targets, train_preds) * 100

        callbacks.append(
            # would be better to use caching, but this increases memory usage by a lot
            ('train_err', EpochScoring(train_err_scoring, name='train_err', on_train=True, use_caching=False))
        )

    # we should not evaluate on the test set until we are done with hyperparameter tuning
    if model_config.add_test_set_eval:
        def test_err_scoring(net, X, y):
            test_preds = net.predict(test_set)
            return 100 - accuracy_score(test_set.targets, test_preds) * 100
        callbacks.append(
            ('test_err', EpochScoring(test_err_scoring, name='test_err', use_caching=False, on_train=True))
        )

    ml_flow_logger = MlflowLogger()
    callbacks.append(
        ml_flow_logger
    )

    # unfortunately, this does not seem to be logged automatically by the Skorch callback
    # so I did it manually (maybe you find out how it can be done by Skorch)
    mlflow.log_param('learning_rate', model_config.lr)
    mlflow.log_param('optimizer', model_config.optimizer.__name__)
    mlflow.log_param('batch_size', model_config.batch_size)
    mlflow.log_param('max_epochs', model_config.max_epochs)
    mlflow.log_param('weight_decay', model_config.weight_decay)
    mlflow.log_param("momentum", model_config.momentum)

    opt_params = {}
    opt_params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    mlflow.log_param('device', opt_params['device'])
    if not model_config.optimizer.__name__ == 'Adam':
        opt_params['optimizer__momentum'] = model_config.momentum
    if model_config.optimizer.__class__.__name__ == 'ResnetMultiHeadAtt':
      opt_params['num_heads'] = model_config.num_heads

    network = NeuralNetClassifier(
        model_config.model,
        lr=model_config.lr,
        optimizer=model_config.optimizer,
        batch_size=model_config.batch_size,
        max_epochs=model_config.max_epochs,
        optimizer__weight_decay=model_config.weight_decay,
        # optimizer__momentum=model_config.momentum,
        iterator_train__shuffle=True, # this is important! otherwise each batch across epochs is the same...
        iterator_valid__shuffle=False,
        train_split=model_config.train_split,
        criterion=torch.nn.CrossEntropyLoss,
        callbacks=callbacks,
        **opt_params
    )

    return network.fit(train_set, np.array(train_set.targets)), network.module_


def eval_model(network, test_set):
    # the network history is a Skorch history object, so checking for presence of keys the usual way does not work
    def try_to_access_from_history(key):
        try:
            return network.history[:, key]
        except:
            return None

    train_loss = try_to_access_from_history('train_loss')
    valid_loss = try_to_access_from_history('valid_loss')

    train_err = try_to_access_from_history('train_err')
    valid_err = try_to_access_from_history('valid_err')
    test_err = try_to_access_from_history('test_err')

    # calculate final accuracy and error (as in the original resnet paper)
    predictions = network.predict(test_set)
    accuracy = accuracy_score(test_set.targets, predictions)
    error = 100 * (1 - accuracy)
    print(f'Test set accuracy: {accuracy}')
    print(f'Test set error: {error}')

    mlflow.log_metric('final accuracy', accuracy)
    mlflow.log_metric('final error', error)

    return train_loss, valid_loss, valid_err, train_err, test_err, accuracy, error


def plot(config, train_loss, valid_loss, valid_err, train_err, test_err):
    if train_loss is not None:
        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_loss, label='Training Loss')
        if valid_loss is not None:
            plt.plot(epochs, valid_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'{config.experiment_name}: Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./plots/{config.underscored_lowercased_name}_train_val_loss.png')
        plt.show()

    if valid_err is not None:
        epochs = range(1, len(valid_err) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, valid_err, label='Validation Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title(f'{config.experiment_name}: Validation Error')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./plots/{config.underscored_lowercased_name}_valid_err.png')
        plt.show()

    if train_err is not None:
        epochs = range(1, len(train_err) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_err, label='Train Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title(f'{config.experiment_name}: Train Error')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./plots/{config.underscored_lowercased_name}_train_err.png')
        plt.show()

    if test_err is not None:
        epochs = range(1, len(test_err) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, test_err, label='Test Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title(f'{config.experiment_name}: Test Error')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./plots/{config.underscored_lowercased_name}_test_err.png')
        plt.show()

    if train_err is not None and test_err is not None:
        epochs = range(1, len(test_err) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_err, label='Train Error')
        plt.plot(epochs, test_err, label='Test Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title(f'{config.experiment_name}: Train and Test Error')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./plots/{config.underscored_lowercased_name}_train_and_test_err.png')
        plt.show()

def print_hyperparams(config):
  model_config = vars(config.model_config)

  for key, value in model_config.items():
    if key != 'model':
        print(f"{key}: {value}")

def main(config_id, debug=False):  # either add default param here or just call main from command line with arg
    config = configs[config_id]()
    
    if debug:
        print(f'=== Model Details ===\n{config.model_config.model}')
        print(f'\n=== Hyperparameter Details ===')
        print_hyperparams(config)
        print(f'\nExperiment name is {config.underscored_lowercased_name}')

    mlflow.set_experiment(experiment_name=config.underscored_lowercased_name)

    with mlflow.start_run() as run:
        if debug:
            print(f"\nRun ID: {mlflow.active_run().info.run_id}")
            print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
            print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

        train_set, test_set = load_data(config.data_config)

        trained_network, model = train(train_set, config.model_config, test_set)

        train_loss, valid_loss, valid_err, train_err, test_err, accuracy, error = eval_model(trained_network, test_set)

        plot(config, train_loss, valid_loss, valid_err, train_err, test_err)
        
        if config.model_config.log_model: mlflow.pytorch.log_model(model, "model")


if __name__ == '__main__':
    random.seed(RANDOM_VAR)
    np.random.seed(RANDOM_VAR)
    torch.manual_seed(RANDOM_VAR)
    torch.cuda.manual_seed(RANDOM_VAR)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()
    parser.add_argument("config_id", type=str)
    args = parser.parse_args()

    main(args.config_id)


