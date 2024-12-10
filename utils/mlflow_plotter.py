#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-09 09:23:55 Monday

@author: Nikhil Kapila
"""

import mlflow, torch, sys, os
from typing import List
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'..')))

class MLFlowPlotter:
    def __init__(self, model_names: List, runids: List, metric_key, tracking_uri='http://127.0.0.1:5000', ui_open=True):
        self.model_names = model_names
        self.run_id_list = runids
        self.tracking_uri = tracking_uri
        self.metric_key = metric_key
        mlflow.set_tracking_uri(self.tracking_uri)
        self.metrics = [mlflow.tracking.MlflowClient().get_metric_history(run_id=run, key=self.metric_key)\
                        for run in self.run_id_list]
        self.epochs = [[plot.step for plot in self.metrics[i]] for i in range(len(self.metrics))]
        self.values = [[plot.value for plot in self.metrics[i]] for i in range(len(self.metrics))]

        if ui_open is False: raise Exception('Switch ON mlflow ui to run the Plotter.')

    def get_history(self):
        return self.metrics
    
    def get_specific_metric(self, metric_key, runs=None):
        if runs is None: runs = self.run_id_list
        return [mlflow.tracking.MlflowClient().get_metric_history(run_id=run, key=metric_key)\
                        for run in runs]
    
    def make_plot_two_metrics(self, title=None, metric_keys=None, error=True):
        if metric_keys is not None and len(metric_keys) > 0:
            metrics = {key: [mlflow.tracking.MlflowClient().get_metric_history(run_id=run, key=key)
                            for run in self.run_id_list]
                    for key in metric_keys}
            
            values = {key: [[plot.value for plot in metrics[key][i]] for i in range(len(metrics[key]))]
                    for key in metric_keys}
        else:
            metric_keys = [self.metric_key]
            values = {metric_keys[0]: self.values}
        
        for key in metric_keys:
            for val, label in zip(values[key], self.model_names):
                plt.plot(val, label=f"{label} ({key})")
        
        plt.xlabel('epochs')
        if error is True: plt.ylabel('error')
        else: plt.ylabel('loss')

        if title is None: plt.title(f'{", ".join(metric_keys)} performance for different models')
        else: plt.title(title)
        
        plt.legend()
        plt.tight_layout()
        fig = plt.gcf()
        plt.show()
        
        return fig

    def make_plot(self, title=None, metric_key=None):
        # epoch = max(self.epochs) # the x axis
        
        if metric_key is not None:
            metrics = [mlflow.tracking.MlflowClient().get_metric_history(run_id=run, key=metric_key)\
                        for run in self.run_id_list]
            # epochs = [[plot.step for plot in metrics[i]] for i in range(len(metrics))]
            values = [[plot.value for plot in metrics[i]] for i in range(len(metrics))]
            # epoch = max(epochs)
        else:
            metric_key = self.metric_key
            values = self.values
            # epochs = self.epochs
        
        for val, label in zip(values, self.model_names):
            plt.plot(val, label=label)
        
        plt.xlabel('epochs')
        plt.ylabel(self.metric_key)
        if title is None: plt.title(f'{metric_key} performance for different models')
        else: plt.title(title)
        plt.legend()
        plt.tight_layout()
        fig = plt.gcf()
        plt.show()
        return fig
        