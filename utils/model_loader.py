#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-04 15:33:04 Wednesday

@author: Nikhil Kapila
"""

import mlflow, torch, sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'..')))

class MLFlowModelLoader:
    def __init__(self, experiment_id, run_id, tracking_uri='http://127.0.0.1:5000', ui_open=True):
        self.exp_id = experiment_id
        self.run_id = run_id
        self.base_path = os.path.join(os.path.dirname(os.getcwd()),'mlruns')
        self.artifact_path = os.path.join(self.base_path, self.exp_id, self.run_id, 'artifacts')
        self.model_path = os.path.join(self.artifact_path, 'model', 'data')
        self.model_logged = False
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = mlflow.tracking.MlflowClient()

        try:
            os.listdir(self.model_path)
            self.model_logged = True
        except Exception as e:
            print(f'It seems that the model for this run was not logged.\n{e}')

        if ui_open is True: self._print_run_summary()

    def _print_run_summary(self):
        run_data = self.client.get_run(self.run_id).data
        # artifacts = self.client.list_artifacts(self.run_id)
        artifacts = os.listdir(self.artifact_path)
        print('Run Summary:')
        print(f'Params:\n {run_data.params}')
        print(f'Metrics:\n {run_data.metrics}')
        print(f'Artifacts:\n {artifacts}')
        if self.model_logged: print(f'Model:\n {os.listdir(self.model_path)}')

    # private function
    def _get_path(self, artifact_name):
        return os.path.join(self.artifact_path, artifact_name)
        
    def _get_history(self, path):
        try:
            import json
            with open(path, 'r') as data:
                history = json.load(data)
                print('Training data loaded successfully.')
                return history
        except Exception as e:
            print(f'Cannot load history.\n{e}')
        
    # private function
    def _get_something(self, object, artifact_name, is_object=True, device='cuda' if torch.cuda.is_available() else 'cpu'):
        path = self._get_path(artifact_name)
        if path is not None:
            try:
                if is_object:
                    object.load_state_dict(torch.load(path, map_location=torch.device(device)))
                    if artifact_name == 'params.pth': object.eval()
                    print(f'Object {artifact_name} loaded.')
                else:
                    return self._get_history(path)
                return object
            except Exception as e:
                print(f'Cannot load specified object. Please check if the correct object (model/optimizer) is passed.\n{e}')
        else:
            print('This artifact does not exist. Please check your run_id and artifact name.')
            return None

    def load_weights(self, model, artifact_name='params.pth', device='cpu'):
        return self._get_something(model, 
                                   artifact_name, 
                                   is_object=True,
                                   device=device)
        
    def load_optimizer_state(self, optimizer, artifact_name='optimizer.pth', device='cpu'):
        return self._get_something(optimizer, 
                                   artifact_name, 
                                   is_object=True,
                                   device=device)
    
    def get_metric_history(self, metric_key='train_err'):
        # https://stackoverflow.com/questions/60616430/mlflow-how-to-read-metrics-or-params-from-an-existing-run
        if self.ui_open is False: raise Exception('Switch ON mlflow ui to run this method.')

        return self.client.get_metric_history(run_id=self.run_id,
                                              key=metric_key)

    def get_mlflow_client(self):
        return self.client

    def get_training_history(self, artifact_name='history.json'):
        return self._get_something(None, 
                                   artifact_name, 
                                   is_object=False)