#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-04 22:07:08 Wednesday

@author: Nikhil Kapila
"""

import random, matplotlib.pyplot as plt, numpy as np
from pytorch_grad_cam import GradCAM, GradCAMElementWise, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch, torchvision, torchvision.transforms as transforms
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'..')))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Viz:
    def __init__(self, models, target_layers, device='cpu'):
        self.models = models
        for model in self.models: 
            model.eval()
            model.to(device)

        self.target_layers = target_layers

        self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4918687901200927, 0.49185976472299225, 0.4918583862227116], std=[0.24697121702736, 0.24696766978537033, 0.2469719877121087])
            ])
        
    def _load_data(self, name='CIFAR-10'):
        if name == 'CIFAR-10':
            train_set = torchvision.datasets.CIFAR10(root='../data',
                                                    train=False,
                                                    download=True)
        return train_set

    def _get_something(self, objects, index=None):
        # from tut: https://jacobgil.github.io/pytorch-gradcam-book/introduction.html#using-from-code-as-a-library
        torch.manual_seed(10)
        data = self._load_data()
        if index is None: index = random.sample(range(len(data)), 1)[0]
        image, label = data[index]
        input = self.transform(image).unsqueeze(0) #unsqueeze to add batch dimension
        
        # torch.random.set_manual_seed(10)
        
        for model in self.models: model.zero_grad()
        cams = objects
        targets = [ClassifierOutputTarget(label)]
        grayscale_cams = [cam(input, targets)[0] for cam in cams]
        rgb_img = np.array(image) / 255.0
        viz = [show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True) for grayscale_cam in grayscale_cams]

        with torch.no_grad():
            torch.manual_seed(10)
            preds = [model(input).argmax() for model in self.models]

        plt.figure(figsize=(7,7))
        plots = len(viz)+1 # number of subplots = 1 for image and other models
        
        plt.subplot(1, plots, 1)
        plt.imshow(rgb_img)
        plt.title(f"Original Image\nLabel: {label}\nIndex: {index}")
        plt.axis("off")

        model_names = [type(model).__name__ for model in self.models]
        # Grad-CAM visualizations
        for i, (v, name, p) in enumerate(zip(viz, model_names, preds), start=2):
            plt.subplot(1, plots, i)
            plt.imshow(v)
            plt.title(f"Grad-CAM:\n{name}\nPred:{p}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def get_gradcam(self, index=None):
        torch.manual_seed(10)
        objects = [GradCAM(model=model, target_layers=[layer]) for model, layer in zip(self.models, self.target_layers)]
        return self._get_something(objects,
                                   index)

    def get_gradcam_elementwise(self, index=None):
        torch.manual_seed(10)
        objects = [GradCAMElementWise(model=model, target_layers=[layer]) for model, layer in zip(self.models, self.target_layers)]
        return self._get_something(objects,
                                   index)

    def get_hi_res_cam(self, index=None):
        torch.manual_seed(10)
        objects = [HiResCAM(model=model, target_layers=[layer]) for model, layer in zip(self.models, self.target_layers)]
        return self._get_something(objects,
                                   index)