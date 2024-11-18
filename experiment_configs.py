import torch
import torchvision
from skorch.dataset import ValidSplit
from torchvision.transforms import transforms

from models.resnet import ResNet20


class ExperimentConfig:
    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config

class DataConfig:
    def __init__(
            self,
            name='CIFAR-10',
            test_size=5000,
            train_transform=None,
            test_transform=None
    ):
        self.name = name
        self.test_size = test_size
        self.train_transform = train_transform
        self.test_transform = test_transform

class ModelConfig:
    def __init__(
            self,
            model=ResNet20,
            lr=1e-4,
            optimizer=torch.optim.SGD,
            batch_size=128,
            max_epochs=50,
            weight_decay=0.0001,
            momentum=0.9,
            train_split=ValidSplit(0.9)
    ):
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.train_split = train_split

configs = {
    "debug_config": ExperimentConfig(
        data_config=DataConfig(),
        model_config=ModelConfig()
    ),
    "resnet_paper_default": ExperimentConfig(
        data_config=DataConfig(
            name='CIFAR-10',
            test_size=10000,
            train_transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
            ]),
            test_transform=transforms.ToTensor(),
        ),
        model_config=ModelConfig(
            model=torchvision.models.SqueezeNet(num_classes=10), # TODO replace with actual resnet
            lr=1e-4,
            optimizer=torch.optim.SGD,
            batch_size=128,
            max_epochs=5,
            weight_decay=0.0001,
            momentum=0.9,
            train_split=ValidSplit(0.9)
        )
    ),
}
