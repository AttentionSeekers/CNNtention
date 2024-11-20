import torch
import torchvision
from skorch.callbacks import LRScheduler
from skorch.dataset import ValidSplit
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models.resnet import BasicBlock
from torchvision.transforms import transforms

from models.cifar10resnet import Cifar10ResNet


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
            model,
            lr=1e-4,
            optimizer=torch.optim.SGD,
            batch_size=128,
            max_epochs=50,
            weight_decay=0.0001,
            momentum=0.9,
            train_split=ValidSplit(0.9),
            scheduler=None
    ):
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.train_split = train_split
        self.scheduler = scheduler


def _get_cifar10_original_paper_config(model):
    iterations_per_epoch = 45000 // 128 # == 351
    return ExperimentConfig(
        data_config=DataConfig(
            name='CIFAR-10',
            # Quote: "which consists of 50k training images and 10k test images"
            test_size=10000,
            # MISSING Quote 1: "The network inputs are 32x32 images, with the per-pixel mean subtracted"
            # Quote 2: "We follow the simple data augmentation in [24] for training:
            # 4 pixels are padded on each side, and a 32x32 is randomly sampled
            # from the padded image or its horizontal flip."
            train_transform=transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
            ]),
            # MISSING Quote 1: "The network inputs are 32x32 images, with the per-pixel mean subtracted"
            # Quote 2: "For testing, we only evaluate the single view of the original 32x32 image".
            test_transform=transforms.Compose([
                transforms.ToTensor(),
            ])

        ),
        model_config=ModelConfig(
            model=model,
            # Quote: "We start with a learning rate of 0.1
            lr=0.1,
            # I think it is SGD, as they use momentum
            optimizer=torch.optim.SGD,
            # Quote: "These models are trained with mini-batch size of 128"
            batch_size=128,
            # Quote: "terminate training at 64k iterations"
            max_epochs=64000 // iterations_per_epoch, # == 182
            # Quote: "We use a weight decay of 0.0001"
            weight_decay=0.0001,
            # Quote: "and momentum of 0.9"
            momentum=0.9,
            # Quote: "which is determined on a 45k/5k train/val split"
            train_split=ValidSplit(0.9),
            # Quote: "divide it by 10 at 32k and 48k iterations"
            scheduler=LRScheduler(
                policy=MultiStepLR,
                milestones=[
                    32000 // iterations_per_epoch, # == 91
                    48000 // iterations_per_epoch # == 136
                ],
                gamma=0.1
            )
        )
    )


configs = {
    "debug_config": ExperimentConfig(
        data_config=DataConfig(),
        model_config=ModelConfig(torchvision.models.SqueezeNet(num_classes=10))
    ),
    "cifar10_resnet20_original_paper": _get_cifar10_original_paper_config(
        Cifar10ResNet(
            BasicBlock,
            [3, 3, 3],
            10
        )
    ),
    "cifar10_resnet32_original_paper": _get_cifar10_original_paper_config(
        Cifar10ResNet(
            BasicBlock,
            [5, 5, 5],
            10
        )
    ),
    "cifar10_resnet44_original_paper": _get_cifar10_original_paper_config(
        Cifar10ResNet(
            BasicBlock,
            [7, 7, 7],
            10
        )
    ),
    "cifar10_resnet56_original_paper": _get_cifar10_original_paper_config(
        Cifar10ResNet(
            BasicBlock,
            [9, 9, 9],
            10
        )
    )
}
