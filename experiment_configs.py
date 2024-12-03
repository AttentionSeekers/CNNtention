import torch
import torchvision
from skorch.callbacks import LRScheduler
from skorch.dataset import ValidSplit
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import transforms

from models.cbamBlock import CBAMBlock
from models.cifar10resnet import Cifar10ResNet
from models.resnet_self_att import ResnetSelfAtt
from models.resnet_multi_head_att import ResnetMultiHeadAtt
from models.originalBasicBlock import OriginalBasicBlock

RANDOM_VAR = 10

class ExperimentConfig:
    def __init__(self, experiment_name, data_config, model_config):
        self.experiment_name = experiment_name
        self.underscored_lowercased_name = self.experiment_name.replace(" ", "_").lower()
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
            reg=None,
            batch_size=128,
            max_epochs=50,
            weight_decay=0.0001,
            momentum=0.9,
            train_split=None,
            scheduler=None,
            log_model=False,
            num_heads=None,
            early_stopping_params:dict=None, 
            # Should be a dict of params: patience, threshold, threshold_mode
            use_early_stopping=False, 
            # should only be used for final evaluation, not for tuning
            add_test_set_eval=False
    ):
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.reg = reg
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.train_split = train_split
        self.scheduler = scheduler
        self.log_model = log_model
        self.use_early_stopping = False
        self.early_stopping_params = early_stopping_params
        if num_heads != None: self.num_heads = num_heads
        self.add_test_set_eval = add_test_set_eval


def _get_cifar10_original_paper_evaluation_config(experiment_name, model) -> ExperimentConfig:
    # see https://github.com/a-martyn/resnet/blob/master/main.ipynb
    means = [0.4918687901200927, 0.49185976472299225, 0.4918583862227116]
    stds = [0.24697121702736, 0.24696766978537033, 0.2469719877121087]

    iterations_per_epoch = 45000 // 128 # == 351
    return ExperimentConfig(
        experiment_name,
        data_config=DataConfig(
            name='CIFAR-10',
            # Quote: "which consists of 50k training images and 10k test images"
            test_size=10000,
            # Quote 1: "The network inputs are 32x32 images, with the per-pixel mean subtracted"
            # Quote 2: "We follow the simple data augmentation in [24] for training:
            # 4 pixels are padded on each side, and a 32x32 is randomly sampled
            # from the padded image or its horizontal flip."
            train_transform=transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=means, std=stds)
            ]),
            # Quote 1: "The network inputs are 32x32 images, with the per-pixel mean subtracted"
            # Quote 2: "For testing, we only evaluate the single view of the original 32x32 image".
            test_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=means, std=stds)
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
            # Quote: "terminate training at 64k iterations which is determined on a 45k/5k train/val split"
            max_epochs=64000 // iterations_per_epoch, # == 182
            # Quote: "We use a weight decay of 0.0001"
            weight_decay=0.0001,
            # Quote: "and momentum of 0.9"
            momentum=0.9,
            # Quote: "We present experiments trained on the training set and evaluated on the test set."
            # Note: They likely tuned on 45k/5k train/val split, but evaluated on the training set (this is why default is None)
            train_split=None,
            # Quote: "divide it by 10 at 32k and 48k iterations [...] which is determined on a 45k/5k train/val split"
            scheduler=LRScheduler(
                policy=MultiStepLR,
                milestones=[
                    32000 // iterations_per_epoch, # == 91
                    48000 // iterations_per_epoch # == 136
                ],
                gamma=0.1 # this is the multiplication factor ("divide it by 10")
            ),
            add_test_set_eval=True
        )
    )


def _get_cifar10_original_paper_training_config(experiment_name, model) -> ExperimentConfig:
    config = _get_cifar10_original_paper_evaluation_config(experiment_name, model)
    config.model_config.train_split = ValidSplit(0.1, random_state=RANDOM_VAR)
    config.model_config.add_test_set_eval = False
    return config


configs = { # mapping keys to lambdas to ensure that the configs are only loaded upon request. Otherwise, the seed may not be applied and we get non-determinstic results!
    "debug_config": lambda: ExperimentConfig(
        "Debug Config",
        data_config=DataConfig(),
        model_config=ModelConfig(torchvision.models.SqueezeNet(num_classes=10))
    ),
    "cifar10_resnet20_original_paper": lambda: _get_cifar10_original_paper_evaluation_config(
        "Original ResNet20",
        Cifar10ResNet(
            OriginalBasicBlock,
            [3, 3, 3],
            10
        )
    ),
    "cifar10_resnet20_original_paper_tuning": lambda: _get_cifar10_original_paper_training_config(
        "Original ResNet20 Tuning",
        Cifar10ResNet(
            OriginalBasicBlock,
            [3, 3, 3],
            10
        )
    ),
    "cifar10_resnet20_cbam_baseline_training": lambda: _get_cifar10_original_paper_training_config(
"CBAM ResNet20 Tuning",
        Cifar10ResNet(
            CBAMBlock, # TODO block not yet implemented properly
            [3, 3, 3],
            10
        )
    ),
    "self_att_tuning": lambda: ExperimentConfig(
        "SelfAtt ResNet20 with Logging",
        DataConfig(name='CIFAR-10',
                   test_size=10000,
                   train_transform=transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4918687901200927, 0.49185976472299225, 0.4918583862227116], std=[0.24697121702736, 0.24696766978537033, 0.2469719877121087])
            ]),
            test_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4918687901200927, 0.49185976472299225, 0.4918583862227116], std=[0.24697121702736, 0.24696766978537033, 0.2469719877121087])
            ])),
        ModelConfig(model=ResnetSelfAtt(OriginalBasicBlock, [3,3,3], 10), 
                    lr=0.005,
                    optimizer=torch.optim.Adam,
                    batch_size=128,
                    max_epochs=64000//(45000 // 128),
                    weight_decay=0.0001,
                    momentum=0.9,
                    train_split=ValidSplit(cv=0.2),
                    scheduler=LRScheduler(
                        policy=MultiStepLR,
                        milestones=[
                            32000 // (45000 // 128), # == 91
                            48000 // (45000 // 128) # == 136
                        ],
                        gamma=0.1 # this is the multiplication factor ("divide it by 10")
                    ),
                    add_test_set_eval=False,
                    log_model=True)
    ),
    "multi_head_att_tuning": lambda: ExperimentConfig(
        "MultiHeadAtt ResNet20 with Logging",
        DataConfig(name='CIFAR-10',
                   test_size=10000,
                   train_transform=transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4918687901200927, 0.49185976472299225, 0.4918583862227116], std=[0.24697121702736, 0.24696766978537033, 0.2469719877121087])
            ]),
            test_transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4918687901200927, 0.49185976472299225, 0.4918583862227116], std=[0.24697121702736, 0.24696766978537033, 0.2469719877121087])
            ])),
        ModelConfig(model=ResnetMultiHeadAtt(OriginalBasicBlock, [3,3,3], 10), 
                    lr=0.005,
                    optimizer=torch.optim.Adam,
                    batch_size=128,
                    max_epochs=64000//(45000 // 128),
                    weight_decay=0.0001,
                    momentum=0.9,
                    train_split=None,
                    num_heads=4,
                    scheduler=LRScheduler(
                        policy=MultiStepLR,
                        milestones=[
                            32000 // (45000 // 128), # == 91
                            48000 // (45000 // 128) # == 136
                        ],
                        gamma=0.1 # this is the multiplication factor ("divide it by 10")
                    ),
                    add_test_set_eval=True,
                    log_model=True)
    )
}
