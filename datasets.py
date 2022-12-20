"""Some data loading module."""
from typing import List
import random

import utils
from pathlib import Path
from torchvision.datasets import MNIST, CelebA, CIFAR10, FashionMNIST
from torchvision import transforms
from data_utils.dataloader import NumpyLoader, GradLoader, Normalize, Cast, AddChannelDim

import configlib

# Configuration arguments
parser = configlib.add_parser("Dataset config")
parser.add_argument("--data_dir", default='/tmp/data', type=str, metavar='DATA_PATH',
        help="The path where to stop the data.")
parser.add_argument("--dataset", default='MNIST', type=str, metavar='DATASET_NAME',
        help="The dataset to use: MNIST, CIFAR10, FMNIST.")
parser.add_argument("--batch_size", "-b", default=128, type=int,
        help="The data loader batch size.")

def get_dataset(c: configlib.Config):
    "Returns the train, test datasets and respective data loaders"
    if c.dataset == 'MNIST':
        return run_get_dataset(c, get_mnist)
    elif c.dataset == 'FMNIST':
        return run_get_dataset(c, get_fmnist)
    elif c.dataset == 'CIFAR10':
        return run_get_dataset(c, get_cifar10)
    elif c.dataset == 'CelebA':
        return run_get_dataset(c, get_celeba)
    else:
        raise NotImplementedError

def run_get_dataset(c: configlib.Config, get_dataset_fn):
    utils.ensure_dir(c.data_dir)
    return get_dataset_fn(c)

def get_mnist(c: configlib.Config):
    normalize = [
        #  transforms.RandomCrop(28, padding=4),
        Cast(),
        Normalize([0.5, ], [0.5, ]),
        AddChannelDim(),
    ]
    transform = transforms.Compose(normalize)

        # Load MNIST dataset and use Jax compatible dataloader
    mnist = MNIST(c.data_dir, download=True, transform=transform)
    im_dataloader = NumpyLoader(
        mnist,
        batch_size=c.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )
    mnist_test = MNIST(c.data_dir, train=False, download=True, transform=transform)
    im_dataloader_test = NumpyLoader(
        mnist_test,
        batch_size=c.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8
    )

    return mnist, mnist_test, im_dataloader, im_dataloader_test

def get_fmnist(c: configlib.Config):
    normalize = [
        #  transforms.RandomCrop(28, padding=4),
        Cast(),
        Normalize([0.5, ], [0.5, ]),
        AddChannelDim(),
    ]
    transform = transforms.Compose(normalize)

    fashion_mnist = FashionMNIST(c.data_dir, download=True, transform=transform)
    im_dataloader = NumpyLoader(
        fashion_mnist,
        batch_size=c.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )
    fashion_mnist_test = FashionMNIST(c.data_dir, train=False, download=True, transform=transform)
    im_dataloader_test = NumpyLoader(
        fashion_mnist_test,
        batch_size=c.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8
    )

    return fashion_mnist, fashion_mnist_test, im_dataloader, im_dataloader_test

def get_cifar10(c: configlib.Config):
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        Cast(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    train_transform = transforms.Compose(augmentations + normalize)
    test_transform = transforms.Compose(normalize)

    cifar10_train = CIFAR10(root=c.data_dir, train=True, download=True, transform=train_transform)
    cifar10_test = CIFAR10(root=c.data_dir, train=False, download=True, transform=test_transform)

    im_dataloader = NumpyLoader(
        cifar10_train,
        batch_size=c.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )
    im_dataloader_test = NumpyLoader(
        cifar10_test,
        batch_size=c.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8
    )
    return cifar10_train, cifar10_test, im_dataloader, im_dataloader_test

class MyCelebA(CelebA):
    def __init__(self, *args, **kwargs):
        if 'attr_name' in kwargs:
            attr_name = kwargs['attr_name']
            del kwargs['attr_name']
        else:
            raise ValueError("MyCelebA requires an attribute name `attr_name' to use as label")

        super(self.__class__, self).__init__(*args, **kwargs)

        self.attr_i = self.attr_names.index(attr_name)

    def __getitem__(self, i):
        data, label = super(self.__class__, self).__getitem__(i)
        return data, label[self.attr_i].item()

def get_celeba(c: configlib.Config):
    normalize = [
        transforms.Resize([256, 256]),
        #  transforms.RandomCrop(224),
        Cast(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    transform = transforms.Compose(normalize)

    celeba = MyCelebA(c.data_dir, download=False, transform=transform, attr_name="Smiling")
    im_dataloader = NumpyLoader(
        celeba,
        batch_size=c.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )
    celeba_test = MyCelebA(c.data_dir, split="test", download=False, transform=transform, attr_name="Smiling")
    im_dataloader_test = NumpyLoader(
        celeba_test,
        batch_size=c.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8
    )

    return celeba, celeba_test, im_dataloader, im_dataloader_test

