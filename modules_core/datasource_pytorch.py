import os
import unicodedata
import string
import glob
import io
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.sampler
import random
import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torchvision
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tensorboardX
import argparse
from datetime import datetime
import sklearn
import sklearn.model_selection
from enum import Enum
import json
import logging
import numpy as np
import traceback, sys
import itertools

from modules.file_utils import FileUtils
from modules.dict_to_obj import DictToObj
from distutils.dir_util import copy_tree

# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, args, is_test_data):
        super().__init__()

        self.args = args
        self.is_test_data = is_test_data

        path_data = f'{self.args.path_data}/{self.args.datasource_type}'
        FileUtils.createDir(path_data)

        if not os.path.exists(f'{self.args.path_data}/{self.args.datasource_type}/lock'):
            with open(f'{self.args.path_data}/{self.args.datasource_type}/lock', 'w') as fp_download_lock:
                fp_download_lock.write('')
            time.sleep(1.0)

        with open(f'{self.args.path_data}/{self.args.datasource_type}/lock', 'r+') as fp_download_lock:
            FileUtils.lock_file(fp_download_lock)

            if self.args.datasource_type == 'fassion_minst':
                self.dataset = torchvision.datasets.FashionMNIST(
                    path_data,
                    download=True,
                    train=not is_test_data,
                    transform=torchvision.transforms.ToTensor()
                )
            elif self.args.datasource_type == 'minst':
                self.dataset = torchvision.datasets.MNIST(
                    path_data,
                    download=True,
                    train=not is_test_data,
                    transform=torchvision.transforms.ToTensor()
                )
            elif self.args.datasource_type == 'cifar_10':
                self.dataset = torchvision.datasets.CIFAR10(
                    path_data,
                    download=True,
                    train=not is_test_data,
                    transform=torchvision.transforms.Compose([
                        torchvision.transforms.Grayscale(),
                        torchvision.transforms.ToTensor()
                    ])
                )
            elif self.args.datasource_type == 'cifar_100':
                self.dataset = torchvision.datasets.CIFAR100(
                    path_data,
                    download=True,
                    train=not is_test_data,
                    transform=torchvision.transforms.Compose([
                        torchvision.transforms.Grayscale(),
                        torchvision.transforms.ToTensor()
                    ])
                )
            elif self.args.datasource_type == 'eminst': # extended minst https://arxiv.org/pdf/1702.05373.pdf
                self.dataset = torchvision.datasets.EMNIST(
                    path_data,
                    download=True,
                    split='balanced',
                    train=not is_test_data,
                    transform=torchvision.transforms.ToTensor()
                )

            FileUtils.unlock_file(fp_download_lock)


        self.classes = self.dataset.test_labels if is_test_data else self.dataset.train_labels
        if isinstance(self.classes, torch.Tensor):
            self.classes = self.classes.numpy()
        self.classes = np.arange(np.max(self.classes) + 1).tolist()
        groups = [{ 'samples': [], 'counter': 0 } for _ in self.classes]

        for img, label_idx in self.dataset:
            groups[int(label_idx)]['samples'].append(img)

        args.input_size = img.size(1) # channels, w, h

        if not is_test_data:
            ids = [int(it) for it in self.args.datasource_exclude_train_class_ids]
            ids = sorted(ids, reverse=True)
            for remove_id in ids:
                del self.classes[remove_id]
                del groups[remove_id]
        else:
            if len(self.args.datasource_include_test_class_ids):
                ids = set(self.classes) - set([int(it) for it in self.args.datasource_include_test_class_ids])
                ids = list(ids)
                ids = sorted(ids, reverse=True)
                for remove_id in ids:
                    del self.classes[remove_id]
                    del groups[remove_id]

        self.classes = np.array(self.classes, dtype=np.int)
        self.size_samples = 0
        for idx, group in enumerate(groups):
            samples = group['samples']
            self.size_samples += int(len(samples) / self.args.triplet_positives)
            #logging.info(f'group:{idx} samples:{len(samples)}')
            random.shuffle(samples)
        self.groups = groups

        logging.info(f'{self.args.datasource_type} {"test" if is_test_data else "train"}: classes: {len(groups)} total: {self.size_samples}')

        if self.args.batch_size % self.args.triplet_positives != 0 or self.args.batch_size <= self.args.triplet_positives:
            logging.error(f'batch does not accommodate triplet_positives {self.args.batch_size} {self.args.triplet_positives}')
            exit()
        self.reshuffle()

    def reshuffle(self):
        random.shuffle(self.groups)
        for idx, group in enumerate(self.groups):
            samples = group['samples']
            random.shuffle(samples)

        idx_group = 0
        count_sample_batches = int(self.size_samples / self.args.batch_size)
        self.size_samples = int(self.args.batch_size * count_sample_batches)
        self.samples = []
        for _ in range(self.size_samples):
            group = self.groups[idx_group]
            idx_group += 1
            if idx_group >= len(self.groups):
                idx_group = 0

            for _ in range(self.args.triplet_positives):
                img = group['samples'][group['counter']]
                self.samples.append((idx_group, img))

                group['counter'] += 1
                if group['counter'] >= len(group['samples']):
                    group['counter'] = 0

        logging.info(f'{"test" if self.is_test_data else "train"} size_samples: {len(self.samples)}')
        # logging.info(f'idx_group: {idx_group}')
        # for idx, group in enumerate(groups):
        #     logging.info(f'group: {idx} counter: {group["counter"]}')

    # 28x28
    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return self.size_samples


def get_data_loaders(args):

    dataset_train = Dataset(args, is_test_data=False)
    dataset_test = Dataset(args, is_test_data=True)

    logging.info('train dataset')
    sampler_train = torch.utils.data.sampler.SequentialSampler(dataset_train) # important for triplet sampling to work correctly
    data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                    batch_size=args.batch_size,
                                                    sampler=sampler_train,
                                                    num_workers=args.datasource_workers)
    logging.info('test dataset')
    sampler_test = torch.utils.data.sampler.SequentialSampler(dataset_test) # important for triplet sampling to work correctly
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=args.batch_size,
                                                   sampler=sampler_test,
                                                   num_workers=args.datasource_workers)

    return data_loader_train, data_loader_test