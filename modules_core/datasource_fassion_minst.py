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


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, args, is_test_data):
        super().__init__()

        self.args = args
        self.is_test_data = is_test_data

        FileUtils.createDir(self.args.path_data)
        self.dataset = torchvision.datasets.FashionMNIST(
            self.args.path_data,
            download=True,
            train=not is_test_data,
            transform=torchvision.transforms.ToTensor()
        )

        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.samples = []
        groups = [{ 'samples': [], 'counter': 0 } for _ in self.dataset.classes]

        for img, label_idx in self.dataset:
            groups[label_idx]['samples'].append(img)

        self.size_samples = 0
        for idx, group in enumerate(groups):
            samples = group['samples']
            self.size_samples += int(len(samples) / self.args.triplet_positives)
            logging.info(f'group:{idx} samples:{len(samples)}')
            random.shuffle(samples)
        random.shuffle(groups)

        if self.args.batch_size % self.args.triplet_positives != 0:
            logging.error(f'batch does not accommodate triplet_positives {self.args.batch_size} {self.args.triplet_positives}')
            exit()

        idx_group = 0
        count_sample_batches = int(self.size_samples / self.args.batch_size)
        self.size_samples = int(self.args.batch_size * count_sample_batches)

        for _ in range(self.size_samples):
            group = groups[idx_group]
            idx_group += 1
            if idx_group >= len(groups):
                idx_group = 0

            for _ in range(self.args.triplet_positives):
                img = group['samples'][group['counter']]
                self.samples.append((idx_group, img))

                group['counter'] += 1
                if group['counter'] >= len(group['samples']):
                    group['counter'] = 0

        logging.info(f'{"test" if is_test_data else "train"} size_samples: {len(self.samples)}')
        # logging.info(f'idx_group: {idx_group}')
        # for idx, group in enumerate(groups):
        #     logging.info(f'group: {idx} counter: {group["counter"]}')

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return self.size_samples


def get_data_loaders(args):

    args.input_size = 28

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