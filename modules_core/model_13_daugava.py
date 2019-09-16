import os
import unicodedata
import string
import glob
import io
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tensorboardX
import argparse
import datetime
import sklearn
import sklearn.model_selection
import logging
import json

import torch
import torch.nn as nn
import torchvision.models
import torchvision.models.resnet
import torch.utils.model_zoo

from modules.block_pass import Pass
from modules.block_resnet_2d_leaky import ResBlock2D, ConvBlock2D
from modules.block_reshape2 import Reshape
from modules.dict_to_obj import DictToObj
import modules.torch_utils as torch_utils
from modules.layer_kaf import KAF, KAF2D
from modules.layer_maxout import MaxoutLinear

# Variant without affline, fully convolutional net
# fixed refined part pooling

class Model(torch.nn.Module):
    def __init__(self, args):

        super(Model, self).__init__()
        self.args = args

        if not hasattr(self.args, 'embedding_function'):
            self.args.embedding_function = 'tanh'

        layers_encoder, output_size = self.__create_layers_encoder(name='enc', input_size=args.input_size)

        logging.info(f'conv outputs: {output_size}x{output_size} {self.channels_conv_size} ')

        self.layers_encoder = layers_encoder

        input_size_emb = self.channels_conv_size * output_size**2
        self.layers_pre_embedding = torch.nn.Sequential(
            Reshape(shape=input_size_emb)
        )

        if self.args.is_layers_embedding_batchnorm:
            self.layers_pre_embedding.add_module('emb_bn_suffix', torch.nn.BatchNorm1d(num_features=input_size_emb))

        if self.args.layers_embedding_dropout > 0:
            self.layers_pre_embedding.add_module(f'dropout_emb', torch.nn.Dropout(p=self.args.layers_embedding_dropout))

        self.layers_embedding_scaler = torch.nn.Sequential(
            Reshape(shape=input_size_emb)
        )
        self.layers_embedding = torch.nn.Sequential(
            Reshape(shape=input_size_emb)
        )

        for idx in range(max(1, args.embedding_layers)):
            output_size_emb = args.embedding_layers_hidden
            output_size_emb_scaler = output_size_emb
            if idx >= args.embedding_layers - 1:
                output_size_emb = args.embedding_size
                output_size_emb_scaler = 1

            self.layers_embedding.add_module(f'emb_linear_{idx}', torch.nn.Linear(input_size_emb, output_size_emb, bias=False))
            if idx < args.embedding_layers - 1:
                self.layers_embedding.add_module(f'emb_relu_linear_{idx}', torch.nn.LeakyReLU(negative_slope=self.args.leaky_relu_slope))

            self.layers_embedding_scaler.add_module(f'emb_scaler_linear_{idx}', torch.nn.Linear(input_size_emb, output_size_emb_scaler, bias=False))
            if idx < args.embedding_layers - 1:
                self.layers_embedding_scaler.add_module(f'emb_scaler_relu_linear_{idx}', torch.nn.LeakyReLU(negative_slope=self.args.leaky_relu_slope))

            input_size_emb = output_size_emb

        if self.args.datasource_classes_train:
            logging.info(f'classification layer classes count: {self.args.datasource_classes_train}')
            layers_classification = []
            layers_emb_size = self.args.embedding_size

            for idx in range(self.args.class_layers - 1):
                layers_classification += [
                    torch.nn.BatchNorm1d(num_features=layers_emb_size),
                    torch.nn.Linear(in_features=layers_emb_size, out_features=self.args.class_layers_hidden)
                ]
                layers_emb_size = self.args.class_layers_hidden

            layers_classification += [
                torch.nn.BatchNorm1d(num_features=layers_emb_size),
                torch.nn.Linear(in_features=layers_emb_size, out_features=self.args.datasource_classes_train),
                torch.nn.Softmax(dim=1)
            ]

            self.layers_classification = torch.nn.Sequential(
                *layers_classification
            )

        torch_utils.init_parameters(self.layers_embedding)

        for name, param in (list(self.layers_embedding.named_parameters()) + list(self.layers_embedding_scaler.named_parameters())):
            if param.requires_grad:
                if name.startswith('emb_') and 'bias' not in name:
                    if len(param.size()) > 1:
                        if args.embedding_init == 'xavier':
                            torch.nn.init.xavier_uniform_(param)
                        elif args.embedding_init == 'xavier_normal':
                            torch.nn.init.xavier_normal_(param)
                        elif args.embedding_init == 'uniform':
                            torch.nn.init.uniform_(param)
                        elif args.embedding_init == 'normal':
                            torch.nn.init.normal_(param)
                        elif args.embedding_init == 'zeros' or args.embedding_init == 'zero':
                            torch.nn.init.zeros_(param)
                        elif args.embedding_init == 'ones' or args.embedding_init == 'one':
                            torch.nn.init.ones_(param)


    def __create_layers_encoder(self, name, input_size):

        self.channels_conv_size = 1 #inout channels

        if self.args.model_encoder == 'resnet18':
            model_pretrained = torchvision.models.resnet18(pretrained=True)
        elif self.args.model_encoder == 'resnet34':
            model_pretrained = torchvision.models.resnet34(pretrained=True)
        elif self.args.model_encoder == 'resnet50':
            model_pretrained = torchvision.models.resnet50(pretrained=True)
        elif self.args.model_encoder == 'resnet101':
            model_pretrained = torchvision.models.resnet101(pretrained=True)
        elif self.args.model_encoder == 'densenet121':
            model_pretrained = torchvision.models.densenet121(pretrained=True)
        elif self.args.model_encoder == 'densenet161':
            model_pretrained = torchvision.models.densenet161(pretrained=True)

        if self.args.model_encoder.startswith('resnet'):

            if self.args.input_features != model_pretrained.conv1.in_channels:
                weight_conv1_pretrained = model_pretrained.conv1.weight.data
                model_pretrained.conv1 = torch.nn.Conv2d(self.args.input_features, model_pretrained.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)

                idx_rgb = 0
                for idx in range(self.args.input_features):
                    model_pretrained.conv1.weight.data[:, idx, :, :] = weight_conv1_pretrained[:, idx_rgb, :, :]
                    idx_rgb += 1
                    if idx_rgb == 2:
                        idx_rgb = 0

            modules = list(model_pretrained.children())[:-1] # drop FC
            layers_conv = torch.nn.Sequential(*modules)

            input_size = 1
            self.channels_conv_size = model_pretrained.fc.in_features
        elif self.args.model_encoder.startswith('densenet'):
            conv1 = list(model_pretrained.features.children())[0]
            if self.args.input_features != conv1.in_channels:
                weight_conv1_pretrained = conv1.weight.data
                conv1 = torch.nn.Conv2d(self.args.input_features, conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)

                idx_rgb = 0
                for idx in range(self.args.input_features):
                    conv1.weight.data[:, idx, :, :] = weight_conv1_pretrained[:, idx_rgb, :, :]
                    idx_rgb += 1
                    if idx_rgb == 2:
                        idx_rgb = 0

                is_first = True
                features = torch.nn.Sequential()
                for name, module in model_pretrained.features.named_children():
                    if is_first:
                        module = conv1
                        is_first = False
                    features.add_module(name, module)
                model_pretrained.features = features

            modules = list(model_pretrained.children())[:-1] # drop FC
            modules.append(torch.nn.AdaptiveAvgPool2d(output_size=1))
            layers_conv = torch.nn.Sequential(*modules)

            input_size = 1
            self.channels_conv_size = model_pretrained.classifier.in_features

            if not self.args.is_model_encoder_pretrained:
                torch_utils.init_parameters(layers_conv)

        return layers_conv, input_size

    def forward(self, x):
        # debug code
        # inp = x
        # for name, each in self.layers_encoder.named_children():
        #     print(f'{name} in: {inp.size()}')
        #     out = each.forward(inp)
        #     print(f'{name} out: {out.size()}')
        #     inp = out

        output_enc = self.layers_encoder.forward(x)

        output_emb_pre = self.layers_pre_embedding.forward(output_enc)

        output_emb = self.layers_embedding.forward(output_emb_pre)
        output_scaler = torch.sigmoid(self.layers_embedding_scaler.forward(output_emb_pre)) # scale vector 0..1

        norm = torch.norm(output_emb.detach(), p=2, dim=1, keepdim=True) # normalized direction vector
        output_vec = output_emb / norm

        output = output_vec * output_scaler

        output_norm = torch_utils.normalize_output(
            output,
            self.args.embedding_norm,
            self.args.embedding_scale)

        return output_norm

    def forward_with_classification(self, x):
        output_norm = self.forward(x)
        output_class = self.layers_classification.forward(output_norm)
        return output_norm, output_class
