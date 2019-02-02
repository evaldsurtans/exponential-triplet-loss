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

from torchvision.models import resnet50, resnet18, resnet34, resnet101, densenet121

from modules.block_resnet_2d_ext import ResBlock2D, ConvBlock2D
from modules.block_reshape import Reshape
from modules.dict_to_obj import DictToObj
import modules.torch_utils as torch_utils

# Added torchvision pre-trained models

class Model(torch.nn.Module):
    def __init__(self, args):

        super(Model, self).__init__()
        self.args = args

        if not hasattr(self.args, 'embedding_function'):
            self.args.embedding_function = 'tanh'

        layers_encoder, output_size = self.__create_layers_encoder(name='enc', input_size=args.input_size)
        self.layers_encoder = layers_encoder

        self.layers_suffix_affine = torch.nn.Sequential()
        for idx_layer in range(self.args.suffix_affine_layers):
            self.layers_suffix_affine.add_module(
                f'layers_suffix_affine_{idx_layer}',
                torch.nn.Linear(in_features=output_size, out_features=args.suffix_affine_layers_hidden, bias=False))
            self.layers_suffix_affine.add_module(f'layers_suffix_affine_relu_{idx_layer}',torch.nn.ReLU())
            output_size = args.suffix_affine_layers_hidden

        self.layers_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=output_size, out_features=self.args.embedding_size, bias=False)
        )

        if self.args.embedding_function == 'sigmoid':
            self.layers_embedding.add_module('emb_sigmoid', torch.nn.Sigmoid())
        elif self.args.embedding_function == 'tanh':
            self.layers_embedding.add_module('emb_tanh', torch.nn.Tanh())
        elif self.args.embedding_function == 'hard_tanh':
            self.layers_embedding.add_module('emb_hard_tanh', torch.nn.Hardtanh())
        elif self.args.embedding_function == 'hard_sink':
            self.layers_embedding.add_module('emb_hard_sink', torch.nn.Hardshrink())
        elif self.args.embedding_function == 'tanh_shrink':
            self.layers_embedding.add_module('emb_tanh_shrink', torch.nn.Tanhshrink())

        torch_utils.init_parameters(self)

    def __create_layers_encoder(self, name, input_size):

        if self.args.pre_type == 'resnet18':
            resnet = resnet18(pretrained=True)
        elif self.args.pre_type == 'resnet34':
            resnet = resnet34(pretrained=True)
        elif self.args.pre_type == 'resnet50':
            resnet = resnet50(pretrained=True)
        elif self.args.pre_type == 'densenet121':
            resnet = densenet121(pretrained=True)

        if not self.args.pre_type == 'densenet121':
            resnet.conv1 = torch.nn.Conv2d(1, 64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3), bias=False)

        for p in resnet.parameters():
            p.requires_grad = not self.args.is_pre_grad_locked

        if self.args.pre_type == 'densenet121':
            layers_conv = list(resnet.children())[0]
            flatten_conv_size = 1024
        else:
            modules = list(resnet.children())[:-2]
            layers_conv = torch.nn.Sequential(*modules)
            flatten_conv_size = 512
            if self.args.pre_type == 'resnet50':
                flatten_conv_size = 2048

        layers_conv.add_module(f'reshape_affine_{name}', Reshape(shape=flatten_conv_size))

        return layers_conv, flatten_conv_size

    def forward(self, x):
        # debug code
        # inp = x
        # for name, each in self.layers_encoder.named_children():
        #     print(f'{name} in: {inp.size()}')
        #     out = each.forward(inp)
        #     print(f'{name} out: {out.size()}')
        #     inp = out

        #x_upsampled = F.interpolate(x, size=(self.input_conv_size, self.input_conv_size), mode='bilinear')

        if self.args.pre_type == 'densenet121':
            x = x.view(-1, 1, self.args.input_size, self.args.input_size).repeat(1, 3, 1, 1).view(-1, 3, self.args.input_size, self.args.input_size)
        output_enc = self.layers_encoder.forward(x)

        output_sufix = self.layers_suffix_affine.forward(output_enc)
        output_emb = self.layers_embedding.forward(output_sufix)

        if self.args.embedding_norm == 'l2':
            norm = torch.norm(output_emb, p=2, dim=1, keepdim=True).detach()
            output_norm = output_emb / norm
        else:
            output_norm = output_emb

        return output_norm

