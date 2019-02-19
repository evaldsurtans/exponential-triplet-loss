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

from modules.block_resnet_2d_leaky import ResBlock2D, ConvBlock2D
from modules.block_reshape import Reshape
from modules.dict_to_obj import DictToObj
import modules.torch_utils as torch_utils

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

        if args.embedding_layers > 0:
            input_size_emb = self.channels_conv_size * output_size**2
            self.layers_embedding = torch.nn.Sequential(
                Reshape(shape=input_size_emb)
            )
            for idx in range(args.embedding_layers):
                output_size_emb = args.embedding_layers_hidden
                if idx == args.embedding_layers - 1:
                    output_size_emb = args.embedding_size

                self.layers_embedding.add_module(f'linear_{idx}', torch.nn.Linear(input_size_emb, output_size_emb, bias=False))
                if idx < args.embedding_layers - 1:
                    self.layers_embedding.add_module(f'relu_linear_{idx}', torch.nn.LeakyReLU(negative_slope=self.args.leaky_relu_slope))
                input_size_emb = output_size_emb
        else:
            # refined part pooling version
            # https://arxiv.org/pdf/1711.09349.pdf
            self.layers_embedding = torch.nn.Sequential(
                torch.nn.BatchNorm2d(num_features=self.channels_conv_size),
                torch.nn.AdaptiveAvgPool2d(output_size=1),
                torch.nn.LeakyReLU(negative_slope=self.args.leaky_relu_slope),
                torch.nn.Conv2d(kernel_size=1, stride=1, in_channels=self.channels_conv_size, out_channels=self.args.embedding_size, bias=False),
                Reshape(shape=self.args.embedding_size)
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

        self.channels_conv_size = 1 #inout channels
        layers_conv = torch.nn.Sequential()
        for idx_layer in range(self.args.conv_resnet_layers):

            if idx_layer == 0:
                channels_conv_size_next = self.args.conv_first_channel_count
                kernel_size = self.args.conv_first_kernel
                stride = 2
                layers_conv.add_module(f'conv_{name}_init_{idx_layer}',
                                       ConvBlock2D(
                                           in_channels=self.channels_conv_size,
                                           out_channels=channels_conv_size_next,
                                           is_conv_bias=False,
                                           kernel_size=kernel_size,
                                           stride=stride
                                       ))
            else:
                channels_conv_size_next = int(self.channels_conv_size * self.args.conv_expansion_rate)
                kernel_size = self.args.conv_kernel
                stride = self.args.conv_stride
                layers_conv.add_module(f'conv_{name}_bottle_{idx_layer}',
                                       ResBlock2D(
                                           in_channels=self.channels_conv_size,
                                           out_channels=channels_conv_size_next,
                                           is_conv_bias=False,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           leaky_relu_slope=self.args.leaky_relu_slope
                                       ))
            self.channels_conv_size = channels_conv_size_next

            padding = int(kernel_size/2)
            input_size = int((input_size+2*padding-(kernel_size-1)-1)/stride + 1)

            for idx_sub_layer in range(self.args.conv_resnet_sub_layers):
                layers_conv.add_module(f'conv_{name}_{idx_layer}_{idx_sub_layer}',
                                       ResBlock2D(
                                           in_channels=self.channels_conv_size,
                                           out_channels=self.channels_conv_size,
                                           is_conv_bias=False,
                                           kernel_size=self.args.conv_kernel,
                                           stride=1,
                                           leaky_relu_slope=self.args.leaky_relu_slope
                                       ))

            if self.args.is_conv_max_pool:
                stride = 2
                layers_conv.add_module(f'maxpool_{name}_{idx_layer}',
                                            torch.nn.MaxPool2d(kernel_size=kernel_size, padding=1, stride=stride))
                input_size = int((input_size+2*padding-(kernel_size-1)-1)/stride + 1)

        layers_conv.add_module(f'last_bn_{name}', torch.nn.BatchNorm2d(self.channels_conv_size))
        layers_conv.add_module(f'last_relu_{name}', torch.nn.LeakyReLU(negative_slope=self.args.leaky_relu_slope))

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

        output_emb = self.layers_embedding.forward(output_enc)

        output_norm = torch_utils.normalize_output(output_emb, self.args.embedding_norm)

        return output_norm

