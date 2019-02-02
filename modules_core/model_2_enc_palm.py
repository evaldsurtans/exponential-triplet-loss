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

from modules.block_resnet_2d_std2 import ResBlock2D, ConvBlock2D, DeConvBlock2D, DeResBlock2D
from modules.block_reshape2 import Reshape
from modules.dict_to_obj import DictToObj
import modules.torch_utils as torch_utils

class Model(torch.nn.Module):
    def __init__(self, args):

        super(Model, self).__init__()
        self.args = args

        self.layers_encoder, output_size, flatten_size = self.__create_layers_encoder(name='enc', input_size=args.input_size, is_deconv=False)
        self.flatten_conv_size = flatten_size

        output_size = self.flatten_conv_size
        self.layers_affine = torch.nn.Sequential()
        for idx_layer in range(self.args.suffix_affine_layers):
            self.layers_affine.add_module(f'layers_affine_{idx_layer}', torch.nn.Linear(in_features=output_size, out_features=args.suffix_affine_layers_hidden))
            self.layers_affine.add_module(f'layers_affine_relu_{idx_layer}', torch.nn.ReLU())

            output_size = args.suffix_affine_layers_hidden

        self.layers_affine.add_module(f'layers_affine_back', torch.nn.Linear(in_features=output_size, out_features=self.flatten_conv_size))
        self.layers_affine.add_module(f'layers_affine_relu_back', torch.nn.ReLU())

        self.layers_decoder, output_size, flatten_size = self.__create_layers_encoder(name='dec', input_size=args.input_size, is_deconv=True)

        torch_utils.init_parameters(self)

    def __create_layers_encoder(self, name, input_size, is_deconv):

        pConvBlock2D = ConvBlock2D
        pResBlock2D = ResBlock2D
        prefix = 'conv'
        if is_deconv:
            prefix = 'deconv'
            pConvBlock2D = DeConvBlock2D
            pResBlock2D = DeResBlock2D

        self.channels_conv_size = 1 #inout channels
        layers_conv = []

        # non bottle necks stride 3, 5, 7 ; padding 1, 2, 3
        # bottle necks stride 4, 6, 8 ; padding 1, 2, 3

        for idx_layer in range(self.args.conv_resnet_layers):

            if idx_layer == 0:
                channels_conv_size_next = self.args.conv_first_channel_count
                kernel_size = self.args.conv_first_kernel + 1
                stride = 2
                layers_conv.append((f'{prefix}_{name}_init_{idx_layer}',
                                       pConvBlock2D(
                                           self.channels_conv_size, # for bottle necks even kernel size
                                           channels_conv_size_next,
                                           is_conv_bias=False,
                                           kernel_size=kernel_size,
                                           stride=stride
                                       )))
            else:
                channels_conv_size_next = int(self.channels_conv_size * self.args.conv_expansion_rate)
                channels_conv_size_next_input = channels_conv_size_next

                kernel_size = self.args.conv_kernel + 1
                stride = self.args.conv_stride
                if is_deconv:
                    if self.args.conv_unet == 'unet_add':
                        layers_conv.append((f'{prefix}_{name}_bottle_{idx_layer}_pre_bn',
                                            torch.nn.BatchNorm2d(self.channels_conv_size)
                                            ))
                    elif self.args.conv_unet == 'unet_cat':
                        channels_conv_size_next_input *= 2

                layers_conv.append((f'{prefix}_{name}_bottle_{idx_layer}',
                                       pResBlock2D(
                                           self.channels_conv_size,
                                           channels_conv_size_next_input,
                                           is_conv_bias=False,
                                           kernel_size=kernel_size,
                                           stride=stride
                                       )))
            self.channels_conv_size = channels_conv_size_next

            padding = int((kernel_size - 1)/2)

            # O = (W - K + 2P)/S + 1
            input_size_float = (input_size-kernel_size+2*padding)/stride + 1.0
            input_size = int(input_size_float)

            if self.args.is_quick_test:
                print(f'layer: {idx_layer} in:{input_size} padding:{padding} kernel:{kernel_size} stride:{stride} out:{input_size_float} input_size:{input_size}')

            if input_size_float - input_size != 0:
                logging.error(f'layer: {idx_layer} input_size not even: {input_size_float}')

            for idx_sub_layer in range(self.args.conv_resnet_sub_layers):
                layers_conv.append((f'{prefix}_{name}_{idx_layer}_{idx_sub_layer}',
                                       pResBlock2D(
                                           self.channels_conv_size,
                                           self.channels_conv_size,
                                           is_conv_bias=False,
                                           kernel_size=self.args.conv_kernel,
                                           stride=1
                                       )))

        flatten_conv_size = int(input_size * input_size * self.channels_conv_size)
        if is_deconv:
            flatten_conv_size = (self.channels_conv_size, input_size, input_size)
        layers_conv.append((f'{prefix}_reshape_affine_{name}', Reshape(shape=flatten_conv_size)))

        if is_deconv:
            layers_conv.reverse()

        layers_encoder = torch.nn.Sequential()
        for name, module in layers_conv:
            layers_encoder.add_module(name, module)

        return layers_encoder, input_size, flatten_conv_size

    def forward(self, x): #Batch, Channel, Width, Height

        u_net_residiuals = {}
        out = x

        for name, each in self.layers_encoder.named_children():
            if self.args.is_quick_test:
                print(f'encode: {name} in: {out.size()}')

            is_forward = True
            if self.args.conv_unet != 'none':
                if 'bottle' in name:
                    if len(out.size()) >= 3: # only conv
                        if self.args.is_quick_test:
                            print(f'u-net: {name} : {out.size()}')
                        name_dec = name.replace('conv_enc', 'deconv_dec')
                        if self.args.conv_unet == 'unet_add':
                            name_dec += '_pre_bn'

                        if self.args.conv_unet == 'unet_cat':
                            is_forward = False
                            out = each.forward(out)

                        u_net_residiuals[name_dec] = out

            if is_forward:
                out = each.forward(out)

            if self.args.is_quick_test:
                print(f'encode: {name} out: {out.size()}')

        out = self.layers_affine.forward(out)

        for name, each in self.layers_decoder.named_children():
            if self.args.is_quick_test:
                print(f'decode: {name} in: {out.size()}')

            is_forward = True
            if self.args.conv_unet != 'none':
                if name in u_net_residiuals.keys():
                    res = u_net_residiuals[name]
                    if self.args.is_quick_test:
                        print(f'u-net: {name} : {out.size()} / {res.size()}')

                    if self.args.conv_unet == 'unet_add':
                        out += res # next goes batch norm
                    elif self.args.conv_unet == 'unet_cat':
                        is_forward = False
                        out = torch.cat([out, res], dim=1)
                        out = each.forward(out)

            if is_forward:
                out = each.forward(out)

            if self.args.is_quick_test:
                print(f'decode: {name} out: {out.size()}')

        return out

