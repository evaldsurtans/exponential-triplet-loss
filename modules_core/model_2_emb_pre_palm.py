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

        with open(args.pre_trained_model, 'r') as fp:
            self.pre_args = DictToObj(**json.load(fp))

        self.size_extra_from_decoder = 0
        self.layers_decoder = torch.nn.Sequential()
        self.layers_encoder, output_size, flatten_size = self.__create_layers_encoder(name='enc', input_size=self.pre_args.input_size, is_deconv=False)
        self.flatten_conv_size = flatten_size

        output_size = self.flatten_conv_size
        self.layers_affine = torch.nn.Sequential()
        for idx_layer in range(self.args.suffix_affine_layers):
            self.layers_affine.add_module(f'layers_affine_{idx_layer}', torch.nn.Linear(in_features=output_size, out_features=self.pre_args.suffix_affine_layers_hidden))
            self.layers_affine.add_module(f'layers_affine_relu_{idx_layer}', torch.nn.ReLU())

            output_size = self.pre_args.suffix_affine_layers_hidden

        self.layers_affine.add_module(f'layers_affine_back', torch.nn.Linear(in_features=output_size, out_features=self.flatten_conv_size))
        self.layers_affine.add_module(f'layers_affine_relu_back', torch.nn.ReLU())

        self.layers_embedding = torch.nn.Sequential()
        input_size = output_size + self.size_extra_from_decoder
        for idx_layer in range(self.args.embedding_layers):
            if idx_layer == self.args.embedding_layers - 1:
                output_size = self.args.embedding_size
            else:
                output_size = self.args.embedding_layers_hidden
            torch.nn.Linear(in_features=input_size, out_features=output_size, bias=False),
            input_size = output_size

        if self.args.embedding_function == 'sigmoid':
            self.layers_embedding.add_module('emb_sigmoid', torch.nn.Sigmoid())
        elif self.args.embedding_function == 'tanh':
            self.layers_embedding.add_module('emb_tanh', torch.nn.Tanh())
        # or none

        torch_utils.init_parameters(self)

        dict_model = torch.load(args.pre_trained_model[:-len('.json')] + '.pt', map_location='cpu')
        self.load_state_dict(dict_model, strict=False)

        if self.args.is_pretrained_locked:
            for name, param in list(self.layers_encoder.named_parameters()) + list(self.layers_decoder.named_parameters()) + list(self.layers_affine.named_parameters()):
                param.requires_grad = False

        logging.info(f'prloaded: {args.pre_trained_model}')

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

        for idx_layer in range(self.pre_args.conv_resnet_layers):

            if idx_layer == 0:
                channels_conv_size_next = self.pre_args.conv_first_channel_count
                kernel_size = self.pre_args.conv_first_kernel + 1
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
                channels_conv_size_next = int(self.channels_conv_size * self.pre_args.conv_expansion_rate)
                channels_conv_size_next_input = channels_conv_size_next

                kernel_size = self.pre_args.conv_kernel + 1
                stride = self.pre_args.conv_stride

                if self.pre_args.conv_unet == 'unet_add':
                    self.size_extra_from_decoder += self.channels_conv_size * self.args.unet_preloaded_pooling_size ** 2
                    self.layers_decoder.add_module(f'{prefix}_{name}_bottle_{idx_layer}_pre_bn',
                                        torch.nn.BatchNorm2d(self.channels_conv_size)
                                        )
                elif self.pre_args.conv_unet == 'unet_cat':
                    self.size_extra_from_decoder += channels_conv_size_next_input * self.args.unet_preloaded_pooling_size ** 2

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

            if self.pre_args.is_quick_test:
                print(f'layer: {idx_layer} in:{input_size} padding:{padding} kernel:{kernel_size} stride:{stride} out:{input_size_float} input_size:{input_size}')

            if input_size_float - input_size != 0:
                logging.error(f'layer: {idx_layer} input_size not even: {input_size_float}')

            for idx_sub_layer in range(self.pre_args.conv_resnet_sub_layers):
                layers_conv.append((f'{prefix}_{name}_{idx_layer}_{idx_sub_layer}',
                                       pResBlock2D(
                                           self.channels_conv_size,
                                           self.channels_conv_size,
                                           is_conv_bias=False,
                                           kernel_size=self.pre_args.conv_kernel,
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

        u_net_residiuals = []
        out = x

        for name, each in self.layers_encoder.named_children():
            if self.args.is_quick_test:
                print(f'encode: {name} in: {out.size()}')

            is_forward = True
            if self.pre_args.conv_unet != 'none':
                if 'bottle' in name:
                    out_size = out.size()
                    if len(out_size) >= 3: # only conv
                        if self.args.is_quick_test:
                            print(f'u-net: {name} : {out.size()}')
                        if self.pre_args.conv_unet == 'unet_add':
                            name += '_pre_bn'
                            for name_layer, layer in self.layers_decoder.named_children():
                                if name_layer == name:
                                    res = layer.forward(out)

                        if self.pre_args.conv_unet == 'unet_cat':
                            is_forward = False
                            out = each.forward(out)
                            res = out

                        res = F.adaptive_avg_pool2d(res, self.args.unet_preloaded_pooling_size)
                        u_net_residiuals.append(res.view(-1, res.size()[1] * self.args.unet_preloaded_pooling_size ** 2)) # batch, channel

            if is_forward:
                out = each.forward(out)

            if self.args.is_quick_test:
                print(f'encode: {name} out: {out.size()}')

        out = self.layers_affine.forward(out)
        if len(u_net_residiuals):
            out = torch.cat([out] + u_net_residiuals, dim=1)
        output_emb = self.layers_embedding.forward(out)

        output_norm = torch_utils.normalize_output(output_emb, self.args.embedding_norm)

        return output_norm

