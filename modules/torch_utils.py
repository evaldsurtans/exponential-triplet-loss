import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import logging


def init_parameters(model):
    total_param_size = 0
    for name, param in model.named_parameters():
        each_param_size = np.prod(param.size())
        total_param_size += each_param_size
        logging.info('{} {} {}'.format(name, param.size(), each_param_size))

        if len(param.size()) > 1:
            if 'conv' in name and name.endswith('.weight'):
                torch.nn.init.kaiming_normal_(param)
            elif '.bn' in name and name.endswith('.weight'):
                torch.nn.init.constant(param, 1)
            elif 'bias' in name:
                param.data.zero_()
            else:
                torch.nn.init.xavier_normal_(param)
        else:
            if 'bias' in name:
                param.data.zero_()
            else:
                # elif 'bias' in name and ('rnn' in name):
                #     param.data.ones_()
                # else:
                torch.nn.init.normal_(param)

    logging.info(f'total_param_size: {total_param_size}')