import matplotlib

from modules.file_utils import FileUtils

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import torch
import torch.nn
import torch.utils.data
import tensorboardX
import numpy as np
import time
import argparse
from datetime import datetime
import torch.nn.functional as F
import json
import unicodedata
import string
import glob
import io
import random
import math
import matplotlib.ticker as ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import copy
import sklearn
import sklearn.model_selection
import shutil
import json
import logging

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from modules.tensorboard_utils import TensorBoardUtils
from modules.logging_utils import LoggingUtils
from modules.data_utils import DataUtils
from modules.args_utils import ArgsUtils
from modules.csv_utils import CsvUtils
from modules.dict_to_obj import DictToObj
from modules.command_txt_utils import CommandTxtUtils
from modules.mathplotlib_utils import MathPlotLibUtils
from modules.centroid_classification_utils import CentroidClassificationUtils

import torchnet as tnt # pip install git+https://github.com/pytorch/tnt.git@master
import traceback, sys

parser = argparse.ArgumentParser(description='Model trainer')

parser.add_argument('-id', default=0, type=int)
parser.add_argument('-repeat_id', default=0, type=int)
parser.add_argument('-report', default='report', type=str)
parser.add_argument('-params_report', nargs='*', required=False) # extra params for report global for series of runs
parser.add_argument('-params_report_local', nargs='*', required=False) # extra params for local run report
parser.add_argument('-name', help='Run name, by default date', default='', type=str)

parser.add_argument('-is_datasource_only', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-device', default='cuda', type=str)

parser.add_argument('-model', default='model_enc_palm', type=str)
parser.add_argument('-datasource', default='datasource_pytorch', type=str)

parser.add_argument('-triplet_positives', default=1, type=int) # in order to use same datasource as for triplet loss
parser.add_argument('-path_data', default='./data', type=str)
parser.add_argument('-datasource_workers', default=8, type=int) #8
parser.add_argument('-datasource_type', default='minst', type=str) # fassion_minst minst

parser.add_argument('-epochs_count', default=10, type=int)

parser.add_argument('-optimizer', default='adam', type=str)
parser.add_argument('-learning_rate', default=1e-5, type=float)
parser.add_argument('-batch_size', default=30, type=int)

parser.add_argument('-tensorboard_batches_debug', default=2, type=int)

parser.add_argument('-suffix_affine_layers', default=2, type=int)
parser.add_argument('-suffix_affine_layers_hidden', default=1024, type=int)

parser.add_argument('-conv_resnet_layers', default=2, type=int)
parser.add_argument('-conv_resnet_sub_layers', default=3, type=int)

parser.add_argument('-is_conv_bias', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-conv_first_channel_count', default=32, type=int) #kvass
parser.add_argument('-conv_first_kernel', default=7, type=int)
parser.add_argument('-conv_kernel', default=3, type=int)
parser.add_argument('-conv_stride', default=2, type=int)
parser.add_argument('-conv_expansion_rate', default=2, type=float) #kvass
parser.add_argument('-is_conv_max_pool', default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-conv_unet', default='unet_add', type=str) # none, unet_add, unet_cat

parser.add_argument('-early_stopping_patience', default=5, type=int)
parser.add_argument('-early_stopping_param', default='train_loss', type=str)
parser.add_argument('-early_stopping_param_coef', default=-1.0, type=float)
parser.add_argument('-early_stopping_delta_percent', default=0.05, type=float)

parser.add_argument('-is_reshuffle_after_epoch', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-is_quick_test', default=True, type=lambda x: (str(x).lower() == 'true'))

args, args_other = parser.parse_known_args()

tmp = ['id', 'name', 'repeat_id', 'epoch', 'train_loss', 'test_loss', 'avg_epoch_time']
if not args.params_report is None:
    for it in args.params_report:
        if not it in tmp:
            tmp.append(it)
args.params_report = tmp

tmp = ['epoch', 'train_loss', 'test_loss', 'epoch_time', 'early_percent_improvement']
if not args.params_report_local is None:
    for it in args.params_report_local:
        if not it in tmp:
            tmp.append(it)
args.params_report_local = tmp


if len(args.name) == 0:
    args.name = datetime.now().strftime('%y-%m-%d_%H-%M-%S')

FileUtils.createDir('./tasks/' + args.report)
run_path = './tasks/' + args.report + '/runs/' + args.name
if os.path.exists(run_path):
    shutil.rmtree(run_path, ignore_errors=True)
    time.sleep(3)
    while os.path.exists(run_path):
        pass

tensorboard_writer = tensorboardX.SummaryWriter(log_dir=run_path)
tensorboard_utils = TensorBoardUtils(tensorboard_writer)
logging_utils = LoggingUtils(filename=os.path.join(run_path, 'log.txt'))

get_data_loaders = getattr(__import__('modules_core.' + args.datasource, fromlist=['get_data_loaders']), 'get_data_loaders')
data_loader_train, data_loader_test = get_data_loaders(args)

ArgsUtils.log_args(args, 'main.py', logging_utils)

if args.is_datasource_only:
    logging.info('is_datasource_only laoded')
    exit()

if not torch.cuda.is_available():
    args.device = 'cpu'
    logging.info('CUDA NOT AVAILABLE')
else:
    logging.info('cuda devices: {}'.format(torch.cuda.device_count()))

Model = getattr(__import__('modules_core.' + args.model, fromlist=['Model']), 'Model')
model = Model(args)

# save model description (important for testing)
with open(os.path.join(run_path + f'/{args.name}.json'), 'w') as outfile:
    json.dump(args.__dict__, outfile, indent=4)

is_data_parallel = False
if args.device == 'cuda' and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, dim=0)
    is_data_parallel = True
    logging.info(f'PARALLEL MODEL {torch.cuda.device_count()}')

model = model.to(args.device)

optimizer_func = None
if args.optimizer == 'adam':
    optimizer_func = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )
elif args.optimizer == 'rmsprop':
    optimizer_func = torch.optim.RMSprop(
        model.parameters(),
        lr=args.learning_rate
    )


def calc_err(meter):
    fpr, tpr, thresholds = roc_curve(meter.targets, meter.scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return fpr, tpr, eer


def forward(batch):
    y = batch[0].to(args.device)
    x = batch[1].to(args.device)

    model_module = model
    if is_data_parallel:
        model_module = model.module

    if hasattr(model_module, 'init_hidden'):
        hidden = model_module.init_hidden(batch_size=x.size(0)) #! Important // cannot be used with data parallel
        output = model.forward(x, hidden)
    else:
        output = model.forward(x)

    # F.smooth_l1_loss
    loss = F.mse_loss(output, x)

    result = dict(
        output=output,
        y=y,
        loss=loss,
        x=x
    )
    return result

state = {
    'epoch': 0,
    'best_param': -1,
    'avg_epoch_time': -1,
    'epoch_time': -1,
    'early_stopping_patience': 0,
    'early_percent_improvement': 0,
    'train_loss': -1,
    'test_loss': -1
}
avg_time_epochs = []
time_epoch = time.time()

CsvUtils.create_local(args)

meters = dict(
    train_loss = tnt.meter.AverageValueMeter(),
    test_loss = tnt.meter.AverageValueMeter()
)
count_tensorboard_batches_debug = 0

for epoch in range(1, args.epochs_count + 1):
    state_before = copy.deepcopy(state)
    logging.info('epoch: {} / {}'.format(epoch, args.epochs_count))

    for key in meters.keys():
        meters[key].reset()

    for data_loader in [data_loader_train, data_loader_test]:
        idx_quick_test = 0

        meter_prefix = 'train'
        if data_loader == data_loader_train:
            model = model.train()
            torch.set_grad_enabled(True)
        else:
            meter_prefix = 'test'
            model = model.eval()
            torch.set_grad_enabled(False)
            count_tensorboard_batches_debug = 0

        for batch in data_loader:
            optimizer_func.zero_grad()
            model.zero_grad()

            result = forward(batch)

            if data_loader == data_loader_train:
                result['loss'].backward()
                optimizer_func.step()

            meters[f'{meter_prefix}_loss'].add(np.median(result['loss'].to('cpu').data))

            if count_tensorboard_batches_debug < args.tensorboard_batches_debug:
                for idx in range(batch[0].size(0)):
                    idx_full = count_tensorboard_batches_debug * batch[0].size(0) + idx
                    tensorboard_utils.addPlot2D(dataXY=batch[1][idx].to('cpu').data.numpy().squeeze(), tag=f'{meter_prefix}_{idx_full}_x', global_step=epoch)
                    tensorboard_utils.addPlot2D(dataXY=result['output'][idx].to('cpu').data.numpy().squeeze(), tag=f'{meter_prefix}_{idx_full}_out', global_step=epoch)
                count_tensorboard_batches_debug += 1

            idx_quick_test += 1
            if args.is_quick_test and idx_quick_test >= 2:
                break

        state[f'{meter_prefix}_loss'] = meters[f'{meter_prefix}_loss'].value()[0]

        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_loss', scalar_value=state[f'{meter_prefix}_loss'], global_step=epoch)

        if args.is_reshuffle_after_epoch:
            # after every epoch reshuffle triplets for better data mining
            data_loader.dataset.reshuffle()

    model_module = model
    if is_data_parallel:
        model_module = model.module

    if epoch == 1:
        state['best_param'] = state[args.early_stopping_param]
    elif state['best_param'] > state[args.early_stopping_param]:
        state['best_param'] = state[args.early_stopping_param]
        torch.save(model_module.state_dict(), os.path.join(run_path, 'best.pt'))

    epoch_time = (time.time() - time_epoch) / 60.0
    percent = epoch / args.epochs_count
    state['epoch_time'] = epoch_time

    avg_time_epochs.append(epoch_time)
    state['avg_epoch_time'] = np.average(avg_time_epochs)
    eta = ((args.epochs_count - epoch) * state['avg_epoch_time'])
    time_epoch = time.time()
    state['epoch'] = epoch

    # early stopping
    percent_improvement = 0
    if epoch > 1:
        if state_before[args.early_stopping_param] != 0:
            percent_improvement = args.early_stopping_param_coef * (state[args.early_stopping_param] - state_before[args.early_stopping_param]) / state_before[args.early_stopping_param]
        if state[args.early_stopping_param] >= 0:
            if args.early_stopping_delta_percent > percent_improvement:
                state['early_stopping_patience'] += 1
            else:
                state['early_stopping_patience'] = 0
        state['early_percent_improvement'] = percent_improvement

    tensorboard_writer.add_scalar(tag='improvement', scalar_value=state['early_percent_improvement'], global_step=epoch)
    torch.save(model_module.state_dict(), os.path.join(run_path, f'{args.name}.pt'))

    logging.info(
        f'{round(percent * 100, 2)}% each: {round(state["avg_epoch_time"], 2)} min eta: {round(eta, 2)} min loss: {round(state["train_loss"], 2)} improve: {round(percent_improvement, 2)}')

    CsvUtils.add_results_local(args, state)
    CsvUtils.add_results(args, state)

    if state['early_stopping_patience'] >= args.early_stopping_patience or \
            math.isnan(percent_improvement) or \
            (percent_improvement == 0 and state['epoch'] > 1):
        logging_utils.info('early stopping')
        break


tensorboard_writer.close()