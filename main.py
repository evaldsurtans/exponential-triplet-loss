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

from modules.model_rnn import RNN
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

parser.add_argument('-model', default='model_pink_skateboard', type=str)
parser.add_argument('-datasource', default='datasource_fassion_minst', type=str)
parser.add_argument('-triplet_sampler', default='triplet_sampler_hard', type=str)

parser.add_argument('-path_data', default='./data', type=str)
parser.add_argument('-datasource_workers', default=1, type=int) #8

parser.add_argument('-epochs_count', default=10, type=int)

parser.add_argument('-optimizer', default='adam', type=str)
parser.add_argument('-learning_rate', default=1e-5, type=float)
parser.add_argument('-batch_size', default=30, type=int)

parser.add_argument('-triplet_positives', default=3, type=int) # ensures batch will have 2 or 3 positives (for speaker_triplet_sampler_hard must have 3)
parser.add_argument('-triplet_loss', default='exp1', type=str)
parser.add_argument('-triplet_loss_margin', default=0.2, type=float)

parser.add_argument('-embedding_function', default='tanh', type=str)
parser.add_argument('-embedding_size', default=1024, type=int)

parser.add_argument('-is_split_affine_layer', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-split_affine_hidden', default=64, type=int)

parser.add_argument('-suffix_affine_layers', default=2, type=int)
parser.add_argument('-suffix_affine_layers_hidden', default=1024, type=int)

parser.add_argument('-conv_resnet_layers', default=5, type=int)
parser.add_argument('-conv_resnet_sub_layers', default=3, type=int)

parser.add_argument('-is_conv_bias', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-conv_first_channel_count', default=32, type=int) #kvass
parser.add_argument('-conv_first_kernel', default=9, type=int)
parser.add_argument('-conv_kernel', default=3, type=int)
parser.add_argument('-conv_stride', default=2, type=int)
parser.add_argument('-conv_expansion_rate', default=2, type=float) #kvass
parser.add_argument('-is_conv_max_pool', default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-early_stopping_patience', default=2, type=int)
parser.add_argument('-early_stopping_param', default='train_dist_delta', type=str)
parser.add_argument('-early_stopping_delta_percent', default=0.1, type=float)
parser.add_argument('-is_quick_test', default=True, type=lambda x: (str(x).lower() == 'true'))

args, args_other = parser.parse_known_args()

tmp = ['id', 'name', 'repeat_id', 'epoch', 'test_acc', 'test_eer', 'train_acc', 'train_eer', 'test_dist_delta', 'train_dist_delta', 'train_dist_positives', 'train_dist_negatives', 'test_dist_positives', 'test_dist_negatives', 'train_loss', 'test_loss', 'avg_epoch_time']
if not args.params_report is None:
    for it in args.params_report:
        if not it in tmp:
            tmp.append(it)
args.params_report = tmp

tmp = ['epoch', 'train_loss', 'test_loss', 'test_acc', 'test_eer', 'test_dist_delta', 'train_dist_delta', 'train_dist_positives', 'train_dist_neg', 'test_dist_pos', 'test_dist_neg', 'train_dist_pos_worst', 'epoch_time', 'early_percent_improvement']
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

TripletSampler = getattr(__import__('modules_core.' + args.triplet_sampler, fromlist=['TripletSampler']), 'TripletSampler')
triplet_sampler = TripletSampler(args)

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

    sampled = triplet_sampler.sample_batch(output, y)

    max_distance = 2.0
    margin_distance = (args.triplet_loss_margin * max_distance)

    if args.triplet_loss == 'exp1':
        loss = torch.mean(torch.exp(torch.clamp(sampled['positives_dist']-margin_distance, 0))) + \
               torch.mean(torch.exp(torch.clamp(max_distance-sampled['negatives_dist']-margin_distance, 0))) - 2.0

    result = dict(
        output=output,
        y=y,
        loss=loss
    )
    return {**result, **sampled}

state = {
    'epoch': 0,
    'best_param': -1,
    'avg_epoch_time': -1,
    'epoch_time': -1,
    'early_stopping_patience': 0,
    'early_percent_improvement': 0,
    'train_loss': -1,
    'test_loss': -1,
    'test_acc': -1,
    'train_acc': -1,
    'test_eer': -1,
    'train_eer': -1,
    'train_dist_positives': -1,
    'train_dist_negatives': -1,
    'test_dist_positives': -1,
    'test_dist_negatives': -1,
    'test_dist_delta': -1,
    'train_dist_delta': -1
}
avg_time_epochs = []
time_epoch = time.time()

CsvUtils.create_local(args)

meters = dict(
    train_loss = tnt.meter.AverageValueMeter(),
    test_loss = tnt.meter.AverageValueMeter(),

    test_acc = tnt.meter.ClassErrorMeter(accuracy=True),
    train_acc = tnt.meter.ClassErrorMeter(accuracy=True),

    test_auc = tnt.meter.AUCMeter(),
    train_auc = tnt.meter.AUCMeter(),

    train_dist_positives = tnt.meter.AverageValueMeter(),
    train_dist_negatives = tnt.meter.AverageValueMeter(),
    test_dist_positives = tnt.meter.AverageValueMeter(),
    test_dist_negatives = tnt.meter.AverageValueMeter(),
)

for epoch in range(1, args.epochs_count + 1):
    state_before = copy.deepcopy(state)
    logging.info('epoch: {} / {}'.format(epoch, args.epochs_count))

    for key in meters.keys():
        meters[key].reset()

    for data_loader in [data_loader_train, data_loader_test]:
        idx_quick_test = 0

        hist_positives_dist = []
        hist_negatives_dist = []

        output_embeddings = []
        output_y = []
        output_y_labels = []

        meter_prefix = 'train'
        if data_loader == data_loader_train:
            model = model.train()
        else:
            meter_prefix = 'test'
            model = model.eval()

        for batch in data_loader:
            optimizer_func.zero_grad()
            model.zero_grad()

            result = forward(batch)

            if data_loader == data_loader_train:
                result['loss'].backward()
                optimizer_func.step()

            meters[f'{meter_prefix}_loss'].add(np.average(result['loss'].data))

            avg_positives_dist_all = np.average(result['positives_dist_all'].data)
            avg_negatives_dist_all = np.average(result['negatives_dist_all'].data)
            hist_positives_dist.append(avg_positives_dist_all)
            hist_negatives_dist.append(avg_negatives_dist_all)

            meters[f'{meter_prefix}_dist_positives'].add(avg_positives_dist_all)
            meters[f'{meter_prefix}_dist_negatives'].add(avg_negatives_dist_all)

            output_embeddings += result['output'].data.numpy().tolist()
            y = result['y'].data.numpy().tolist()
            output_y += y
            output_y_labels += [data_loader_test.dataset.classes[it] for it in y]

            idx_quick_test += 1
            if args.is_quick_test and idx_quick_test >= 2:
                break

        predicted, target, target_y = CentroidClassificationUtils.calulate_classes(np.array(output_embeddings), np.array(output_y))
        tensorboard_utils.addHistogramsTwo(np.array(hist_positives_dist), np.array(hist_negatives_dist), f'hist_{meter_prefix}', epoch)
        tensorboard_writer.add_histogram(f'{meter_prefix}_dist_positives', np.array(hist_positives_dist), epoch, bins='doane')
        tensorboard_writer.add_histogram(f'{meter_prefix}_dist_negatives', np.array(hist_negatives_dist), epoch, bins='doane')

        meters[f'{meter_prefix}_acc'].add(predicted, target_y)

        tmp1 = predicted.permute(1, 0).data
        tmp2 = target.permute(1, 0).data
        meters[f'{meter_prefix}_auc'].add(tmp1[0], tmp2[0])

        tensorboard_writer.add_embedding(mat=torch.tensor(np.array(output_embeddings)), metadata=output_y_labels, global_step=epoch, tag=f'{meter_prefix}_embeddings')

        state[f'{meter_prefix}_acc'] = meters[f'{meter_prefix}_acc'].value()[0]
        fpr, tpr, eer = calc_err(meters[f'{meter_prefix}_auc'])
        state[f'{meter_prefix}_eer'] = eer

        state[f'{meter_prefix}_loss'] = meters[f'{meter_prefix}_loss'].value()[0]

        state[f'{meter_prefix}_dist_positives'] = meters[f'{meter_prefix}_dist_positives'].value()[0]
        state[f'{meter_prefix}_dist_negatives'] = meters[f'{meter_prefix}_dist_negatives'].value()[0]

        state[f'{meter_prefix}_dist_pos'] = meters[f'{meter_prefix}_dist_positives'].value()[0]
        state[f'{meter_prefix}_dist_neg'] = meters[f'{meter_prefix}_dist_negatives'].value()[0]

        state[f'{meter_prefix}_dist_delta'] = state[f'{meter_prefix}_dist_neg'] - state[f'{meter_prefix}_dist_pos']

        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_loss', scalar_value=state[f'{meter_prefix}_loss'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_dist_delta', scalar_value=state[f'{meter_prefix}_dist_delta'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_dist_positives', scalar_value=state[f'{meter_prefix}_dist_positives'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_dist_negatives', scalar_value=state[f'{meter_prefix}_dist_negatives'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_acc', scalar_value=state[f'{meter_prefix}_acc'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_eer', scalar_value=state[f'{meter_prefix}_eer'], global_step=epoch)

        tensorboard_utils.addPlot1D(
            data=(meters[f'{meter_prefix}_auc'].value()[2], meters[f'{meter_prefix}_auc'].value()[1]),
            tag=f'{meter_prefix}_auc',
            global_step=epoch,
            axis_labels=[
                'False positives',
                'True positives'
            ]
        )

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
            percent_improvement = (state[args.early_stopping_param] - state_before[args.early_stopping_param]) / state_before[args.early_stopping_param]
        if state[args.early_stopping_param] >= 0:
            if args.early_stopping_delta_percent > percent_improvement:
                state['early_stopping_patience'] += 1
            else:
                state['early_stopping_patience'] = 0
        state['early_percent_improvement'] = percent_improvement

    tensorboard_writer.add_scalar(tag='improvement', scalar_value=state['early_percent_improvement'], global_step=epoch)
    torch.save(model_module.state_dict(), os.path.join(run_path, f'{args.name}.pt'))

    logging.info(
        f'{round(percent * 100, 2)}% each: {round(state["avg_epoch_time"], 2)}min eta: {round(eta, 2)} min loss: {round(state["train_acc"], 2)} loss: {round(state["train_loss"], 2)} improve: {round(percent_improvement, 2)}')

    CsvUtils.add_results_local(args, state)
    CsvUtils.add_results(args, state)

    if state['early_stopping_patience'] >= args.early_stopping_patience or \
            math.isnan(percent_improvement) or \
            (percent_improvement == 0 and state['epoch'] > 1):
        logging_utils.info('early stopping')
        break

tensorboard_writer.close()