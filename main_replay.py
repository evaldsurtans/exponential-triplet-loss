import matplotlib
from torch.autograd import Variable

import rank_based
from modules.file_utils import FileUtils
from modules.torch_utils import to_numpy

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

parser.add_argument('-model', default='model_emb_pre_palm', type=str)
parser.add_argument('-pre_trained_model', default='./tasks/test_dec29_enc_123_123.json', type=str)
parser.add_argument('-is_pretrained_locked', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-unet_preloaded_pooling_size', default=1, type=int)

parser.add_argument('-datasource', default='datasource_pytorch', type=str)
parser.add_argument('-triplet_sampler', default='triplet_sampler_hard', type=str)

parser.add_argument('-path_data', default='./data', type=str)
parser.add_argument('-datasource_workers', default=8, type=int) #8
parser.add_argument('-datasource_type', default='mnist', type=str) # fassion_mnist mnist

parser.add_argument('-epochs_count', default=10, type=int)

parser.add_argument('-optimizer', default='adam', type=str)
parser.add_argument('-learning_rate', default=1e-5, type=float)
parser.add_argument('-batch_size', default=30, type=int)

parser.add_argument('-triplet_positives', default=3, type=int) # ensures batch will have 2 or 3 positives (for speaker_triplet_sampler_hard must have 3)
parser.add_argument('-triplet_loss', default='exp1', type=str)
parser.add_argument('-coef_loss_neg', default=1.0, type=float)
parser.add_argument('-triplet_loss_margin', default=0.2, type=float)
parser.add_argument('-is_triplet_loss_margin_auto', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-lossless_beta', default=2.0, type=float)

parser.add_argument('-embedding_function', default='tanh', type=str)
parser.add_argument('-embedding_size', default=32, type=int)
parser.add_argument('-embedding_norm', default='l1', type=str)

parser.add_argument('-embedding_layers', default=2, type=int)
parser.add_argument('-embedding_layers_hidden', default=512, type=int)

parser.add_argument('-suffix_affine_layers', default=2, type=int)
parser.add_argument('-suffix_affine_layers_hidden', default=1024, type=int)

parser.add_argument('-conv_resnet_layers', default=3, type=int)
parser.add_argument('-conv_resnet_sub_layers', default=3, type=int)

parser.add_argument('-is_conv_bias', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-conv_first_channel_count', default=32, type=int) #kvass
parser.add_argument('-conv_first_kernel', default=9, type=int)
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
parser.add_argument('-is_quick_test', default=False, type=lambda x: (str(x).lower() == 'true'))

args, args_other = parser.parse_known_args()

tmp = ['id', 'name', 'repeat_id', 'epoch', 'test_acc', 'test_acc2', 'test_eer', 'train_acc', 'train_acc2', 'train_eer', 'test_dist_delta', 'train_dist_delta', 'train_dist_positives', 'train_dist_negatives', 'test_dist_positives', 'test_dist_negatives', 'train_loss', 'test_loss', 'avg_epoch_time']
if not args.params_report is None:
    for it in args.params_report:
        if not it in tmp:
            tmp.append(it)
args.params_report = tmp

tmp = ['epoch', 'train_loss', 'test_loss', 'test_acc', 'test_acc2', 'test_eer', 'train_acc', 'train_acc2', 'train_eer', 'test_dist_delta', 'train_dist_delta', 'train_dist_positives', 'train_dist_neg', 'test_dist_pos', 'test_dist_neg', 'train_dist_pos_worst', 'epoch_time', 'early_percent_improvement']
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

# save model description (important for testing)
with open(os.path.join(run_path + f'/{args.name}.json'), 'w') as outfile:
    json.dump(args.__dict__, outfile, indent=4)

is_data_parallel = False
if args.device == 'cuda' and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, dim=0)
    is_data_parallel = True
    logging.info(f'PARALLEL MODEL {torch.cuda.device_count()}')

model = model.to(args.device)

experience_neg = rank_based.Experience(dict(
    batch_size=args.batch_size,
    beta_zero=0.5,
    alpha=0.7,
    replace_old=True,
    learn_start=0,
    size=data_loader_train.dataset.size_samples,
    total_steps=args.epochs_count * data_loader_train.dataset.size_samples
))

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

def forward(batch, is_train):
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

    # TODO get from replay half batch
    # TODO add calc of dists from replay
    # TODO update replay buffer values

    # TODO new store negatives from sampled
    # TODO select most negative from all given samples
    #if is_train:

    max_distance = 2.0
    if args.is_triplet_loss_margin_auto:
        margin_distance = max_distance / len(data_loader_train.dataset.classes)
    else:
        margin_distance = (args.triplet_loss_margin * max_distance)

    if args.triplet_loss == 'exp1':
        loss = torch.exp(torch.mean(torch.clamp(sampled['positives_dist']-margin_distance, 0))) + \
               torch.exp(torch.mean(torch.clamp(max_distance-sampled['negatives_dist']-margin_distance, 0))) - 2.0
    elif args.triplet_loss == 'exp1_neg_all':
        loss = torch.exp(torch.mean(torch.clamp(sampled['positives_dist']-margin_distance, 0))) + \
               torch.exp(torch.mean(torch.clamp(max_distance-sampled['negatives_dist_all']-margin_distance, 0))) - 2.0
    elif args.triplet_loss == 'exp_neg_all_only':
        loss = torch.exp(torch.mean(torch.clamp(max_distance-sampled['negatives_dist_all']-margin_distance, 0))) - 1.0
    elif args.triplet_loss == 'exp2':
        loss = torch.exp(torch.mean(torch.clamp(sampled['positives_dist']-margin_distance, 0))) + \
               torch.exp(torch.mean(torch.clamp(max_distance-sampled['negatives_dist']-margin_distance, 0))) - 2.0
    elif args.triplet_loss == 'exp2_coef_neg_all':
        loss = (torch.exp(torch.mean(torch.clamp(sampled['positives_dist']-margin_distance, 0))) - 1.0) + \
               args.coef_loss_neg * (torch.exp(torch.mean(torch.clamp(max_distance-sampled['negatives_dist_all']-margin_distance, 0))) - 1.0)
    if args.triplet_loss == 'exp2_coef':
        loss = (torch.exp(torch.mean(torch.clamp(sampled['positives_dist']-margin_distance, 0))) - 1.0) + \
               args.coef_loss_neg * (torch.exp(torch.mean(max_distance-sampled['negatives_dist'])) - 1.0)
    elif args.triplet_loss == 'standard':
        delta = sampled['positives_dist'] - sampled['negatives_dist'] + margin_distance
        loss = torch.mean(torch.clamp(delta, min=0))
    elif args.triplet_loss == 'standard2':
        delta = torch.mean(sampled['positives_dist']) - torch.mean(sampled['negatives_dist'] + margin_distance)
        loss = torch.clamp(delta, min=0)
    elif args.triplet_loss == 'standard2_all':
        delta = torch.mean(sampled['positives_dist']) - torch.mean(sampled['negatives_dist_all'] + margin_distance)
        loss = torch.clamp(delta, min=0)
    elif args.triplet_loss == 'lossless':
        loss = -torch.mean(torch.log10(1e-7 + 1.0 - sampled['positives_dist'] / args.lossless_beta)) - torch.mean(torch.log10(1e-7 + 1.0 - (2.0 - sampled['negatives_dist']) / args.lossless_beta))
    elif args.triplet_loss == 'lifted':
        # https://arxiv.org/pdf/1511.06452.pdf
        delta = torch.mean(sampled['positives_dist']) + torch.log(torch.mean(torch.exp(margin_distance - sampled['positives_dist'])) + torch.mean(torch.exp(margin_distance - sampled['negatives_dist'])))
        loss = torch.clamp(delta, min=0)
    elif args.triplet_loss == 'lifted2':
        delta = torch.mean(sampled['positives_dist']) + torch.log(torch.sum(torch.exp(margin_distance - sampled['positives_dist'])) + torch.sum(torch.exp(margin_distance - sampled['negatives_dist'])))
        loss = torch.clamp(delta, min=0)

    result = dict(
        output=output,
        y=y,
        loss=loss,
        x=x
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
    'test_auc': -1,
    'train_auc': -1,
    'test_acc2': -1,
    'train_acc2': -1,
    'test_auc2': -1,
    'train_auc2': -1,
    'test_eer': -1,
    'train_eer': -1,
    'test_eer2': -1,
    'train_eer2': -1,
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

    test_acc2 = tnt.meter.ClassErrorMeter(accuracy=True),
    train_acc2 = tnt.meter.ClassErrorMeter(accuracy=True),

    test_auc = tnt.meter.AUCMeter(),
    train_auc = tnt.meter.AUCMeter(),

    test_auc2 = tnt.meter.AUCMeter(),
    train_auc2 = tnt.meter.AUCMeter(),

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
        output_y_images = []

        meter_prefix = 'train'
        if data_loader == data_loader_train:
            model = model.train()
        else:
            meter_prefix = 'test'
            model = model.eval()
            torch.set_grad_enabled(False)

        for batch in data_loader:
            optimizer_func.zero_grad()
            torch.set_grad_enabled(True)

            result = forward(batch, is_train=(data_loader == data_loader_train))

            if data_loader == data_loader_train:
                result['loss'].backward(retain_graph=True)
                optimizer_func.step()

            meters[f'{meter_prefix}_loss'].add(np.median(to_numpy(result['loss'])))

            avg_positives_dist_all = np.median(to_numpy(result['positives_dist_all']))
            avg_negatives_dist_all = np.median(to_numpy(result['negatives_dist_all']))
            hist_positives_dist.append(avg_positives_dist_all)
            hist_negatives_dist.append(avg_negatives_dist_all)

            meters[f'{meter_prefix}_dist_positives'].add(avg_positives_dist_all)
            meters[f'{meter_prefix}_dist_negatives'].add(avg_negatives_dist_all)

            output_embeddings += to_numpy(result['output'].to('cpu')).tolist()
            output_y_images += to_numpy(result['x']).tolist()
            y = to_numpy(result['y']).tolist()
            output_y += y
            output_y_labels += [data_loader_test.dataset.classes[it] for it in y]

            idx_quick_test += 1
            if args.is_quick_test and idx_quick_test >= 2:
                break

        histogram_bins = 'auto'
        #histogram_bins = 100
        tensorboard_utils.addHistogramsTwo(np.array(hist_positives_dist), np.array(hist_negatives_dist), f'hist_{meter_prefix}', epoch)
        tensorboard_writer.add_histogram(f'{meter_prefix}_dist_positives', np.array(hist_positives_dist), epoch, bins=histogram_bins)
        tensorboard_writer.add_histogram(f'{meter_prefix}_dist_negatives', np.array(hist_negatives_dist), epoch, bins=histogram_bins)

        predicted, target, target_y = CentroidClassificationUtils.calulate_classes(np.array(output_embeddings), np.array(output_y), type='range')

        meters[f'{meter_prefix}_acc'].add(predicted, target_y)

        tmp1 = predicted.permute(1, 0).data
        tmp2 = target.permute(1, 0).data
        meters[f'{meter_prefix}_auc'].add(tmp1[0], tmp2[0])

        predicted, target, target_y = CentroidClassificationUtils.calulate_classes(np.array(output_embeddings), np.array(output_y), type='closest')

        meters[f'{meter_prefix}_acc2'].add(predicted, target_y)

        tmp1 = predicted.permute(1, 0).data
        tmp2 = target.permute(1, 0).data
        meters[f'{meter_prefix}_auc2'].add(tmp1[0], tmp2[0])

        # label_img: :math:`(N, C, H, W)
        tensorboard_writer.add_embedding(
            mat=torch.tensor(np.array(output_embeddings)),
            label_img=torch.FloatTensor(np.array(output_y_images)),
            metadata=output_y_labels,
            global_step=epoch, tag=f'{meter_prefix}_embeddings')

        state[f'{meter_prefix}_acc'] = meters[f'{meter_prefix}_acc'].value()[0]
        fpr, tpr, eer = calc_err(meters[f'{meter_prefix}_auc'])
        state[f'{meter_prefix}_eer'] = eer

        state[f'{meter_prefix}_acc2'] = meters[f'{meter_prefix}_acc2'].value()[0]
        fpr, tpr, eer = calc_err(meters[f'{meter_prefix}_auc2'])
        state[f'{meter_prefix}_eer2'] = eer

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
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_acc2', scalar_value=state[f'{meter_prefix}_acc2'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_eer2', scalar_value=state[f'{meter_prefix}_eer2'], global_step=epoch)

        tensorboard_utils.addPlot1D(
            data=(meters[f'{meter_prefix}_auc'].value()[2], meters[f'{meter_prefix}_auc'].value()[1]),
            tag=f'{meter_prefix}_auc',
            global_step=epoch,
            axis_labels=[
                'False positives',
                'True positives'
            ]
        )
        tensorboard_utils.addPlot1D(
            data=(meters[f'{meter_prefix}_auc2'].value()[2], meters[f'{meter_prefix}_auc2'].value()[1]),
            tag=f'{meter_prefix}_auc2',
            global_step=epoch,
            axis_labels=[
                'False positives',
                'True positives'
            ]
        )

        if data_loader == data_loader_train:
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
        f'{round(percent * 100, 2)}% each: {round(state["avg_epoch_time"], 2)} min eta: {round(eta, 2)} min acc: {round(state["train_acc"], 2)} loss: {round(state["train_loss"], 2)} improve: {round(percent_improvement, 2)}')

    CsvUtils.add_results_local(args, state)
    CsvUtils.add_results(args, state)

    if state['early_stopping_patience'] >= args.early_stopping_patience or \
            math.isnan(percent_improvement) or \
            (percent_improvement == 0 and state['epoch'] > 1):
        logging_utils.info('early stopping')
        break


tensorboard_writer.close()