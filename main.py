import matplotlib
from torch.autograd import Variable

from modules.file_utils import FileUtils
from modules.math_utils import normalize_vec
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

parser.add_argument('-is_datasource_only', default=False, type=lambda x: (str(x).lower() == 'true')) # just test datasource
parser.add_argument('-device', default='cuda', type=str)

parser.add_argument('-model', default='model_12_dobe', type=str)
parser.add_argument('-model_encoder', default='resnet18', type=str)
parser.add_argument('-is_model_encoder_pretrained', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-layers_embedding_type', default='last', type=str) #refined, pooled, last

parser.add_argument('-pre_trained_model', default='./tasks/test_dec29_enc_123_123.json', type=str)
parser.add_argument('-is_pretrained_locked', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-unet_preloaded_pooling_size', default=1, type=int)

parser.add_argument('-datasource', default='datasource_pytorch', type=str)
parser.add_argument('-datasource_is_grayscale', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-datasource_classes_train', default=0, type=int)

parser.add_argument('-is_class_loss', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-class_loss_coef', default=1e-1, type=float) # 0 means adapative coeficient

parser.add_argument('-triplet_sampler', default='triplet_sampler_4', type=str)
parser.add_argument('-triplet_sampler_var', default='hard', type=str) # hard, all
parser.add_argument('-filter_samples', nargs='*', default=['none']) # abs_margin semi_hard hard
parser.add_argument('-triplet_similarity', default='cos', type=str) # cos euclidean
parser.add_argument('-embedding_norm', default='unit_range', type=str) #unit_range l2 none

parser.add_argument('-path_data', default='./data', type=str)
parser.add_argument('-datasource_workers', default=8, type=int) #8
parser.add_argument('-datasource_type', default='cifar_10', type=str) # fassion_minst minst
parser.add_argument('-datasource_exclude_train_class_ids', nargs='*', default=[])
parser.add_argument('-datasource_include_test_class_ids', nargs='*', default=[])
parser.add_argument('-datasource_size_samples', default=0, type=int) # 0 automatic

parser.add_argument('-epochs_count', default=20, type=int)

parser.add_argument('-optimizer', default='adam', type=str)
parser.add_argument('-learning_rate', default=1e-5, type=float)
parser.add_argument('-weight_decay', default=0, type=float)
parser.add_argument('-batch_size', default=114, type=int)

parser.add_argument('-triplet_positives', default=3, type=int) # ensures batch will have 2 or 3 positives (for speaker_triplet_sampler_hard must have 3)
parser.add_argument('-triplet_loss', default='exp13', type=str)
parser.add_argument('-coef_loss_neg', default=1.0, type=float)
parser.add_argument('-triplet_loss_margin', default=0.2, type=float)
parser.add_argument('-is_triplet_loss_margin_auto', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-lossless_beta', default=1.2, type=float)

parser.add_argument('-exp_coef', default=2.0, type=float)
parser.add_argument('-overlap_coef', default=1.2, type=float)

parser.add_argument('-abs_coef', default=1.5, type=float)
parser.add_argument('-tan_coef', default=20.0, type=float)
parser.add_argument('-sin_coef', default=20.0, type=float)
parser.add_argument('-kl_coef', default=1e-3, type=float)

parser.add_argument('-slope_coef', default=3.0, type=float)
parser.add_argument('-neg_coef', default=2.0, type=float)
parser.add_argument('-pos_coef', default=3.0, type=float)

parser.add_argument('-neg_loss_coef', default=0.0, type=float)
parser.add_argument('-pos_loss_coef', default=0.0, type=float)

parser.add_argument('-is_center_loss', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-center_loss_min_count', default=100, type=int)
parser.add_argument('-center_loss_coef', default=0.0, type=float)

parser.add_argument('-is_kl_loss', default=False, type=lambda x: (str(x).lower() == 'true')) #TODO

parser.add_argument('-embedding_function', default='tanh', type=str)
parser.add_argument('-embedding_size', default=32, type=int)

parser.add_argument('-embedding_layers', default=0, type=int)
parser.add_argument('-embedding_layers_hidden', default=512, type=int)
parser.add_argument('-embedding_layers_hidden_func', default='maxout', type=str)
parser.add_argument('-embedding_layers_last_norm', default='instance', type=str) # none instance batch layer local

parser.add_argument('-suffix_affine_layers', default=2, type=int)
parser.add_argument('-suffix_affine_layers_hidden', default=1024, type=int)

parser.add_argument('-conv_resnet_layers', default=3, type=int)
parser.add_argument('-conv_resnet_sub_layers', default=3, type=int)

parser.add_argument('-is_pre_grad_locked', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-pre_type', default='resnet34', type=str)

parser.add_argument('-is_conv_bias', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-conv_first_channel_count', default=32, type=int) #kvass
parser.add_argument('-conv_first_kernel', default=9, type=int)
parser.add_argument('-conv_kernel', default=3, type=int)
parser.add_argument('-conv_stride', default=2, type=int)
parser.add_argument('-conv_expansion_rate', default=2, type=float) #kvass
parser.add_argument('-is_conv_max_pool', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-is_linear_at_end', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-leaky_relu_slope', default=0.1, type=float)

parser.add_argument('-conv_unet', default='unet_add', type=str) # none, unet_add, unet_cat

parser.add_argument('-suffix_affine_layers_hidden_func', default='relu', type=str) #kaf maxout relu lin
parser.add_argument('-suffix_affine_layers_hidden_params', default=8, type=int)

parser.add_argument('-early_stopping_patience', default=3, type=int)
parser.add_argument('-early_stopping_param', default='test_loss', type=str)
parser.add_argument('-early_stopping_param_coef', default=-1.0, type=float)
parser.add_argument('-early_stopping_delta_percent', default=0.01, type=float)

parser.add_argument('-is_reshuffle_after_epoch', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-is_quick_test', default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-max_embeddings_histograms', default=1000, type=int)
parser.add_argument('-max_embeddings_per_class_test', default=500, type=int)
parser.add_argument('-max_embeddings_per_class_train', default=500, type=int)

parser.add_argument('-max_embeddings_projector_classes', default=50, type=int)
parser.add_argument('-max_embeddings_projector_samples', default=200, type=int)

args, args_other = parser.parse_known_args()

tmp = [
    'id',
    'name',
    'repeat_id',
    'epoch',
    'test_acc_range',
    'test_acc_closest',
    'best_acc_range',
    'best_acc_closest',
    'test_eer',
    'train_acc_range',
    'train_acc_closest',
    'train_eer',
    'test_dist_delta',
    'test_dist_positives',
    'test_dist_negatives',
    'train_dist_delta',
    'train_dist_positives',
    'train_dist_negatives',
    'test_dist_delta_hard',
    'test_dist_positives_hard',
    'test_dist_negatives_hard',
    'train_dist_delta_hard',
    'train_dist_positives_hard',
    'train_dist_negatives_hard',
    'test_count_positives',
    'test_count_negatives',
    'train_count_positives',
    'train_count_negatives',
    'test_count_positives_all',
    'test_count_negatives_all',
    'train_count_positives_all',
    'train_count_negatives_all',
    'test_negative_max',
    'train_negative_max',
    'test_max_dist',
    'train_max_dist',
    'test_loss',
    'train_loss',
    'train_loss_emb',
    'train_loss_class',
    'train_loss_pos',
    'train_loss_neg',
    'test_loss_pos',
    'test_loss_neg',
    'train_loss_center',
    'test_loss_center',
    'avg_epoch_time']
if not args.params_report is None:
    for it in args.params_report:
        if not it in tmp:
            tmp.append(it)
args.params_report = tmp

tmp = [
    'epoch',
    'test_acc_range',
    'test_acc_closest',
    'best_acc_range',
    'best_acc_closest',
    'test_eer',
    'train_acc_range',
    'train_acc_closest',
    'train_eer',
    'test_dist_delta',
    'test_dist_positives',
    'test_dist_negatives',
    'train_dist_delta',
    'train_dist_positives',
    'train_dist_negatives',
    'test_dist_delta_hard',
    'test_dist_positives_hard',
    'test_dist_negatives_hard',
    'train_dist_delta_hard',
    'train_dist_positives_hard',
    'train_dist_negatives_hard',
    'test_count_positives',
    'test_count_negatives',
    'train_count_positives',
    'train_count_negatives',
    'test_count_positives_all',
    'test_count_negatives_all',
    'train_count_positives_all',
    'train_count_negatives_all',
    'test_negative_max',
    'train_negative_max',
    'test_max_dist',
    'train_max_dist',
    'test_loss',
    'train_loss',
    'train_loss_emb',
    'train_loss_pos',
    'train_loss_neg',
    'test_loss_pos',
    'test_loss_neg',
    'train_loss_class',
    'train_loss_center',
    'test_loss_center',
    'epoch_time',
    'early_percent_improvement']
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
is_logged_cnorm = False

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

optimizer_func = None
if args.optimizer == 'adam':
    optimizer_func = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
elif args.optimizer == 'rmsprop':
    optimizer_func = torch.optim.RMSprop(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )


def calc_err(meter):
    fpr, tpr, eer = -1, -1, -1
    try:
        fpr, tpr, thresholds = roc_curve(meter.targets, meter.scores)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
    except Exception as e:
        logging.error(str(e))
        exc_type, exc_value, exc_tb = sys.exc_info()
        logging.error('\n'.join(traceback.format_exception(exc_type, exc_value, exc_tb)))
    return fpr, tpr, eer


def forward(batch, output_by_y, is_train):
    global is_logged_cnorm

    y = batch[0].to(args.device)
    x = batch[1].to(args.device)

    model_module = model
    if is_data_parallel:
        model_module = model.module

    output = None
    output_class = None
    if is_train:
        if hasattr(model_module, 'forward_with_classification'):
            output, output_class = model.forward_with_classification(x)

    if output is None:
        if hasattr(model_module, 'init_hidden'):
            hidden = model_module.init_hidden(batch_size=x.size(0)) #! Important // cannot be used with data parallel
            output = model.forward(x, hidden)
        else:
            output = model.forward(x)

    max_distance = 2.0 # cosine distance

    if args.is_triplet_loss_margin_auto:
        margin_distance = args.overlap_coef * max_distance / args.datasource_classes_train
        if args.triplet_similarity == 'cos':
            margin_distance *= 2.0
    else:
        margin_distance = args.triplet_loss_margin

    K = args.datasource_classes_train
    C_norm = args.overlap_coef/K
    if args.triplet_similarity == 'cos':
        C_norm *= 2.0

    # in case of exp mining we should use normalized margin from C_norm
    if 'abs_margin' in args.filter_samples or 'abs_margin_asym' in args.filter_samples:
        margin_distance = C_norm

    sampled = triplet_sampler.sample_batch(output, y, margin_distance, max_distance)

    if not is_logged_cnorm:
        is_logged_cnorm = True
        logging.info(f'K/classes_train: {K} C_norm: {C_norm} max_distance: {max_distance}')

    pos_norm = sampled['positives_dist'] / max_distance
    neg_norm = sampled['negatives_dist'] / max_distance

    loss_pos = 0
    loss_neg = 0

    if args.triplet_loss == 'exp13':
        eps = 1e-20
        loss_pos_inner = torch.mean(torch.clamp(pos_norm - C_norm, 0.0))
        loss_neg_inner = torch.mean(torch.clamp(0.5 - neg_norm, 0.0))

        pos_loss_coef = args.pos_loss_coef
        neg_loss_coef = args.neg_loss_coef

        if pos_loss_coef == 0 and neg_loss_coef == 0:
            # automatic balancing of coefs
            if loss_pos_inner > 0:
                # adaptive coeficient
                pos_loss_coef = loss_neg_inner.detach() / loss_pos_inner.detach()
                neg_loss_coef = 1.0

        loss_pos = -torch.log(1.0 - (loss_pos_inner/(1.-C_norm)) + eps) * pos_loss_coef
        loss_neg = -torch.log(1.0 - (loss_neg_inner/0.5) + eps) * neg_loss_coef
        loss = loss_pos + loss_neg
    elif args.triplet_loss == 'exp12':
        eps = 1e-20
        loss_pos = torch.mean(torch.clamp(pos_norm - C_norm, 0.0))
        loss_neg = torch.mean(torch.clamp(0.5 - neg_norm, 0.0))
        loss = -torch.log(1.0 - (loss_pos/(1.-C_norm+args.pos_coef)) + args.pos_coef + eps) + \
               -torch.log(1.0 - (loss_neg/(0.5+args.neg_coef)) + args.neg_coef + eps)
    elif args.triplet_loss == 'exp11':
        loss_pos = torch.mean(torch.clamp(pos_norm - C_norm, 0.0))
        loss_neg = torch.mean(torch.clamp(0.5 - neg_norm, 0.0))
        loss = torch.exp(args.neg_coef * loss_neg) + torch.exp(args.pos_coef * loss_pos) - 2.0
    elif args.triplet_loss == 'exp10':
        loss_pos = torch.mean(torch.clamp(pos_norm - C_norm, 0.0))
        loss_neg = torch.mean(torch.clamp(0.5 - neg_norm, 0.0))
        loss = loss_pos**args.pos_coef + loss_neg**args.neg_coef
    elif args.triplet_loss == 'exp9':
        pos = sampled['positives_dist'] / max_distance
        neg = sampled['negatives_dist'] / max_distance

        loss_pos = torch.mean(torch.clamp(pos - C_norm, 0.0) ** args.exp_coef)
        loss_neg = torch.mean(torch.clamp(0.5 - neg, 0.0) ** args.exp_coef)
        loss = (loss_pos + loss_neg)**(1.0/args.exp_coef)
    elif args.triplet_loss == 'exp8d':
        pos = sampled['positives_dist'] / max_distance
        neg = sampled['negatives_dist'] / max_distance

        loss_pos = torch.mean(args.pos_coef * torch.clamp(pos - C_norm, 0.0))
        loss_neg = torch.mean(args.neg_coef * torch.clamp(0.5 - neg, 0.0))
        loss = loss_pos + loss_neg
    elif args.triplet_loss == 'exp8e':
        pos = sampled['positives_dist'] / max_distance
        neg = sampled['negatives_dist'] / max_distance

        loss_pos = torch.mean(args.slope_coef * torch.exp(torch.clamp(pos - C_norm, 0.0))) - 1.0
        loss_neg = torch.mean(args.neg_coef * torch.exp(args.slope_coef * torch.clamp(0.5 - neg, 0.0))) - 1.0
        loss = loss_pos + loss_neg
    elif args.triplet_loss == 'exp8c':
        pos = sampled['positives_dist'] / max_distance
        neg = sampled['negatives_dist'] / max_distance

        loss_pos = torch.mean(args.slope_coef * torch.exp(pos)) - 1.0
        loss_neg = torch.mean(torch.exp(args.slope_coef * args.neg_coef * torch.clamp(0.5 - neg, 0.0))) - 1.0
        loss = loss_pos + loss_neg
    elif args.triplet_loss == 'exp8b':
        pos = sampled['positives_dist'] / max_distance
        neg = sampled['negatives_dist'] / max_distance

        loss_pos = torch.mean(args.slope_coef * torch.exp(pos)) - 1.0
        loss_neg = args.neg_coef * torch.mean(torch.exp(args.slope_coef * 2.0 * torch.clamp(0.5 - neg, 0.0))) - args.neg_coef
        loss = loss_pos + loss_neg
    elif args.triplet_loss == 'exp7':
        pi_k = K * 2 - 1
        if args.triplet_similarity == 'cos':
            pi_k = K - 1
        C_limit = 1.5*C_norm

        pos = sampled['positives_dist'] / max_distance
        neg = sampled['negatives_dist'] / max_distance

        indexes_neg_valid = []
        for idx in range(neg.size(0)):
            if neg[idx] < C_limit:
                indexes_neg_valid.append(idx)

        loss_pos = torch.mean(torch.tan(args.tan_coef * torch.clamp(pos - C_norm, 0.0)))

        loss_neg = 0
        if len(indexes_neg_valid) > 0:
            indexes_neg_valid = torch.LongTensor(indexes_neg_valid).to(args.device)
            neg = torch.index_select(neg, dim=0, index=indexes_neg_valid)
            loss_neg = torch.mean(torch.sin(pi_k*np.pi*neg - np.pi*1.5) * args.sin_coef + args.sin_coef)

        loss = loss_pos + loss_neg

    elif args.triplet_loss == 'exp5':
        C = max_distance / len(data_loader_train.dataset.classes)
        O = C * args.overlap_coef
        pos = sampled['positives_dist']
        neg = sampled['negatives_dist']

        delta_neg = (0.5*C + O)
        indexes_abs_part = []
        indexes_exp_part = []
        for idx in range(neg.size(0)):
            if delta_neg - neg[idx] < 0:
                indexes_exp_part.append(idx)
            else:
                indexes_abs_part.append(idx)

        coef_neg_abs = 1.0
        coef_neg_exp = 1.0
        if len(indexes_exp_part) > 0:
            coef_neg_abs = len(indexes_abs_part) / len(indexes_exp_part)
        else:
            coef_neg_exp = 0.0

        if len(indexes_abs_part) > 0:
            coef_neg_exp = len(indexes_exp_part) / len(indexes_abs_part)
        else:
            coef_neg_abs = 0.0

        indexes_abs_part = torch.LongTensor(indexes_abs_part).to(args.device)
        indexes_exp_part = torch.LongTensor(indexes_exp_part).to(args.device)

        loss_pos = torch.exp(2.0 * torch.mean(torch.clamp(pos - O, 0.0))) - 1.0
        if coef_neg_abs > 0:
            loss_neg_abs = 30.0 * torch.mean(torch.clamp(delta_neg - torch.index_select(neg, dim=0, index=indexes_abs_part) - O*0.5, 0.0))
        else:
            loss_neg_abs = 0
        if coef_neg_exp > 0:
            loss_neg_exp = torch.exp(2.0 * torch.mean(torch.clamp((delta_neg - torch.index_select(neg, dim=0, index=indexes_exp_part)) * -1.0 - O*0.5, 0.0))) - 1.0
        else:
            loss_neg_exp = 0
        loss = loss_pos + coef_neg_abs * loss_neg_abs + coef_neg_exp * loss_neg_exp

    if args.triplet_loss == 'exp4':
        pos_norm = sampled['positives_dist']/max_distance
        neg_norm = sampled['negatives_dist']/max_distance
        margin = 1.0 / len(data_loader_train.dataset.classes)
        radius_cluster = margin * args.overlap_coef / 2.0
        loss = torch.exp(args.exp_coef * torch.mean(torch.clamp(pos_norm-radius_cluster*2.0, 0.0))) + \
            torch.exp(args.exp_coef * torch.mean(torch.clamp(torch.abs(neg_norm - margin) - radius_cluster, 0.0))) - 2.0
    elif args.triplet_loss == 'exp2':
        loss = (torch.exp(args.exp_coef *
                          torch.mean(
                              torch.clamp((sampled['positives_dist']/max_distance)-margin_distance, 0))) - 1.0) + \
               (args.coef_loss_neg * (torch.exp(args.exp_coef *
                                               torch.mean(
                                                   torch.clamp(((max_distance-sampled['negatives_dist'])/max_distance)-margin_distance, 0))) - 1.0))

    elif args.triplet_loss == 'exp2_pairs':
        loss = torch.mean((torch.exp(args.exp_coef *
                                     torch.clamp((sampled['positives_dist']/max_distance)-margin_distance, 0)) - 1.0) +
                          (args.coef_loss_neg * (torch.exp(args.exp_coef *
                                                          torch.clamp(((max_distance-sampled['negatives_dist'])/max_distance)-margin_distance, 0)) - 1.0)))

    elif args.triplet_loss == 'exp2_neg_all':
        loss = (torch.exp(args.exp_coef *
                          torch.mean(
                              torch.clamp((sampled['positives_dist']/max_distance)-margin_distance, 0))) - 1.0) + \
               (args.coef_loss_neg * (torch.exp(args.exp_coef *
                                               torch.mean(
                                                   torch.clamp(((max_distance-sampled['negatives_dist_all_filtred'])/max_distance)-margin_distance, 0))) - 1.0))

    elif args.triplet_loss == 'exp3_neg_all':
        loss = torch.exp(args.exp_coef *
                          torch.mean(
                              torch.clamp(((max_distance-sampled['negatives_dist_all_filtred'])/max_distance)-margin_distance, 0))) - 1.0

    # todo fix mean
    elif args.triplet_loss == 'ratio':
        # exp_coef for cosine distance - could be dynamic here
        # max 1.0 => 4.0
        # 2.0 => 2.0
        # 4.0 => 1.0
        pos_part = torch.exp(args.exp_coef * torch.mean(sampled['positives_dist']))
        neg_part = torch.exp(args.exp_coef * torch.mean(sampled['negatives_dist']))
        div_part = pos_part + neg_part
        loss = (pos_part/div_part)**2 + (1.0 - (neg_part/div_part)**2)

    # todo fix mean
    elif args.triplet_loss == 'ratio_neg_all':
        # exp_coef for cosine distance - could be dynamic here
        # max 1.0 => 4.0
        # 2.0 => 2.0
        # 4.0 => 1.0
        pos_part = torch.exp(args.exp_coef * torch.mean(sampled['positives_dist']))
        neg_part = torch.exp(args.exp_coef * torch.mean(sampled['negatives_dist_all_filtred']))
        div_part = pos_part + neg_part
        loss = (pos_part/div_part)**2 + (1.0 - (neg_part/div_part)**2)

    elif args.triplet_loss == 'smooth5':
        loss = F.smooth_l1_loss(torch.clamp(sampled['positives_dist']-margin_distance, 0), torch.zeros(sampled['positives_dist'].size()).to(args.device), reduction='mean') + \
                 args.coef_loss_neg * F.smooth_l1_loss(torch.clamp(max_distance-sampled['negatives_dist']-margin_distance, 0), torch.zeros(sampled['negatives_dist'].size()).to(args.device), reduction='mean')
    elif args.triplet_loss == 'smooth5_neg_all':
        loss = (torch.exp(F.smooth_l1_loss(sampled['positives_dist'], torch.zeros(sampled['positives_dist'].size()).to(args.device) * margin_distance, reduction='mean')) - 1.0) + \
               args.coef_loss_neg * (torch.exp(F.smooth_l1_loss(max_distance-sampled['negatives_dist_all_filtred'], torch.zeros(sampled['negatives_dist_all_filtred'].size()).to(args.device) * margin_distance, reduction='mean')) - 1.0)

    elif args.triplet_loss == 'smooth6':
        loss = F.smooth_l1_loss(sampled['positives_dist'], torch.ones(sampled['positives_dist'].size()).to(args.device) * margin_distance, reduction='mean') + \
                 args.coef_loss_neg * F.smooth_l1_loss(max_distance-sampled['negatives_dist'], torch.ones(sampled['negatives_dist'].size()).to(args.device) * margin_distance, reduction='mean')
    elif args.triplet_loss == 'smooth6_neg_all':
        loss = F.smooth_l1_loss(sampled['positives_dist'], torch.ones(sampled['positives_dist'].size()).to(args.device) * margin_distance, reduction='mean') + \
                 args.coef_loss_neg * F.smooth_l1_loss(max_distance-sampled['negatives_dist_all_filtred'], torch.ones(sampled['negatives_dist_all_filtred'].size()).to(args.device) * margin_distance, reduction='mean')

    elif args.triplet_loss == 'standard':
        delta = sampled['positives_dist'] - sampled['negatives_dist'] + margin_distance
        #print(f'samples: {delta.size(0)}')
        if delta.size(0) == 0: # no valid triplets mined
            loss = None
        else:
            loss = torch.mean(torch.clamp(delta, min=0))
    elif args.triplet_loss == 'standard_neg_all':
        delta = sampled['positives_dist'] - sampled['negatives_dist_all_filtred'] + margin_distance
        loss = torch.mean(torch.clamp(delta, min=0))
    elif args.triplet_loss == 'lossless':
        pos = torch.mean(sampled['positives_dist'] / max_distance)
        neg = torch.mean(sampled['negatives_dist'] / max_distance)
        b = args.lossless_beta
        e = 1e-20
        loss = -torch.log10(-pos/b + 1.0 +e) - torch.log10(-(1.0-neg)/b + 1 + e)

    loss_emb = loss.detach().clone()
    loss_center = None
    if args.is_center_loss:
        centers = []
        outputs_by_centers = []
        cached_centers_by_y = {}
        list_y = to_numpy(y).tolist()

        # output_by_y => all centers from current epoch
        for idx, each_y in enumerate(list_y):
            if each_y in output_by_y.keys():
                if each_y not in cached_centers_by_y.keys():
                    if len(output_by_y[each_y]['embeddings']) > args.center_loss_min_count:
                        cached_centers_by_y[each_y] = np.average(output_by_y[each_y]['embeddings'], axis=0)
                        if args.embedding_norm == 'l2': #TODO unit_range
                            cached_centers_by_y[each_y] = normalize_vec(cached_centers_by_y[each_y])
                        cached_centers_by_y[each_y] = torch.FloatTensor(cached_centers_by_y[each_y]).to(args.device)
                if each_y in cached_centers_by_y.keys():
                    centers.append(cached_centers_by_y[each_y])
                    outputs_by_centers.append(output[idx])

        if len(centers) > 0:
            centers = torch.stack(centers).to(args.device)
            outputs_by_centers = torch.stack(outputs_by_centers).to(args.device)
            if args.triplet_similarity == 'cos':
                centers_dist = 1. - F.cosine_similarity(centers, outputs_by_centers, dim=1, eps=1e-20) # -1 .. 1 => 0 .. 2
            else:
                centers_dist = F.pairwise_distance(centers, outputs_by_centers, eps=1e-20) # 0 .. 2

            if args.is_center_loss:
                if args.triplet_loss == 'exp13':
                    eps = 1e-20
                    loss_center_inner = torch.mean(torch.clamp(pos_norm - C_norm, 0.0))
                    loss_center = -torch.log(1.0 - (loss_center_inner/(1.-C_norm)) + eps)
                else:
                    loss_center = torch.clamp(centers_dist - C_norm, 0.0)

                center_loss_coef = args.center_loss_coef
                if center_loss_coef <= 0.0:
                    center_loss_coef = 0.0
                    if loss_center > 0:
                        # adaptive coeficient
                        center_loss_coef = loss_emb.detach() / loss_center.detach()
                loss_center = loss_center * center_loss_coef
                loss += loss_center

    loss_class = None
    if is_train and args.is_class_loss:
        if output_class is not None:
            y_hot_enc = y.to('cpu').reshape(y.size(0), 1)
            tmp = torch.arange(args.datasource_classes_train).reshape(1, args.datasource_classes_train)
            y_hot_enc = (y_hot_enc == tmp).float() # one hot encoded
            y_hot_enc = y_hot_enc.detach().to(args.device)

            loss_class = torch.mean(-y_hot_enc*torch.log(output_class))
            class_loss_coef = args.class_loss_coef
            if class_loss_coef <= 0.0:
                class_loss_coef = 0.0
                if loss_class > 0:
                    # adaptive coeficient
                    class_loss_coef = loss_emb.detach() / loss_class.detach()
            loss_class = loss_class * class_loss_coef
            loss += loss_class

    result = dict(
        output=output,
        output_class=output_class,
        y=y,
        loss_center=prep_loss_for_stats(loss_center),
        loss_pos=prep_loss_for_stats(loss_pos),
        loss_neg=prep_loss_for_stats(loss_neg),
        loss_class=prep_loss_for_stats(loss_class),
        loss_emb=prep_loss_for_stats(loss_emb),
        loss=loss,
        x=x
    )
    return {**result, **sampled}


def prep_loss_for_stats(loss_val):
    if loss_val is not None:
        if isinstance(loss_val, torch.Tensor):
            loss_val = loss_val.item()
    return loss_val

state = {
    'epoch': 0,
    'best_param': -1,
    'avg_epoch_time': -1,
    'epoch_time': -1,
    'early_stopping_patience': 0,
    'early_percent_improvement': 0,
    'train_loss': -1,
    'test_loss': -1,
    'train_loss_class': -1,
    'test_loss_class': -1,
    'train_loss_emb': -1,
    'test_loss_emb': -1,
    'train_loss_center': -1,
    'test_loss_center': -1,
    'train_loss_pos': -1,
    'test_loss_pos': -1,
    'train_loss_neg': -1,
    'test_loss_neg': -1,
    'test_acc_range': -1,
    'best_acc_range': -1,
    'train_acc_range': -1,
    'test_auc_range': -1,
    'train_auc_range': -1,
    'test_acc_closest': -1,
    'best_acc_closest': -1,
    'train_acc_closest': -1,
    'test_auc_closest': -1,
    'train_auc_closest': -1,
    'test_max_dist': -1,
    'train_max_dist': -1,
    'test_eer': -1,
    'train_eer': -1,
    'test_eer2': -1,
    'train_eer2': -1,
    'train_dist_positives': -1,
    'train_dist_negatives': -1,
    'test_dist_positives': -1,
    'test_dist_negatives': -1,
    'test_dist_delta': -1,
    'train_dist_delta': -1,
    'train_dist_positives_hard': -1,
    'train_dist_negatives_hard': -1,
    'test_dist_positives_hard': -1,
    'test_dist_negatives_hard': -1,
    'test_dist_delta_hard': -1,
    'train_dist_delta_hard': -1,
    'train_count_positives': -1,
    'train_count_negatives': -1,
    'test_count_positives': -1,
    'test_count_negatives': -1,
    'train_count_positives_all': -1,
    'train_count_negatives_all': -1,
    'test_count_positives_all': -1,
    'test_count_negatives_all': -1,
    'test_negative_max': -1,
    'train_negative_max': -1,
}
avg_time_epochs = []
time_epoch = time.time()

CsvUtils.create_local(args)

meters = dict(
    train_loss = tnt.meter.AverageValueMeter(),
    test_loss = tnt.meter.AverageValueMeter(),

    train_loss_emb = tnt.meter.AverageValueMeter(),
    test_loss_emb = tnt.meter.AverageValueMeter(),

    train_loss_center = tnt.meter.AverageValueMeter(),
    test_loss_center = tnt.meter.AverageValueMeter(),

    train_loss_pos = tnt.meter.AverageValueMeter(),
    test_loss_pos = tnt.meter.AverageValueMeter(),

    train_loss_neg = tnt.meter.AverageValueMeter(),
    test_loss_neg = tnt.meter.AverageValueMeter(),

    train_loss_class = tnt.meter.AverageValueMeter(),
    test_loss_class = tnt.meter.AverageValueMeter(),

    test_acc_range = tnt.meter.ClassErrorMeter(accuracy=True),
    train_acc_range = tnt.meter.ClassErrorMeter(accuracy=True),

    test_acc_closest = tnt.meter.ClassErrorMeter(accuracy=True),
    train_acc_closest = tnt.meter.ClassErrorMeter(accuracy=True),

    test_auc_range = tnt.meter.AUCMeter(),
    train_auc_range = tnt.meter.AUCMeter(),

    test_auc_closest = tnt.meter.AUCMeter(),
    train_auc_closest = tnt.meter.AUCMeter(),

    train_dist_positives = tnt.meter.AverageValueMeter(),
    train_dist_negatives = tnt.meter.AverageValueMeter(),
    test_dist_positives = tnt.meter.AverageValueMeter(),
    test_dist_negatives = tnt.meter.AverageValueMeter(),

    train_count_positives = tnt.meter.AverageValueMeter(),
    train_count_negatives = tnt.meter.AverageValueMeter(),
    test_count_positives = tnt.meter.AverageValueMeter(),
    test_count_negatives = tnt.meter.AverageValueMeter(),

    train_count_positives_all = tnt.meter.AverageValueMeter(),
    train_count_negatives_all = tnt.meter.AverageValueMeter(),
    test_count_positives_all = tnt.meter.AverageValueMeter(),
    test_count_negatives_all = tnt.meter.AverageValueMeter(),

    train_dist_positives_hard = tnt.meter.AverageValueMeter(),
    train_dist_negatives_hard = tnt.meter.AverageValueMeter(),
    test_dist_positives_hard = tnt.meter.AverageValueMeter(),
    test_dist_negatives_hard = tnt.meter.AverageValueMeter(),
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
        hist_positives_dist_hard = []
        hist_negatives_dist_hard = []

        output_by_y = {}

        meter_prefix = 'train'
        if data_loader == data_loader_train:
            model = model.train()
            torch.set_grad_enabled(True)
        else:
            meter_prefix = 'test'
            model = model.eval()
            torch.set_grad_enabled(False)

        negative_max = 0

        for batch in data_loader:
            optimizer_func.zero_grad()
            is_train = (data_loader == data_loader_train)

            result = forward(batch, output_by_y, is_train)

            if is_train:
                if result['loss'] is not None:
                    if result['loss'].grad_fn is not None: # in case all samples in batch does not require training
                        result['loss'].backward()
                        optimizer_func.step()

            if result['loss'] is not None:
                meters[f'{meter_prefix}_loss'].add(np.median(to_numpy(result['loss'])))
            if result['loss_class'] is not None:
                meters[f'{meter_prefix}_loss_class'].add(np.median(to_numpy(result['loss_class'])))
            if result['loss_emb'] is not None:
                meters[f'{meter_prefix}_loss_emb'].add(np.median(to_numpy(result['loss_emb'])))
            if result['loss_pos'] is not None:
                meters[f'{meter_prefix}_loss_pos'].add(np.median(to_numpy(result['loss_pos'])))
            if result['loss_neg'] is not None:
                meters[f'{meter_prefix}_loss_neg'].add(np.median(to_numpy(result['loss_neg'])))
            if result['loss_center'] is not None:
                meters[f'{meter_prefix}_loss_center'].add(np.median(to_numpy(result['loss_center'])))

            if args.is_quick_test:
                print(f"count pos:{float(result['positives_dist'].size(0))} neg:{float(result['negatives_dist'].size(0))}")
                print(f"count all pos:{float(result['positives_dist_all_filtred'].size(0))} neg:{float(result['negatives_dist_all_filtred'].size(0))}")

            meters[f'{meter_prefix}_count_positives'].add(float(result['positives_dist'].size(0)))
            meters[f'{meter_prefix}_count_negatives'].add(float(result['negatives_dist'].size(0)))

            meters[f'{meter_prefix}_count_positives_all'].add(float(result['positives_dist_all_filtred'].size(0)))
            meters[f'{meter_prefix}_count_negatives_all'].add(float(result['negatives_dist_all_filtred'].size(0)))

            if len(hist_positives_dist) < args.max_embeddings_histograms or args.max_embeddings_histograms <= 0:
                avg_positives_dist_all = np.average(to_numpy(result['positives_dist_all']))
                np_negatives_dist_all = to_numpy(result['negatives_dist_all'])
                negative_max = max(negative_max, np.max(np_negatives_dist_all))
                avg_negatives_dist_all = np.average(np_negatives_dist_all)

                hist_positives_dist.append(avg_positives_dist_all)
                hist_negatives_dist.append(avg_negatives_dist_all)

                meters[f'{meter_prefix}_dist_positives'].add(avg_positives_dist_all)
                meters[f'{meter_prefix}_dist_negatives'].add(avg_negatives_dist_all)

                if result['positives_dist'].size(0) > 0:
                    avg_positives_dist_hard = np.average(to_numpy(result['positives_dist']))
                    meters[f'{meter_prefix}_dist_positives_hard'].add(avg_positives_dist_hard)
                    hist_positives_dist_hard.append(avg_positives_dist_hard)

                if result['negatives_dist'].size(0) > 0:
                    avg_negatives_dist_hard = np.average(to_numpy(result['negatives_dist']))
                    hist_negatives_dist_hard.append(avg_negatives_dist_hard)
                    meters[f'{meter_prefix}_dist_negatives_hard'].add(avg_negatives_dist_hard)

            output = to_numpy(result['output'].to('cpu')).tolist()
            y = to_numpy(result['y']).tolist()
            images = to_numpy(result['x']).tolist()

            max_samples_per_class = args.max_embeddings_per_class_test
            if data_loader == data_loader_train:
                max_samples_per_class = args.max_embeddings_per_class_train

            for idx, y_each in enumerate(y):
                # dictionary by y
                if y_each not in output_by_y.keys():
                    if data_loader == data_loader_train:
                        y_label = data_loader_train.dataset.classes[y_each]
                    else:
                        y_label = data_loader_test.dataset.classes[y_each]
                    output_by_y[y_each] = {
                        'embeddings': [],
                        'images': [],
                        'label': y_label,
                    }
                if max_samples_per_class <= 0 or len(output_by_y[y_each]['embeddings']) < max_samples_per_class:
                    output_by_y[y_each]['embeddings'].append(output[idx])
                    output_by_y[y_each]['images'].append(images[idx])

            idx_quick_test += 1
            if args.is_quick_test and idx_quick_test >= 5:
                break

        try:

            histogram_bins = 'auto'
            #histogram_bins = 'doane'

            tensorboard_writer.add_histogram(f'hist_{meter_prefix}_dist_positives', np.array(hist_positives_dist), epoch, bins=histogram_bins)
            tensorboard_writer.add_histogram(f'hist_{meter_prefix}_dist_negatives', np.array(hist_negatives_dist), epoch, bins=histogram_bins)

            tensorboard_writer.add_histogram(f'hist_{meter_prefix}_dist_positives_hard', np.array(hist_positives_dist_hard), epoch, bins=histogram_bins)
            tensorboard_writer.add_histogram(f'hist_{meter_prefix}_dist_negatives_hard', np.array(hist_negatives_dist_hard), epoch, bins=histogram_bins)

            tensorboard_utils.addHistogramsTwo(np.array(hist_positives_dist), np.array(hist_negatives_dist), f'hist_{meter_prefix}_all', epoch)
            tensorboard_utils.addHistogramsTwo(np.array(hist_positives_dist_hard), np.array(hist_negatives_dist_hard), f'hist_{meter_prefix}_hard', epoch)

        except Exception as e:
            logging.error(str(e))
            exc_type, exc_value, exc_tb = sys.exc_info()
            logging.error('\n'.join(traceback.format_exception(exc_type, exc_value, exc_tb)))


        output_embeddings = []
        output_y_labels = []
        output_y = []
        output_y_images = []
        for key in output_by_y.keys():
            output_embeddings += output_by_y[key]['embeddings']
            output_y_images += output_by_y[key]['images']
            label = output_by_y[key]['label']
            for _ in range(len(output_by_y[key]['embeddings'])):
                output_y_labels.append(label)
                output_y.append(key)

        predicted, target, target_y, class_max_dist, class_centroids, distances_precomputed = CentroidClassificationUtils.calulate_classes(
            np.array(output_embeddings),
            np.array(output_y),
            type='closest',
            norm=args.embedding_norm,
            triplet_similarity=args.triplet_similarity,
            class_max_dist=None,
            class_centroids=None,
            distances_precomputed=None
        )

        meters[f'{meter_prefix}_acc_closest'].add(predicted, target_y)

        tmp1 = predicted.permute(1, 0).data
        tmp2 = target.permute(1, 0).data
        meters[f'{meter_prefix}_auc_closest'].add(tmp1[0], tmp2[0])
        state[f'{meter_prefix}_max_dist'] = np.average(list(class_max_dist.values()))

        predicted, target, target_y, class_max_dist, class_centroids, distances_precomputed = CentroidClassificationUtils.calulate_classes(
            np.array(output_embeddings),
            np.array(output_y),
            type='range',
            norm=args.embedding_norm,
            triplet_similarity=args.triplet_similarity,
            class_max_dist=class_max_dist,
            class_centroids=class_centroids,
            distances_precomputed=distances_precomputed)

        meters[f'{meter_prefix}_acc_range'].add(predicted, target_y)

        tmp1 = predicted.permute(1, 0).data
        tmp2 = target.permute(1, 0).data
        meters[f'{meter_prefix}_auc_range'].add(tmp1[0], tmp2[0])


        # sampling of embeddings
        output_embeddings = []
        output_y_labels = []
        output_y = []
        output_y_images = []
        count_labels = 0
        for key in output_by_y.keys():
            count_labels += 1
            if count_labels > args.max_embeddings_projector_classes:
                break # tensorboard limits 50 clases
            embeddings = output_by_y[key]['embeddings']
            images = output_by_y[key]['images']

            if len(embeddings) > args.max_embeddings_projector_samples:
                embeddings = embeddings[:args.max_embeddings_projector_samples]
                images = images[:args.max_embeddings_projector_samples]

            output_embeddings += embeddings
            output_y_images += images

            label = output_by_y[key]['label']
            for _ in range(len(embeddings)):
                output_y_labels.append(label)
                output_y.append(key)

        # label_img: :math:`(N, C, H, W)
        tensorboard_writer.add_embedding(
            mat=torch.tensor(np.array(output_embeddings)),
            label_img=torch.FloatTensor(np.array(output_y_images)),
            metadata=output_y_labels,
            global_step=epoch, tag=f'{meter_prefix}_embeddings')

        # sampling of closest class embedding distances pairs
        hist_positives_dist_closest = []
        hist_negatives_dist_closest = []
        keys = list(output_by_y.keys())
        for idx_key_a, key_a in enumerate(keys):

            closest_key_b = key_a
            closest_dist = float('Inf')

            for idx_key_b in range(idx_key_a+1, len(keys)):
                key_b = keys[idx_key_b]

                # find closest class
                dist = CentroidClassificationUtils.get_distance(class_centroids[key_a], class_centroids[key_b], args.triplet_similarity)
                if closest_dist > dist:
                    closest_dist = dist
                    closest_key_b = key_b

            if closest_key_b != key_a:
                embeddings_positive = np.array(output_by_y[key_a]['embeddings'])
                embeddings_negative = np.array(output_by_y[closest_key_b]['embeddings'])

                embeddings_positive_center = np.tile(class_centroids[key_a], (len(embeddings_positive), 1))
                hist_positives_dist_closest += CentroidClassificationUtils.get_distance(embeddings_positive, embeddings_positive_center, args.triplet_similarity).tolist()
                embeddings_positive_center = np.tile(class_centroids[key_a], (len(embeddings_negative), 1))
                hist_negatives_dist_closest += CentroidClassificationUtils.get_distance(embeddings_negative, embeddings_positive_center, args.triplet_similarity).tolist()

        tensorboard_writer.add_histogram(f'hist_{meter_prefix}_dist_closest_pos', np.array(hist_positives_dist_closest), epoch, bins=histogram_bins)
        tensorboard_writer.add_histogram(f'hist_{meter_prefix}_dist_closest_neg', np.array(hist_negatives_dist_closest), epoch, bins=histogram_bins)
        tensorboard_utils.addHistogramsTwo(np.array(hist_positives_dist_closest), np.array(hist_negatives_dist_closest), f'hist_{meter_prefix}_closest', epoch)

        state[f'{meter_prefix}_negative_max'] = negative_max
        state[f'{meter_prefix}_acc_range'] = meters[f'{meter_prefix}_acc_range'].value()[0]
        fpr, tpr, eer = calc_err(meters[f'{meter_prefix}_auc_range'])
        state[f'{meter_prefix}_eer'] = eer

        state[f'{meter_prefix}_acc_closest'] = meters[f'{meter_prefix}_acc_closest'].value()[0]
        fpr, tpr, eer = calc_err(meters[f'{meter_prefix}_auc_closest'])
        state[f'{meter_prefix}_eer2'] = eer

        if meters[f'{meter_prefix}_loss'].n > 0:
            state[f'{meter_prefix}_loss'] = meters[f'{meter_prefix}_loss'].value()[0]

        if meters[f'{meter_prefix}_loss_emb'].n > 0:
            state[f'{meter_prefix}_loss_emb'] = meters[f'{meter_prefix}_loss_emb'].value()[0]

        if meters[f'{meter_prefix}_loss_center'].n > 0:
            state[f'{meter_prefix}_loss_center'] = meters[f'{meter_prefix}_loss_center'].value()[0]

        if meters[f'{meter_prefix}_loss_class'].n > 0:
            state[f'{meter_prefix}_loss_class'] = meters[f'{meter_prefix}_loss_class'].value()[0]

        if meters[f'{meter_prefix}_loss_neg'].n > 0:
            state[f'{meter_prefix}_loss_neg'] = meters[f'{meter_prefix}_loss_neg'].value()[0]

        if meters[f'{meter_prefix}_loss_pos'].n > 0:
            state[f'{meter_prefix}_loss_pos'] = meters[f'{meter_prefix}_loss_pos'].value()[0]

        if meter_prefix == 'test':
            if state[f'best_acc_closest'] < state[f'{meter_prefix}_acc_closest']:
                state[f'best_acc_closest'] = state[f'{meter_prefix}_acc_closest']
            if state[f'best_acc_range'] < state[f'{meter_prefix}_acc_range']:
                state[f'best_acc_range'] = state[f'{meter_prefix}_acc_range']

        state[f'{meter_prefix}_dist_positives'] = meters[f'{meter_prefix}_dist_positives'].value()[0]
        state[f'{meter_prefix}_dist_negatives'] = meters[f'{meter_prefix}_dist_negatives'].value()[0]

        state[f'{meter_prefix}_count_positives'] = meters[f'{meter_prefix}_count_positives'].value()[0]
        state[f'{meter_prefix}_count_negatives'] = meters[f'{meter_prefix}_count_negatives'].value()[0]

        state[f'{meter_prefix}_count_positives_all'] = meters[f'{meter_prefix}_count_positives_all'].value()[0]
        state[f'{meter_prefix}_count_negatives_all'] = meters[f'{meter_prefix}_count_negatives_all'].value()[0]

        state[f'{meter_prefix}_dist_delta'] = meters[f'{meter_prefix}_dist_negatives'].value()[0] - meters[f'{meter_prefix}_dist_positives'].value()[0]

        state[f'{meter_prefix}_dist_positives_hard'] = meters[f'{meter_prefix}_dist_positives_hard'].value()[0]
        state[f'{meter_prefix}_dist_negatives_hard'] = meters[f'{meter_prefix}_dist_negatives_hard'].value()[0]

        state[f'{meter_prefix}_dist_delta_hard'] = meters[f'{meter_prefix}_dist_negatives_hard'].value()[0] - meters[f'{meter_prefix}_dist_positives_hard'].value()[0]

        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_loss', scalar_value=state[f'{meter_prefix}_loss'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_loss_emb', scalar_value=state[f'{meter_prefix}_loss_emb'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_loss_center', scalar_value=state[f'{meter_prefix}_loss_center'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_loss_neg', scalar_value=state[f'{meter_prefix}_loss_neg'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_loss_pos', scalar_value=state[f'{meter_prefix}_loss_pos'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_loss_class', scalar_value=state[f'{meter_prefix}_loss_class'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_dist_delta', scalar_value=state[f'{meter_prefix}_dist_delta'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_dist_positives', scalar_value=state[f'{meter_prefix}_dist_positives'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_dist_negatives', scalar_value=state[f'{meter_prefix}_dist_negatives'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_dist_positives_hard', scalar_value=state[f'{meter_prefix}_dist_positives_hard'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_dist_negatives_hard', scalar_value=state[f'{meter_prefix}_dist_negatives_hard'], global_step=epoch)

        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_count_positives', scalar_value=state[f'{meter_prefix}_count_positives'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_count_negatives', scalar_value=state[f'{meter_prefix}_count_negatives'], global_step=epoch)

        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_count_positives_all', scalar_value=state[f'{meter_prefix}_count_positives_all'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_count_negatives_all', scalar_value=state[f'{meter_prefix}_count_negatives_all'], global_step=epoch)

        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_negative_max', scalar_value=state[f'{meter_prefix}_negative_max'], global_step=epoch)

        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_acc_range', scalar_value=state[f'{meter_prefix}_acc_range'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_eer', scalar_value=state[f'{meter_prefix}_eer'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_acc_closest', scalar_value=state[f'{meter_prefix}_acc_closest'], global_step=epoch)
        tensorboard_writer.add_scalar(tag=f'{meter_prefix}_eer2', scalar_value=state[f'{meter_prefix}_eer2'], global_step=epoch)

        tensorboard_utils.addPlot1D(
            data=(meters[f'{meter_prefix}_auc_range'].value()[2], meters[f'{meter_prefix}_auc_range'].value()[1]),
            tag=f'{meter_prefix}_auc_range',
            global_step=epoch,
            axis_labels=[
                'False positives',
                'True positives'
            ]
        )
        tensorboard_utils.addPlot1D(
            data=(meters[f'{meter_prefix}_auc_closest'].value()[2], meters[f'{meter_prefix}_auc_closest'].value()[1]),
            tag=f'{meter_prefix}_auc_closest',
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
            if math.isnan(percent_improvement):
                percent_improvement = 0

        if state[args.early_stopping_param] >= 0:
            if args.early_stopping_delta_percent > percent_improvement:
                state['early_stopping_patience'] += 1
            else:
                state['early_stopping_patience'] = 0
        state['early_percent_improvement'] = percent_improvement

    tensorboard_writer.add_scalar(tag='improvement', scalar_value=state['early_percent_improvement'], global_step=epoch)
    torch.save(model_module.state_dict(), os.path.join(run_path, f'{args.name}.pt'))

    logging.info(
        f'{args.name} {round(percent * 100, 2)}% each: {round(state["avg_epoch_time"], 2)} min eta: {round(eta, 2)} min acc: {round(state["train_acc_range"], 2)} loss: {round(state["train_loss"], 2)} improve: {round(percent_improvement, 2)}')

    CsvUtils.add_results_local(args, state)
    CsvUtils.add_results(args, state)

    if state['early_stopping_patience'] >= args.early_stopping_patience:
        logging_utils.info(f'{args.name} early stopping')
        break


tensorboard_writer.close()