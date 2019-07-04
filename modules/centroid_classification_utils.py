import multiprocessing
import os
import unicodedata
import string
import glob
import io
from multiprocessing import Process

import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.sampler
import random
import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tensorboardX
import argparse
from datetime import datetime
import sklearn
import sklearn.model_selection
from enum import Enum
import json
import logging
import numpy as np
import traceback, sys
import itertools
from numpy import dot
from numpy.linalg import norm
import logging

from modules.dict_to_obj import DictToObj
from modules.math_utils import cosine_similarity, normalize_vec


class CentroidClassificationUtils(object):

    @staticmethod
    def get_distance(x1, x2, triplet_similarity):
        n_jobs = 8 if x1.shape[0] > 1 else 1
        is_item = False
        if isinstance(x1, np.ndarray):
            if len(x1.shape) == 1:
                is_item = True
                x1 = np.expand_dims(x1, 0)
                x2 = np.expand_dims(x2, 0)
        else:
            if len(x1.size()) == 1:
                x1 = x1.unsqueeze(0)
                x2 = x2.unsqueeze(0)
                is_item = True

        # latest needed: conda install -c anaconda scikit-learn
        with sklearn.config_context(working_memory=1024):
            if triplet_similarity == 'cos':
                if isinstance(x1, np.ndarray):
                    dist = np.zeros((0, ))
                    for each in sklearn.metrics.pairwise.pairwise_distances_chunked(x1, x2, metric="cosine", n_jobs=n_jobs):
                        dist = np.concatenate((dist, np.diag(each)), axis=0)
                else:
                    dist = 1. - F.cosine_similarity(x1, x2, dim=1, eps=1e-20) # -1 .. 1 => 0 .. 2
            else:
                if isinstance(x1, np.ndarray):
                    dist = np.zeros((0, ))
                    for each in sklearn.metrics.pairwise.pairwise_distances_chunked(x1, x2, metric="euclidean", n_jobs=n_jobs):
                        dist = np.concatenate((dist, np.diag(each)), axis=0)
                else:
                    dist = F.pairwise_distance(x1, x2, eps=1e-20) # 0 .. 2

        if is_item:
            dist = dist[0]
        return dist


    # @type = 'range' 'closest'
    @staticmethod
    def calulate_classes(embeddings, y_list, type='range', norm='l2', triplet_similarity='cos', class_max_dist=None, class_centroids=None, distances_precomputed=None):

        class_centroids_noncomputed = {}
        for idx_emb, embedding in enumerate(embeddings):
            y = y_list[idx_emb]
            if y not in class_centroids_noncomputed:
                class_centroids_noncomputed[y] = []
            class_centroids_noncomputed[y].append(embedding)

        if distances_precomputed is None:
            distances_precomputed = {}

        if class_max_dist is None:
            class_centroids = {}
            class_max_dist = {}
            for key in class_centroids_noncomputed.keys():
                np_all_centroids = np.array(class_centroids_noncomputed[key])
                class_centroids[key] = np.average(np_all_centroids, axis=0)
                if norm == 'l2':
                    class_centroids[key] = normalize_vec(class_centroids[key])

                np_class_centroids_tiled = np.tile(class_centroids[key], (len(np_all_centroids),1))
                list_dists = CentroidClassificationUtils.get_distance(np_class_centroids_tiled, np_all_centroids, triplet_similarity).tolist()
                list_dists = sorted(list_dists, reverse=False)
                list_dists = list_dists[:max(2, int(len(list_dists) * 0.9))] # drop 10 top percent embeddings as they could contain noise
                class_max_dist[key] = list_dists[-1] # last largest distance

        predicted = np.zeros( (embeddings.shape[0], int(np.max(list(class_centroids.keys()))) + 1), dtype=np.float )
        target = np.zeros( (embeddings.shape[0], int(np.max(list(class_centroids.keys()))) + 1), dtype=np.float )

        def process_class(y_idx):
            y_embedding = class_centroids[y_idx]
            max_dist = class_max_dist[y_idx]

            dists = None
            if y_idx in distances_precomputed.keys():
                dists = distances_precomputed[y_idx]
            else:
                distances_precomputed[y_idx] = {}

            if dists is None:
                logging.info(f'center calc: {y_idx} len: {len(list(class_max_dist.keys()))} / {len(embeddings)} sim: {triplet_similarity}')
                # calculate if in range of some centroid other than real one
                np_class_centroids_tiled = np.tile(y_embedding, (len(embeddings),1))
                dists = CentroidClassificationUtils.get_distance(embeddings, np_class_centroids_tiled, triplet_similarity)
                distances_precomputed[y_idx] = dists

            if type == 'range':
                for idx_emb, dist in enumerate(dists):
                    if max_dist > dist:
                        predicted[idx_emb][y_idx] += 1.0
            else:
                idx_emb = np.argmin(dists)
                predicted[idx_emb][y_idx] = 1.0

        for y_idx in class_max_dist.keys():
            process_class(y_idx)

        if type == 'range':
            # predicted sum can be 0 if none in the range
            predicted = predicted /(np.sum(predicted, axis=1, keepdims=True) + 1e-18)

        return torch.tensor(np.array(predicted)), torch.tensor(np.array(target)), torch.tensor(np.array(y_list)), class_max_dist, class_centroids, distances_precomputed


