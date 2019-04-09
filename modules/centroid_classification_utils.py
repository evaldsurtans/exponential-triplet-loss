import os
import unicodedata
import string
import glob
import io
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

from modules.dict_to_obj import DictToObj
from modules.math_utils import cosine_similarity, normalize_vec


class CentroidClassificationUtils(object):

    @staticmethod
    def get_distance(x1, x2, triplet_similarity):
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

        if triplet_similarity == 'cos':
            if isinstance(x1, np.ndarray):
                dist = sklearn.metrics.pairwise.cosine_distances(x1, x2)[0]
            else:
                dist = 1. - F.cosine_similarity(x1, x2, dim=1, eps=1e-20) # -1 .. 1 => 0 .. 2
        else:
            if isinstance(x1, np.ndarray):
                dist = sklearn.metrics.pairwise.paired_euclidean_distances(x1, x2)
            else:
                dist = F.pairwise_distance(x1, x2, eps=1e-20) # 0 .. 2

        if is_item:
            dist = dist[0]
        return dist


    # @type = 'range' 'closest'
    @staticmethod
    def calulate_classes(embeddings, y_list, type='range', norm='l2', triplet_similarity='cos', class_max_dist=None, class_centroids=None, distances_precomputed={}):

        class_centroids_noncomputed = {}
        for idx_emb, embedding in enumerate(embeddings):
            y = y_list[idx_emb]
            if y not in class_centroids_noncomputed:
                class_centroids_noncomputed[y] = []
            class_centroids_noncomputed[y].append(embedding)

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
        target_y = []

        for idx_emb, embedding in enumerate(embeddings):

            if np.isnan(embedding).any():
                logging.error(f'something wrong comming out of model calulate_classes NaN: {idx_emb}')
                continue

            y_idx_real = y_list[idx_emb]

            closest_dist = float('Inf')
            closest_idx = list(class_centroids.keys())[0]

            for key in class_centroids.keys():

                y_idx = key
                y_embedding = class_centroids[key]
                max_dist = class_max_dist[key]

                dist = None
                if key in distances_precomputed.keys():
                    if idx_emb in distances_precomputed[key].keys():
                        dist = distances_precomputed[key][idx_emb]
                else:
                    distances_precomputed[key] = {}

                if dist is None:
                    # calculate if in range of some centroid other than real one
                    dist = CentroidClassificationUtils.get_distance(embedding, y_embedding, triplet_similarity)
                    distances_precomputed[key][idx_emb] = dist

                if type == 'range':
                    if max_dist > dist:
                        predicted[idx_emb][y_idx] += 1.0

                if closest_dist > dist:
                    closest_dist = dist
                    closest_idx = y_idx

            if type == 'closest':
                predicted[idx_emb][closest_idx] = 1.0

            target[idx_emb][y_idx_real] = 1.0
            target_y.append(y_idx_real)

        if type == 'range':
            predicted = predicted /(np.sum(predicted, axis=1, keepdims=True) + 1e-18)

        return torch.tensor(np.array(predicted)), torch.tensor(np.array(target)), torch.tensor(np.array(target_y)), class_max_dist, class_centroids, distances_precomputed


