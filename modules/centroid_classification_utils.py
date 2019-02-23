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

    # @type = 'range' 'closest'
    @staticmethod
    def calulate_classes(embeddings, y_list, type='range', norm='l2'):

        class_centroids_noncomputed = {}
        for idx, embedding in enumerate(embeddings):
            y = y_list[idx]
            if y not in class_centroids_noncomputed:
                class_centroids_noncomputed[y] = []
            class_centroids_noncomputed[y].append(embedding)

        class_centroids = {}
        class_max_dist = {}
        for key in class_centroids_noncomputed.keys():
            np_all_centroids = np.array(class_centroids_noncomputed[key])
            class_centroids[key] = np.average(np_all_centroids, axis=0)
            if norm == 'l2':
                class_centroids[key] = normalize_vec(class_centroids[key])

            list_dists = []
            for emb in np_all_centroids:
                list_dists.append(cosine_similarity(class_centroids[key], emb))
            list_dists = sorted(list_dists, reverse=False)
            list_dists = list_dists[:max(2, int(len(list_dists) * 0.9))] # drop 10 top percent embeddings as they could contain noise
            class_max_dist[key] = list_dists[-1] # last largest distance

        predicted = np.zeros( (embeddings.shape[0], len(class_centroids.keys())), dtype=np.float )
        target = np.zeros( (embeddings.shape[0], len(class_centroids.keys())), dtype=np.float )
        target_y = []

        for idx, embedding in enumerate(embeddings):
            y_idx_real = y_list[idx]

            closest_dist = float('Inf')
            closest_idx = list(class_centroids.keys())[0]

            for key in class_centroids.keys():

                y_idx = key
                y_embedding = class_centroids[key]
                max_dist = class_max_dist[key]

                # calculate if in range of some centroid other than real one
                dist = cosine_similarity(embedding, y_embedding)

                if type == 'range':
                    if max_dist > dist:
                        predicted[idx][y_idx] += 1.0

                if closest_dist > dist:
                    closest_dist = dist
                    closest_idx = y_idx

            if type == 'closest':
                predicted[idx][closest_idx] = 1.0

            target[idx][y_idx_real] = 1.0
            target_y.append(y_idx_real)

        if type == 'range':
            predicted = predicted / np.sum(predicted, keepdims=True)
        max_dist = np.average(list(class_max_dist.values()))

        return torch.tensor(np.array(predicted)), torch.tensor(np.array(target)), torch.tensor(np.array(target_y)), max_dist


