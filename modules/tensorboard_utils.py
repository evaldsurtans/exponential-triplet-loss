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

class TensorBoardUtils(object):
    def __init__(self, tensorboard_writer):
        super(TensorBoardUtils, self).__init__()
        self.tensorboard_writer = tensorboard_writer

    def addPlotConfusionMatrix(self, dataXY, ticks, tag, global_step=0, is_percent_added=False):
        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(dataXY, cmap=plt.cm.jet if is_percent_added else plt.cm.binary,
                        interpolation='nearest')

        width, height = dataXY.shape

        if is_percent_added:
            for x in range(width):
                for y in range(height):
                    ax.annotate(str(dataXY[x][y]), xy=(y, x),
                                horizontalalignment='center',
                                verticalalignment='center')

        cb = fig.colorbar(res)
        if ticks is not None:
            plt.xticks(range(width), ticks)
            plt.yticks(range(height), ticks)

        plt.xlabel('Actual class')
        plt.ylabel('Predicted class')

        fig.set_tight_layout(True)
        canvas = FigureCanvas(fig)
        canvas.draw()

        # width, height = fig.get_size_inches() * fig.get_dpi()
        width, height = fig.canvas.get_width_height()
        image = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
        self.tensorboard_writer.add_image(tag=tag, img_tensor=image, global_step=global_step)

        plt.close(fig)


    def addHistogramsTwo(self, data_positives, data_negatives, tag, global_step=0):
        fig = plt.figure()
        plt.clf()

        n, bins, patches = plt.hist(data_positives, 100, density=True, facecolor='g', alpha=0.75)
        n, bins, patches = plt.hist(data_negatives, 100, density=True, facecolor='r', alpha=0.75)

        plt.xlabel('Distance')
        plt.ylabel('Samples')

        fig.set_tight_layout(True)
        canvas = FigureCanvas(fig)
        canvas.draw()

        # width, height = fig.get_size_inches() * fig.get_dpi()
        width, height = fig.canvas.get_width_height()
        image = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
        self.tensorboard_writer.add_image(tag=tag, img_tensor=image, global_step=global_step)

        plt.close(fig)


    def addPlot2D(self, dataXY, tag, global_step=0):
        dataXY = np.copy(dataXY)
        min_value = np.min(dataXY)
        dataXY += abs(min_value)

        max_value = np.max(dataXY)
        dataXY /= max_value

        dataXY *= 255
        dataXY = dataXY.astype(dtype=np.uint8)

        image = np.transpose(dataXY) # H, W
        image = np.expand_dims(image, axis=2)
        image = np.tile(image, (1,1,3))

        self.tensorboard_writer.add_image(tag=tag, img_tensor=image, global_step=global_step)


    def addPlot1D(self, data, tag, global_step=0, axis_labels=None):
        data = np.copy(data)
        fig = plt.figure()

        if not axis_labels is None:
            plt.xlabel(axis_labels[0])
            plt.ylabel(axis_labels[1])
            if len(data) > 1:
                plt.plot(data[0], data[1])
        else:
            plt.plot(data)

        fig.set_tight_layout(True)
        canvas = FigureCanvas(fig)
        canvas.draw()

        # width, height = fig.get_size_inches() * fig.get_dpi()
        width, height = fig.canvas.get_width_height()
        image = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
        self.tensorboard_writer.add_image(tag=tag, img_tensor=image, global_step=global_step)

        plt.close(fig)