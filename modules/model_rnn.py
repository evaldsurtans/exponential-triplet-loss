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
from modules.lstm_cell_opt_2 import LSTM_Cell

class RNN(torch.nn.Module):
    def __init__(self,
                 is_cuda,
                 input_size,
                 timesteps_size,
                 hidden_size,
                 output_size,
                 output_timesteps_size,
                 layers_size):

        super(RNN, self).__init__()

        self.is_cuda = is_cuda
        self.hidden_size = hidden_size
        self.layers_size = layers_size
        self.timesteps_size = timesteps_size
        self.output_size = output_size
        self.output_timesteps_size = output_timesteps_size

        self.lstm_layers_1 = []
        self.lstm_layers_1_parallel = []
        '''
        for i in range(self.layers_size):
            lstm = LSTM_Cell(
                input_size = input_size if i == 0 else hidden_size,  # features
                hidden_size = hidden_size,
                init_forget_gate = 3.0
            )
            self.lstm_layers_1.append(lstm)

        for i in range(self.layers_size):
            lstm = LSTM_Cell(
                input_size = input_size if i == 0 else hidden_size,  # features
                hidden_size = hidden_size,
                init_forget_gate = None
            )
            self.lstm_layers_1_parallel.append(lstm)
'''
        #self.layer_1_lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=self.layers_size)
        self.layer_1_lstm = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=self.layers_size, batch_first=True)

        for weights_layer in self.layer_1_lstm.all_weights:
            for each in weights_layer:
                if len(each.size()) > 1:
                    torch.nn.init.xavier_normal(each)

        self.layer_2_linear = torch.nn.Linear(hidden_size, hidden_size)
        #self.layer_2_relu = torch.nn.ReLU()
        self.layer_3_linear = torch.nn.Linear(hidden_size, hidden_size)
        #self.layer_3_relu = torch.nn.ReLU()
        self.layer_4_linear = torch.nn.Linear(hidden_size, output_size)

        torch.nn.init.xavier_normal(self.layer_2_linear.weight)
        torch.nn.init.xavier_normal(self.layer_3_linear.weight)
        torch.nn.init.xavier_normal(self.layer_4_linear.weight)

    def last_timestep(self, unpacked, lengths):
        """
        unpacked: B * maxSentenceLen * en_hidden_size
        sent_len: B*1  the real length of every sentence
        """
        # Index of the last output for each sequence.
        # https://github.com/Liwb5/machineTranslation/blob/8be3b4b9759d627c5ccb78fc9de69513bd09a894/3code/decoder.py
        last_indexes = (lengths - 1)
        batch_last_indexes = last_indexes.view(-1, 1)
        idx = batch_last_indexes.expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(dim=1, index=idx).squeeze()

    def forward(self, input):

        batch_size = input.shape[0]
        #lengths = torch.autograd.Variable(torch.from_numpy(np.array([self.timesteps_size for _ in range(batch_size)])))

        hx = torch.autograd.Variable(input.data.new(self.layers_size,
                                                    batch_size,
                                                    self.hidden_size).zero_(), requires_grad=False)
        if self.is_cuda:
            hx = hx.cuda()

        lstm_out, hidden = self.layer_1_lstm.forward(input, hx)

        #lstm_out = self.last_timestep(lstm_out, lengths)

        # lstm_out (batch, steps, hidden)
        lstm_out = lstm_out[:, -self.output_timesteps_size:, :]
        lstm_out = lstm_out.permute(1, 0, 2)
        # lstm_out (steps, batch, hidden)
        output = torch.autograd.Variable(torch.zeros([self.output_timesteps_size, batch_size, self.output_size]))
        if self.is_cuda:
            output = output.cuda()

        for idx in range(self.output_timesteps_size):
            each_timestep = lstm_out[idx]
            last_outputs = self.layer_2_linear.forward(each_timestep)
            last_outputs = F.relu(last_outputs)
            last_outputs = self.layer_3_linear.forward(last_outputs)
            last_outputs = F.relu(last_outputs)
            output[idx] = self.layer_4_linear.forward(last_outputs)

        # output (batch, steps, hidden)
        output = output.permute(1, 0, 2)
        return output

class RNN_NativeLSTM(torch.nn.Module):
    def __init__(self,
                 input_size,
                 timesteps_size,
                 hidden_size,
                 output_size,
                 layers_size):

        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.layers_size = layers_size
        self.timesteps_size = timesteps_size

        self.layer_1_lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=self.layers_size)
        self.layer_2_linear = torch.nn.Linear(hidden_size, output_size)


    def last_timestep(self, unpacked, lengths):
        """
        unpacked: B * maxSentenceLen * en_hidden_size
        sent_len: B*1  the real length of every sentence
        """
        # Index of the last output for each sequence.
        # https://github.com/Liwb5/machineTranslation/blob/8be3b4b9759d627c5ccb78fc9de69513bd09a894/3code/decoder.py
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0),
                                               unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(1, idx).squeeze()

    def forward(self, input):

        batch_size = input.shape[0]
        lengths = torch.autograd.Variable(torch.from_numpy(np.array([self.timesteps_size for _ in range(batch_size)])))
        hidden_variables_lstm = (
            torch.autograd.Variable(torch.zeros(self.layers_size, batch_size, self.hidden_size)),
            torch.autograd.Variable(torch.zeros(self.layers_size, batch_size, self.hidden_size)))

        packed = torch.nn.utils.rnn.pack_padded_sequence(input=input, lengths=list(lengths.data), batch_first=True)
        lstm_out, hidden = self.layer_1_lstm.forward(packed, hidden_variables_lstm)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        last_outputs = self.last_timestep(lstm_out, lengths)
        output = self.layer_2_linear(last_outputs)

        return output


'''
        for i, lstm in enumerate(self.lstm_layers_1):
            lstm_out = lstm.forward(input if i == 0 else lstm_out)


        for i, lstm in enumerate(self.lstm_layers_1_parallel):
            lstm_out_parallel = lstm.forward(input if i == 0 else lstm_out_parallel)
        '''
# lstm_out = self.last_timestep(lstm_out, lengths)
# lstm_out_parallel = self.last_timestep(lstm_out_parallel, lengths)

# hidden_variables_lstm = (
#    torch.autograd.Variable(torch.zeros(self.layers_size, batch_size, self.hidden_size)),
#    torch.autograd.Variable(torch.zeros(self.layers_size, batch_size, self.hidden_size)))

# packed = torch.nn.utils.rnn.pack_padded_sequence(input=input, lengths=list(lengths.data), batch_first=True)
# lstm_out, hidden = self.layer_1_lstm.forward(packed, hidden_variables_lstm)
# lstm_out_parallel, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

# lstm_out_parallel = lstm_out_parallel.contiguous().view(lstm_out_parallel.size(0), lstm_out_parallel.size(1) * lstm_out_parallel.size(2))
# lstm_out = lstm_out.contiguous().view(lstm_out.size(0), lstm_out.size(1) * lstm_out.size(2))

# dim_concat = 0 if len(lstm_out.size()) == 1 else 1
# last_outputs = torch.cat((lstm_out_parallel, lstm_out), dim=dim_concat)