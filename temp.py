import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

mins = []
maxs = []
avgs = []
deltas = []

def func1(output_emb):
    #output_emb = F.tanh(output_emb)
    norm = torch.norm(output_emb, p=2, dim=1, keepdim=True)
    output_norm = output_emb / norm
    return output_norm

def func_angle(vec1, vec2):
    norm1 = torch.norm(vec1, p=2, dim=1, keepdim=True)
    norm2 = torch.norm(vec2, p=2, dim=1, keepdim=True)
    rad = torch.acos(torch.bmm(vec1.unsqueeze(1), vec2.unsqueeze(-1))/norm1 * norm2)
    return 180. * rad / np.pi

def func_pairwise(vec1, vec2):
    dist = F.pairwise_distance(vec1, vec2, p=2)
    return dist

def func_cosine_distance(vec1, vec2):
    dist = 1.0 - F.cosine_similarity(vec1, vec2)
    return dist

dim = 128
a = torch.ones((1, dim))
b = -torch.ones((1, dim))

a = func1(a)
b = func1(b)

dist = func_cosine_distance(a, b)

reduction_dim = 2
for dim in range(reduction_dim, 128):
    print(f'\n\ndim = {dim}')
    input1 = torch.FloatTensor(1000, dim).uniform_(-1., 1.)
    input2 = torch.FloatTensor(1000, dim).uniform_(-1., 1.)

    input1b = func1(input1)
    input2b = func1(input2)

    #output = func_pairwise(input1b, input2b)
    #output = mdist.forward(input1b, input2b)
    #output = func_cosine_distance(input1b, input2b)
    output = F.smooth_l1_loss(input1b, input2b)
    np_output = output.data.numpy()

    # np_embeddings = input1b.data.numpy()
    # pca = PCA(n_components=reduction_dim, copy=True)
    # pca.fit(np_embeddings)
    # input1b = pca.transform(np.array(np_embeddings))
    #
    # np_embeddings = input2b.data.numpy()
    # pca = PCA(n_components=reduction_dim, copy=True)
    # pca.fit(np_embeddings)
    # input2b = pca.transform(np.array(np_embeddings))
    #
    # np_output = cosine_distances(input1b, input2b)

    print(f'min = {np.min(np_output)}')
    print(f'median = {np.median(np_output)}')
    print(f'avg = {np.average(np_output)}')
    print(f'max = {np.max(np_output)}')
    print(f'delta = {np.max(np_output) - np.min(np_output)}')

    mins.append(np.min(np_output))
    maxs.append(np.max(np_output))
    avgs.append(np.average(np_output))
    #print(output)


fig, ax = plt.subplots(sharex=True)

deltas = np.array(maxs) - np.array(mins)
# ax.plot(np.arange(len(deltas)), deltas, 'r-', label='max - min')

ax.plot(np.arange(len(mins)), mins, 'r-', label='min')
ax.plot(np.arange(len(maxs)), maxs, 'g-', label='max')
ax.plot(np.arange(len(avgs)), avgs, 'b-', label='avg')

ax.set(xlabel='dimensions', ylabel='distance',
       title='')
ax.legend()
ax.grid()

plt.show()