import io
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.sampler


def l2(output_emb):
    norm = torch.norm(output_emb, p=2, dim=1, keepdim=True).detach()
    output_norm = output_emb / norm
    return output_norm

x1 = torch.ones((32, 8))
x2 = -torch.ones((32, 8))

x1 = l2(x1)
x2 = l2(x2)

distances = 1. - F.cosine_similarity(x1, x2, dim=1, eps=1e-20)

output = torch.ones((6, 8))
output[2:,:] = -output[2:,:]

distances_batch = []
batch_size = output.size(0)
for i in range(batch_size):

    x1 = output[i, :].expand(batch_size, -1)
    x2 = output # whole batch
    # bug DIM
    distances = 1. - F.cosine_similarity(x1, x2, dim=1, eps=1e-20) # -1 .. 1
    # dot product ROUNDING PROBLEM IF 1.0 , need to be 1.00001
    # result = 0 .. 2.0
    distances_batch.append(distances)

dists = torch.stack(distances_batch)
print(distances)