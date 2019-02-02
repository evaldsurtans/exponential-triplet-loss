import matplotlib
# matplotlib.use("TkAgg")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# plt.ion() # interactive non-blocking mode
# plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
#neg = np.arange(0, 10, 0.25)
#pos = np.arange(0, 10, 0.25)

min_dist = 0.0
max_dist = 2.0

#vis_max_dist = max_dist

vis_max_dist = 4.0/100

neg = np.arange(min_dist, vis_max_dist, (vis_max_dist-min_dist)/100)
pos = np.arange(min_dist, max_dist, (max_dist-min_dist)/100)

#max_dist = 2.0

neg, pos = np.meshgrid(neg, pos)
#R = np.sqrt(neg**2 + pos**2)
#Z = np.sin(R)




# margin loss standard triplet loss
#margin = 0.5
#Z = np.maximum(np.zeros_like(neg), pos - neg + margin)

#exp2
# margin = 0.2
# coef_exp = 3.0
# neg_coef = 1.5
# Z = np.exp(coef_exp *
#            np.maximum(np.zeros_like(pos), (pos/max_dist) - margin)) - 1.0 + \
#     neg_coef * (np.exp(coef_exp *
#                        np.maximum(np.zeros_like(pos), ((max_dist - neg)/max_dist) - margin)) - 1.0)


margin = 0.2
coef_exp = 1.0
neg_coef = 1.0
coef_overlap = 1.2

k = 10

margin = 1.0 / k # allways will have good margin
pos_norm = (pos/max_dist)
radius_cluster = margin * coef_overlap / 2.0

# print(f'margin: {margin}')
# print(f'radius_cluster: {radius_cluster}')
# print(f'pos_norm: {pos_norm}')

neg_norm = (neg/max_dist)

#np.maximum(np.zeros_like(pos), ((max_dist - neg)/max_dist) - margin))

# Z = np.exp(coef_exp *
#            np.maximum(np.zeros_like(pos), pos_norm - pos_delta)) - 1.0 + \
#     neg_coef * (
#             np.exp(coef_exp * np.maximum(np.zeros_like(pos), np.abs(neg_norm - margin) - pos_delta))
#             - 1.0)
#Z = pos_norm + np.abs(neg_norm - margin)

#Z = np.exp(2.0 * (pos_norm - radius_cluster * 2.0)) + np.exp(2.0 * np.abs(neg_norm - (margin + radius_cluster))) - 2.0
#Z = np.exp(2.0 * pos_norm) - 1.0 + 6.0 * np.abs(neg_norm - (margin + radius_cluster))

# exp4
# Z = np.exp(2.0 * np.maximum(np.zeros_like(pos_norm), (pos_norm - radius_cluster * 2.0) )) + \
#     np.exp(2.0 * np.maximum(np.zeros_like(pos_norm), np.abs(neg_norm - margin) - radius_cluster) ) - 2.0


#exp5
K = 10
O_k = 1.1
C = max_dist / K
O = O_k * C

print(f'O={O}')
print(f'C={C}')

# exp5 pamata ideja
# Z = np.exp(2.0 * np.maximum(np.zeros_like(pos), (pos - O))) + \
#     np.exp(2.0 * np.maximum(np.zeros_like(neg), np.abs((0.5*C + O) - neg) - O) )
#Z = np.exp(2.0 * np.abs((0.5*C + O) - neg) )

mask = ((0.5*C + O) - neg) < 0
mask = mask.astype(np.float)
mask_abs = np.abs(mask - 1.0)

delta = (0.5*C + O)


# exp5
# Z = (np.exp(2.0 * np.maximum(np.zeros_like(pos), (pos - O))) - 1.0) + \
#     mask_abs * 30 * np.maximum(np.zeros_like(neg), (delta - neg) - O/2) + \
#     mask * (np.exp(2.0 * np.maximum(np.zeros_like(neg), ((0.5*C + O) - neg) * -1.0 - O/2)) - 1.0)

#func = lambda it: -np.log(1.0 - it)
#func = lambda it: -np.log10(1.0 - it)
func = _tan = lambda it: np.tan(1.3*it)
#func = lambda it: np.exp(1 * it) - 1.0


# Z = func( np.maximum(np.zeros_like(pos), (pos - O)) / max_dist ) + \
#     func( np.maximum(np.zeros_like(neg), np.abs((0.5*C + O) - neg) - O) / max_dist )

# exp6 ar tan sakuma
# Z = (func( np.maximum(np.zeros_like(pos), (pos - O)) / max_dist ) ) + \
#     mask_abs * 1.5 * np.maximum(np.zeros_like(neg), (delta - neg) - O/2) + \
#     mask * (func( np.maximum(np.zeros_like(neg), ((0.5*C + O) - neg) * -1.0 - O/2) / max_dist ) )

K = 100
pi_k = K * 2 - 1
C_norm = 1.0/K
C_limit = 1.5*C_norm
sin_coef = 1.0/K # jo mazaks x/.. jo mazakas bedrites
tan_coef = 1.0/K

#- np.pi
_sin = lambda it: sin_coef * np.sin(pi_k*np.pi*it-np.pi*1.5) + sin_coef
#Z = _tan( np.maximum(np.zeros_like(pos), (pos - O)) / max_dist )
#Z = _sin( (np.abs((0.5*C + O) - neg) - O) / max_dist)

neg_norm = neg / max_dist
print(f'max_dist: {max_dist}')
#Z = _sin(neg_norm)



mask = neg_norm < C_limit
mask = mask.astype(np.float)


Z = tan_coef * _tan( np.maximum(np.zeros_like(pos_norm), (pos_norm - C_norm)) ) + \
    mask * _sin( neg_norm )


# nedot pos piemērus, kas ir iekshpus
# nedot neg piemerus kas ir Arpus radius_cluster


loss_min = np.min(Z)
loss_max = np.max(Z)
col_min = float('Inf')
col_max = float('-Inf')

for i in range(pos.shape[0]):
    for j in range(pos.shape[1]):
        each_pos = pos[i, j]
        each_neg = neg[i, j]

        # hard
        # if each_pos + margin <= each_neg:
        #     Z[i,j] = np.nan
        # semi
        # if each_neg <= each_pos:
        #     Z[i,j] = np.nan

        # abs_margins
        # if each_pos/max_dist - margin <= 0 and (max_dist - each_neg)/max_dist - margin < 0:
        #     Z[i,j] = np.nan

        # if each_pos/max_dist < radius_cluster * 2.0 and \
        #         each_neg/max_dist > margin - radius_cluster and \
        #         each_neg/max_dist < margin + radius_cluster:
        #     Z[i,j] = np.nan

        if Z[i,j] != np.nan:
            col_min = min(col_min, Z[i,j])
            col_max = max(col_max, Z[i,j])

# buggy
#Z = np.exp(pos) - 1.0 + np.exp(max_dist - neg) - 1.0

# fixed
#Z = np.exp(coef_exp*pos/max_dist) - 1.0 + neg_coef * np.exp(coef_exp*(max_dist - neg)/max_dist) - 1.0



#Z = np.exp(np.maximum(np.zeros_like(pos), pos - 0.2)) - 1.0 + np.exp(np.maximum(np.zeros_like(neg), 2.0 - neg - 0.2)) - 1.0


# lossless_beta = 20.0
# dims = 16.0
# Z = - np.log10(1.0 - pos / lossless_beta) - np.log10(1.0 - (dims - neg)/lossless_beta)

#Z = np.exp(pos) - 1.0 + 6 * np.exp(1.0 - neg) - 1.0


# ratio loss
#Z = (np.exp(pos)/(np.exp(pos)+np.exp(neg)))**2 + (1.0 - (np.exp(neg)/(np.exp(pos)+np.exp(neg)))**2)

#Z = (np.exp(coef_exp*pos)/(np.exp(coef_exp*pos)+np.exp(coef_exp*neg)))**2 + (1.0 - (np.exp(coef_exp*neg)/(np.exp(coef_exp*pos)+np.exp(coef_exp*neg)))**2)

part_pos = np.exp(coef_exp*pos)
part_neg = np.exp(coef_exp*neg)
part_div = part_pos + part_neg
#Z = (part_pos/part_div)**2 + (1.0 - (part_neg/part_div)**2)

# Plot the surface.
cmap = cm.coolwarm

#cmap.set_under(color='black')
#col_min = 1e-20
surf = ax.plot_surface(neg, pos, Z, cmap=cmap,
                       linewidth=0, antialiased=False, vmin=col_min, vmax=col_max)
ax.set_zlim(loss_min, loss_max)

ax.set_xlabel('neg')
ax.set_ylabel('pos')
ax.set_zlabel('Loss')

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

#ax.view_init(30, 45)
ax.view_init(30, -45)
plt.show()

# while True:
#     plt.draw()
#     plt.pause(.001)
# for angle in range(0, 360):
#     print(angle)
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)