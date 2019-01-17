import matplotlib
#matplotlib.use("TkAgg")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

#plt.ion() # interactive non-blocking mode
#plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
#neg = np.arange(0, 10, 0.25)
#pos = np.arange(0, 10, 0.25)

min_dist = 0.0
max_dist = 2.0

coef_exp = 2.0
neg_coef = 1.0


neg = np.arange(min_dist, max_dist, (max_dist-min_dist)/100)
pos = np.arange(min_dist, max_dist, (max_dist-min_dist)/100)

#max_dist = 2.0

neg, pos = np.meshgrid(neg, pos)
#R = np.sqrt(neg**2 + pos**2)
#Z = np.sin(R)




# margin loss standard triplet loss
margin = 0.5
Z = np.maximum(np.zeros_like(neg), pos - neg + margin)

for i in range(pos.shape[0]):
    for j in range(pos.shape[1]):
        each_pos = pos[i, j]
        each_neg = neg[i, j]

        # semi-hard
        if each_pos + margin <= each_neg or each_neg <= each_pos:
            Z[i,j] = -1

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
surf = ax.plot_surface(neg, pos, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlabel('neg')
ax.set_ylabel('pos')
ax.set_zlabel('Loss')

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.view_init(30, 45)
plt.show()

# while True:
#     plt.draw()
#     plt.pause(.001)
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)