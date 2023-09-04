import numpy as np
import torch
import logging
import os
import sys
import scipy.io as sio
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

cylinder_256_512 = sio.loadmat('/mnt/jfs/lyuyanfang/datasets/cylinder/input_256_512_cylidner.mat')['data']
cylinder_128_256 = sio.loadmat('/mnt/jfs/lyuyanfang/datasets/cylinder/input_128_256_cylidner.mat')['data']
cylinder_64_128 = sio.loadmat('/mnt/jfs/lyuyanfang/datasets/cylinder/input_64_128_cylidner.mat')['data']
cylinder_32_64 = sio.loadmat('/mnt/jfs/lyuyanfang/datasets/cylinder/input_32_64_cylidner.mat')['data']


fig, ax = plt.subplots(figsize=(7,5),dpi=400)

gs = GridSpec(384, 512)
ax1 = plt.subplot(gs[:256, :])
ax2 = plt.subplot(gs[256:, :256])
ax3 = plt.subplot(gs[256:, 256:384])
ax4 = plt.subplot(gs[256:, 384:])

data1 = cylinder_256_512
data2 = cylinder_128_256
data3 = cylinder_64_128
data4 = cylinder_32_64

h1 = ax1.imshow(data1, cmap=plt.cm.tab20b)
h2 = ax2.imshow(data2, cmap=plt.cm.tab20b)
# h1 = ax1.imshow(data1,cmap=plt.cm.jet)
# h2 = ax2.imshow(data2,cmap=plt.cm.jet)
h3 = ax3.imshow(data3, cmap=plt.cm.tab20b)
h4 = ax4.imshow(data4, cmap=plt.cm.tab20b)

ax1.set_title('$Prediction$', fontsize=10)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h1, cax=cax1)

ax2.set_title('$Original$', fontsize=10)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h2, cax=cax2)

ax3.set_title('$Error$', fontsize=10)
divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h3, cax=cax3)

ax4.set_title('$Error$', fontsize=10)
divider4 = make_axes_locatable(ax4)
cax4 = divider4.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h4, cax=cax4)

plt.show()
plt.savefig('/home/ubuntu/new_sample1/Data/X.png')
plt.close()