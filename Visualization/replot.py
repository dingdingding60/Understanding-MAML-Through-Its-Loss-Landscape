

import os
import fnmatch
import plot_2D
import h5py
import argparse
import copy
import h5py
import torch
import time
import socket
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import h5py
import argparse
import numpy as np
from os.path import exists
import seaborn as sns
# def setup_surface_file(args, sf, dir_file):
#     # skip if the direction file already exists
#     if os.path.exists(sf):
#         f = h5py.File(sf, 'r')
#         if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
#             f.close()
#             print ("%s is already set up" % sf)
#             return
#     else:
#         raise("no such file")
def plot_2d_contour(sf, surf_name='train_loss', show=False):
    """Plot 2D contour map and 3D surface."""

    f = h5py.File(sf, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)
    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    elif surf_name == 'train_err' or surf_name == 'test_err' :
        Z = 100 - np.array(f[surf_name][:])
    else:
        print ('%s is not found in %s' % (surf_name, sf))
    # if args.norm:
    #     from sklearn.preprocessing import normalize
    #     Z= normalize(Z)
    # print('------------------------------------------------------------------')
    # print('plot_2d_contour')
    # print('------------------------------------------------------------------')
    # print("loading surface file: " + sf)
    # print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(x), len(y)))
    # print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(Z), surf_name, np.min(Z)))
    # print(Z)

    if (len(x) <= 1 or len(y) <= 1):
        print('The length of coordinates is not enough for plotting contours')
        return

    # # --------------------------------------------------------------------
    # # Plot 2D contours
    # # --------------------------------------------------------------------
    # fig = plt.figure()
    # CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    # plt.clabel(CS, inline=1, fontsize=8)
    # fig.savefig(sf + '_' + surf_name + '_2dcontour' + '.pdf', dpi=300,
    #             bbox_inches='tight', format='pdf')

    # fig = plt.figure()
    # print(sf + '_' + surf_name + '_2dcontourf' + '.pdf')
    # CS = plt.contourf(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    # fig.savefig(sf + '_1' + surf_name + '_2dcontourf' + '.pdf', dpi=300,
    #             bbox_inches='tight', format='pdf')

    # # --------------------------------------------------------------------
    # # Plot 2D heatmaps
    # # --------------------------------------------------------------------
    # fig = plt.figure()
    # sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
    #                        xticklabels=False, yticklabels=False)
    # sns_plot.invert_yaxis()
    # sns_plot.get_figure().savefig(sf + '_' + surf_name + '_2dheat.pdf',
    #                               dpi=300, bbox_inches='tight', format='pdf')

    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------------------------------------------------
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, -Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(sf + '_1' + surf_name + '_3dsurface.pdf', dpi=300,
                bbox_inches='tight', format='pdf')
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(-X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(sf + '_2' + surf_name + '_3dsurface.pdf', dpi=300,
                bbox_inches='tight', format='pdf')
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(-X, -Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(sf + '_3' + surf_name + '_3dsurface.pdf', dpi=300,
                bbox_inches='tight', format='pdf')
    f.close()
    if show: plt.show()


def find_files_with_70_and_h5(root_folder):
    for folder_path, _, file_names in os.walk(root_folder):
        for file_name in file_names:
            if '70' in file_name and "1ttsk" in file_name and fnmatch.fnmatch(file_name, '*.h5'):
                try:
                    plot_2d_contour(os.path.join(folder_path,file_name) )
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")
                    continue
# Replace 'your_folder_path' with the path to the folder you want to search in
find_files_with_70_and_h5('/root/autodl-tmp/checkpoints/miniImagenet')