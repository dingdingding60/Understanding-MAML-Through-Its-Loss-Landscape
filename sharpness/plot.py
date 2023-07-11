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
# def setup_surface_file(args, args.sf, dir_file):
#     # skip if the direction file already exists
#     if os.path.exists(args.sf):
#         f = h5py.File(args.sf, 'r')
#         if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
#             f.close()
#             print ("%s is already set up" % args.sf)
#             return
#     else:
#         raise("no such file")

def count_local_minima(matrix):
    rows, cols = matrix.shape
    local_minima_count = 0

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            current = matrix[i, j]
            neighbors = [
                matrix[i-1, j-1], matrix[i-1, j], matrix[i-1, j+1],
                matrix[i, j-1], matrix[i, j+1],
                matrix[i+1, j-1], matrix[i+1, j], matrix[i+1, j+1],
            ]

            if current < min(neighbors):
                local_minima_count += 1
                
    return local_minima_count
def plot_2d_contour(args, surf_name='train_loss', show=False):
    """Plot 2D contour map and 3D surface."""

    f = h5py.File(args.sf, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)
    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    elif surf_name == 'train_err' or surf_name == 'test_err' :
        Z = 100 - np.array(f[surf_name][:])
    else:
        print ('%s is not found in %s' % (surf_name, args.sf))
        
    print(count_local_minima(Z))
    if args.norm:
        from sklearn.preprocessing import normalize
        Z= normalize(Z)
    print('------------------------------------------------------------------')
    print('plot_2d_contour')
    print('------------------------------------------------------------------')
    print("loading surface file: " + args.sf)
    print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(x), len(y)))
    print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(Z), surf_name, np.min(Z)))
    print(Z)

    if (len(x) <= 1 or len(y) <= 1):
        print('The length of coordinates is not enough for plotting contours')
        return

    # # --------------------------------------------------------------------
    # # Plot 2D contours
    # # --------------------------------------------------------------------
    # fig = plt.figure()
    # CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    # plt.clabel(CS, inline=1, fontsize=8)
    # fig.savefig(args.sf + '_' + surf_name + '_2dcontour' + '.pdf', dpi=300,
    #             bbox_inches='tight', format='pdf')

    # fig = plt.figure()
    # print(args.sf + '_' + surf_name + '_2dcontourf' + '.pdf')
    # CS = plt.contourf(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    # fig.savefig(args.sf + '_1' + surf_name + '_2dcontourf' + '.pdf', dpi=300,
    #             bbox_inches='tight', format='pdf')

    # # --------------------------------------------------------------------
    # # Plot 2D heatmaps
    # # --------------------------------------------------------------------
    # fig = plt.figure()
    # sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
    #                        xticklabels=False, yticklabels=False)
    # sns_plot.invert_yaxis()
    # sns_plot.get_figure().savefig(args.sf + '_' + surf_name + '_2dheat.pdf',
    #                               dpi=300, bbox_inches='tight', format='pdf')

    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------------------------------------------------
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # surf = ax.plot_surface(X, -Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # fig.savefig(args.sf + '_1' + surf_name + '_3dsurface.pdf', dpi=300,
    #             bbox_inches='tight', format='pdf')
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # surf = ax.plot_surface(-X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # fig.savefig(args.sf + '_2' + surf_name + '_3dsurface.pdf', dpi=300,
    #             bbox_inches='tight', format='pdf')
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # surf = ax.plot_surface(-X, -Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # fig.savefig(args.sf + '_3' + surf_name + '_3dsurface.pdf', dpi=300,
    #             bbox_inches='tight', format='pdf')
    # f.close()
    # if show: plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--sf', default='')
    parser.add_argument('--norm', action='store_true', default=False,)
    parser.add_argument('--log', action='store_true', default=False,)
    parser.add_argument('--vmin', type=float,default=0.0001)
    parser.add_argument('--vmax', type=float,default=10.0)
    parser.add_argument('--vlevel', type=float, default=0.2)
    args = parser.parse_args()
    vmin=args.vmin
    vmax=args.vmax
    vlevel= args.vlevel
    plot_2d_contour(args, 'train_loss')