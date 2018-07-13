__author__ = 'AlexLlamas'
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import math


def plot_x_and_y(x, y):
    fig = plt.figure(2)
    gs = gridspec.GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[:1, :1])
    ax1.plot(x, color='blue')
    # ax1.set_xlim(0, x.shape[0])
    # ax1.set_ylim(-10, 10)
    # ax1.set_yticklabels([])
    # ax1.set_xticklabels([])
    ax1.set_xlabel('samples in x')
    ax1.set_ylabel('values')

    ax2 = fig.add_subplot(gs[1, :1])
    ax2.plot(y, color='green')
    # ax2.set_xlim(0, y.shape[0])
    # ax2.set_ylim(-10, 10)
    ax2.set_xlabel('samples in y')
    ax2.set_ylabel('values')
    gs.update(wspace=0.5, hspace=0.5)

    fig.show()

def plot_sample(x, y, numBinx, numBiny):

    fig = plt.figure(1)
    gs = gridspec.GridSpec(4,4)
    ax1 = fig.add_subplot(gs[:3, :3])
    ax1.scatter(x, y, color='blue')
    # ax1.set_xlim(-10, 10)
    # ax1.set_ylim(-10, 10)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')


    ax2 = fig.add_subplot(gs[3,:3])
    ax2.hist(x, numBinx, facecolor='g')
    ax2.set_xticklabels([])
    ax2.yaxis.set_visible(False)
    ax2.set_xlabel('Histogram of X')


    ax3 = fig.add_subplot(gs[:3, 3])
    ax3.hist(y, numBiny, orientation='horizontal', facecolor='g')
    ax3.set_yticklabels([])
    ax3.xaxis.set_visible(False)
    ax3.set_ylabel('Histogram of Y')
    """
    ax4 = fig.add_subplot(gs[3, 3])
    H, xedges, yedges = np.histogram2d(y, x, bins=(numBinx, numBiny))
    X, Y = np.meshgrid(xedges, yedges)
    ax4.pcolormesh(X, Y, H)
    ax4.set_aspect('equal')
    ax4.xaxis.set_visible(False)
    ax4.yaxis.set_visible(False)
    """
    gs.update(wspace=0.5, hspace=0.5)
    # H, xedges, yedges = np.histogram2d(y, x, bins=(numBinx, numBiny))
    # print str(H)
    plt.show()
