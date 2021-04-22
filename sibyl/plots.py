import numpy as np
import pandas as pd

from scipy import signal
from scipy import interpolate
from scipy.fftpack import fft, fftfreq

from matplotlib import patches
import matplotlib.pyplot as plt

from typing import List

# Functions courtesy of https://github.com/ijmax/EEG-processing-python/blob/main/topograph.py

CHANNEL_NAMES = ["AF1", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF2"]

def reshape_data_topograph(df: pd.DataFrame):
    ch_data = df[CHANNEL_NAMES].values.reshape((256, 14))
    return np.swapaxes(ch_data, 0, -1)

def get_psds(data, fs=256, f_range=[0.5, 30]):
    '''
    Calculate signal power using Welch method.
    Input: data- mxn matrix (m: number of channels, n: samples of signals)
           fs- Sampling frequency (default 128Hz)
           f_range- Frequency range (default 0.5Hz to 30Hz)
    Output: Power values and PSD values
    '''
    powers = []
    psds = list()
    for sig in data:
        freq, psd = signal.welch(sig, fs)
        idx = np.logical_and(freq >= f_range[0], freq <= f_range[1])
        powers = np.append(powers, sum(psd[idx]))
        psds.append(psd[idx])

    return powers, psds

def plot_topomap(data, ax, fig, draw_cbar=True):
    '''
    Plot topographic plot of EEG data. This specialy design for Emotiv 14 electrode data.
    This can be change for any other arrangement by changing ch_pos (channel position array)
    Input: data- 1D array 14 data values
           ax- Matplotlib subplot object to be plotted every thing
           fig- Matplot lib figure object to draw colormap
           draw_cbar- Visualize color bar in the plot
    '''
    N = 300
    xy_center = [2,2]
    radius = 2

    # electrode positions
    # AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
    ch_pos = [[1,4],[0.1,3], [1.5,3.5], [0.5,2.5],
             [-0.1,2], [0.4,0.4], [1.5,0], [2.5,0],
             [3.6,0.4], [4.1,2], [3.5,2.5], [2.5,3.5],
             [3.9,3], [3,4]]
    x,y = [],[]
    for i in ch_pos:
        x.append(i[0])
        y.append(i[1])

    # interpolate EEG data into grid contour
    xi = np.linspace(-2, 6, N)
    yi = np.linspace(-2, 6, N)
    zi = interpolate.griddata((x, y), data, (xi[None,:], yi[:,None]), method='cubic')

    dr = xi[1] - xi[0]
    for i in range(N):
        for j in range(N):
            r = np.sqrt((xi[i] - xy_center[0])**2 + (yi[j] - xy_center[1])**2)
            if (r - dr/2) > radius:
                zi[j,i] = "nan"

    # draw topograph contour
    dist = ax.contourf(xi, yi, zi, 60, cmap = plt.get_cmap('coolwarm'), zorder = 1)
    ax.contour(xi, yi, zi, 15, linewidths = 0.5,colors = "grey", zorder = 2)

    # draw color bar
    if draw_cbar:
        cbar = fig.colorbar(dist, ax=ax, format='%.1e')
        cbar.ax.tick_params(labelsize=8)

    # draw electrodes
    ax.scatter(x, y, marker = 'o', c = 'b', s = 15, zorder = 3)

    # draw head
    circle = patches.Circle(xy = xy_center, radius = radius, edgecolor = "k", facecolor = "none", zorder=4)
    ax.add_patch(circle)

    for _, spine in ax.spines.items():
        spine.set_linewidth(0)

    ax.set_xticks([])
    ax.set_yticks([])

    # draw ears
    circle = patches.Ellipse(xy = [0,2], width = 0.4, height = 1.0, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    circle = patches.Ellipse(xy = [4,2], width = 0.4, height = 1.0, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)

    # draw nose
    xy = [[1.6,3.6], [2,4.3],[2.4,3.6]]
    polygon = patches.Polygon(xy = xy, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(polygon)

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)

    return ax

def plot_eeg(data, labels: List[str], ax: plt.Axes, fs: int=256, vspace: int=50):
    bases = vspace * np.arange(data.shape[1])
    eeg_values = data + bases

    time = np.arange(eeg_values.shape[0]) / fs

    ax.plot(time, eeg_values, color="k")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels")

    ax.yaxis.set_ticks(bases)
    ax.yaxis.set_ticklabels(labels)

def plot_eeg_fft(data, ax: plt.Axes, fs: int=256):
    N = data.shape[0]
    yf = fft(data)
    xf = fftfreq(N, 1 / fs)[:N//2]

    ax.plot(xf, abs(yf[:N//2]))
    ax.grid()
    pass
