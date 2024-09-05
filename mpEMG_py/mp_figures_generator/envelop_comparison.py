"""
Author: jpenaloza
Description: script to compare envelop calculation
"""

# %%
# load the libraries required to run this script
# filesystem paths
from pathlib import Path

# Plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mp_diaphragmEMG.mpstyle import load_plt_config, ColorPalette

# Data loading and manipulation
from neo.io import CedIO
import numpy as np


# units
# import quantities as pq

# filter and envelop functions
from mp_diaphragmEMG.gstats import min_max_normalization, dc_removal
from mp_diaphragmEMG.filters import notch_filter, butter_bandpass_filter
from mp_diaphragmEMG.linear_envelope_methods import (
    mean_envelope,
    rms_envelope,
    lp_envelope,
    tdt_envelope,
)

# %%
load_plt_config()
colors_ = ColorPalette()
# define home (Drive were data is located)
home = Path.home()
# define the path to the data file
data_path = r"Documents\data\Methods-Paper_EMG\A12_Baseline_Eupnea.smrx"
# build the path to the data file
working_path = home.joinpath(data_path)
print(working_path)
# %%
# load the data using CedIO
reader = CedIO(filename=working_path)
data = reader.read()
data_block = data[0].segments[0]
# %%
# extract the timeseries to work with and its sample frenquency
tsx_emg = data_block.analogsignals[0]
fs = float(tsx_emg.sampling_rate)

tsx_emg_notch = notch_filter(tsx=tsx_emg.magnitude.squeeze(), fs=fs)
tsx_emg_bp = butter_bandpass_filter(
    tsx=tsx_emg_notch, lowcut=0.1, highcut=1000, fs=fs, order=4
)
# %%
# choose the window size to work with

start = int(10 * fs)
end = int(12 * fs)

# plot the original data vs the notch+bp filter
plt.plot(tsx_emg.magnitude.squeeze()[start:end])
plt.plot(tsx_emg_bp[start:end])
# %%
# envelop
tsx_sample = tsx_emg_bp[start:end]  # sample dataset
window_size = int(0.055 * fs)  # window size
tsx_ma = mean_envelope(tsx_sample, window_size)  # moving average envelope
tsx_rms = rms_envelope(tsx_sample, window_size)  # moving rms envelope
tsx_lp = lp_envelope(tsx_sample, 4, fs)  # low-pass filter
tsx_tdt = tdt_envelope(tsx_sample, 4, fs)  # tdt envelope
# %%
# normalize all the envelopes for vizualization

tsx_ma_norm = min_max_normalization(tsx_ma)
tsx_rms_norm = min_max_normalization(tsx_rms)
tsx_lp_norm = min_max_normalization(tsx_lp)
tsx_tdt_norm = min_max_normalization(dc_removal(tsx_tdt, method="mean"))

# %%
# get the relative time to use in the plot
# this is calculated by getting the length of the array and multiply it by 1/fs

time = np.arange(0, tsx_tdt_norm.__len__()) * (1 / fs)

fig, axs = plt.subplots(4, 1, figsize=(15, 12))

# Subplot 0
axs[0].plot(time, tsx_sample, color=colors_.colorplot["1"], label="pre-processed")
axs[0].set_title(
    "pre-processed unilateral diaphragm EMG signal", fontsize=16, fontweight="bold"
)
axs[0].set_xticklabels([])
axs[0].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["1"],
            label="Pre-processed",
            markerfacecolor=colors_.colorplot["1"],
            markersize=15,
        ),
    ],
    prop={"weight": "bold"},
    loc="upper right",
    bbox_to_anchor=(1, 1),
    borderaxespad=1,
)


# Subplot 1
axs[1].plot(
    time, tsx_sample, color=colors_.colorplot["1"], alpha=0.5, label="pre-processed"
)
axs[1].plot(time, tsx_ma_norm, color=colors_.colorplot["3"], label="MA LE")
axs[1].fill_between(
    time,
    tsx_ma_norm,
    0,
    where=(tsx_ma_norm < 1.1),
    color=colors_.colorplot["4"],
    alpha=0.15,
)
axs[1].set_title(
    "Normalized Moving Averager (window size = 0.055s)", fontsize=16, fontweight="bold"
)
axs[1].set_xticklabels([])
axs[1].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["1"],
            alpha=0.5,
            label="Pre-processed",
            markerfacecolor=colors_.colorplot["1"],
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["3"],
            label="MA LE",
            markerfacecolor=colors_.colorplot["4"],
            markersize=15,
        ),
    ],
    prop={"weight": "bold"},
    loc="upper right",
    bbox_to_anchor=(1, 1),
    borderaxespad=1,
)

# Subplot 2
axs[2].plot(
    time, tsx_sample, color=colors_.colorplot["1"], alpha=0.5, label="pre-processed"
)
axs[2].plot(time, tsx_rms_norm, color=colors_.colorplot["5"], label="RMS LE")
axs[2].fill_between(
    time,
    tsx_rms_norm,
    0,
    where=(tsx_rms_norm < 1.1),
    color=colors_.colorplot["6"],
    alpha=0.15,
)
axs[2].set_title(
    "Normalized Moving RMS (window size = 0.055s)", fontsize=16, fontweight="bold"
)
axs[2].set_xticklabels([])
axs[2].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["1"],
            alpha=0.5,
            label="Pre-processed",
            markerfacecolor=colors_.colorplot["1"],
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["5"],
            label="RMS LE",
            markerfacecolor=colors_.colorplot["6"],
            markersize=15,
        ),
    ],
    prop={"weight": "bold"},
    loc="upper right",
    bbox_to_anchor=(1, 1),
    borderaxespad=1,
)

# Subplot 3
axs[3].plot(
    time, tsx_sample, color=colors_.colorplot["1"], alpha=0.5, label="pre-processed"
)
axs[3].plot(time, tsx_lp_norm, color=colors_.colorplot["7"], label="LP LE")
axs[3].fill_between(
    time,
    tsx_lp_norm,
    0,
    where=(tsx_lp_norm < 1.1),
    color=colors_.colorplot["8"],
    alpha=0.15,
)
axs[3].set_title(
    "Normalized Low-Pass filter (cutOff = 4Hz)", fontsize=16, fontweight="bold"
)
axs[3].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["1"],
            alpha=0.5,
            label="Pre-processed",
            markerfacecolor=colors_.colorplot["1"],
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["7"],
            label="LP LE",
            markerfacecolor=colors_.colorplot["8"],
            markersize=15,
        ),
    ],
    prop={"weight": "bold"},
    loc="upper right",
    bbox_to_anchor=(1, 1),
    borderaxespad=1,
)


for ax in axs[1:]:
    ax.set_ylabel("A.Units", fontsize=14)

axs[0].set_ylabel("V", fontsize=14)
axs[3].set_xlabel("Time (s)", fontsize=14)

plt.tight_layout()  # Adjust the layout
plt.show()
# %%
fig.savefig("figs/envelop_comparison.png", dpi=600)

# %%
