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
import matplotlib.patches as patches
from mp_emgdiaphragm.mpstyle import load_plt_config, ColorPalette

# Data loading and manipulation
from neo.io import CedIO
import numpy as np


# units
# import quantities as pq

# filter and envelop functions
from mp_emgdiaphragm.gstats import min_max_normalization, dc_removal
from mp_emgdiaphragm.filters import notch_filter, butter_bandpass_filter
from mp_emgdiaphragm.linear_envelope_methods import (
    get_mean_envelope,
    get_rms_envelope,
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
    tsx=tsx_emg_notch, lowcut=4, highcut=1000, fs=fs, order=4
)
# %%
# choose the window size to work with

start = int(10 * fs)
end = int(45 * fs)

# plot the original data vs the notch+bp filter
plt.plot(tsx_emg.magnitude.squeeze()[start:end])
plt.plot(tsx_emg_bp[start:end])
# %%
# envelop
tsx_sample = tsx_emg_bp[start:end]  # sample dataset
overlap = 0.99
window_type = "rectangular"
window_size = int(0.055 * fs)  # window size
window_beta = None
rectify_method = "square"

tsx_ma = get_mean_envelope(
    tsx_sample, overlap, window_type, window_size, window_beta, rectify_method
)  # moving average envelope

tsx_rms = get_rms_envelope(
    tsx_sample, overlap, window_type, window_size, window_beta
)  # moving rms envelope

tsx_lp = lp_envelope(tsx_sample, 4, fs)  # low-pass filter
tsx_tdt = tdt_envelope(tsx_sample, 4, fs)  # tdt envelope
# %%
# normalize all the envelopes for vizualization

tsx_ma_norm = min_max_normalization(dc_removal(tsx_ma, method="mean"))
tsx_rms_norm = min_max_normalization(dc_removal(tsx_rms, method="mean"))
tsx_lp_norm = min_max_normalization(dc_removal(tsx_lp, method="mean"))
tsx_tdt_norm = min_max_normalization(dc_removal(tsx_tdt, method="mean"))

# %%
# get the relative time to use in the plot
# this is calculated by getting the length of the array and multiply it by 1/fs

time = np.arange(0, tsx_tdt_norm.__len__()) * (1 / fs)

fig, axs = plt.subplots(4, 1, figsize=(15, 10))
fsize = 30
markersize = 20
# Subplot 0
axs[0].plot(
    time,
    tsx_sample,
    color=colors_.colorplot["6"],
    alpha=1,
    linewidth=0.5,
    label="pre-processed",
)
axs[0].set_title(
    "Pre-processed unilateral diaphragm EMG signal",
    fontsize=fsize,
    fontweight="bold",
)
axs[0].set_xticklabels([])
axs[0].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["5"],
            label="Pre-processed",
            markerfacecolor=colors_.colorplot["6"],
            markersize=markersize,
        ),
    ],
    prop={"weight": "bold", "size": 18},
    loc="upper right",
    bbox_to_anchor=(1, 0.5),
    borderaxespad=1,
)


# Subplot 1
# Filtered signal
axs[1].plot(
    time,
    tsx_sample,
    color=colors_.colorplot["6"],
    alpha=0.85,
    linewidth=0.5,
    label="pre-processed",
)
# Linear Envelope
axs[1].plot(time, tsx_ma_norm, color=colors_.colorplot["1"], label="MA LE")
axs[1].fill_between(
    time,
    tsx_ma_norm,
    0,
    where=(tsx_ma_norm < 1.1),
    color=colors_.colorplot["2"],
    alpha=0.15,
)
axs[1].set_title(
    "Normalized Moving Averager (window size = 0.055s)",
    fontsize=fsize,
    fontweight="bold",
)
axs[1].set_xticklabels([])
axs[1].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["1"],
            label="MA LE",
            markerfacecolor=colors_.colorplot["2"],
            markersize=markersize,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["5"],
            label="Pre-processed",
            alpha=0.80,
            markerfacecolor=colors_.colorplot["6"],
            markersize=markersize,
        ),
    ],
    prop={"weight": "bold", "size": 18},
    loc="upper right",
    bbox_to_anchor=(1, 0.5),
    borderaxespad=1,
)

# Subplot 2
# filtered Signal
axs[2].plot(
    time,
    tsx_sample,
    color=colors_.colorplot["6"],
    alpha=0.85,
    linewidth=0.5,
    label="pre-processed",
)
# Linear Envelope RMS
axs[2].plot(time, tsx_rms_norm, color="#0070FF", label="RMS LE")
axs[2].fill_between(
    time,
    tsx_rms_norm,
    0,
    where=(tsx_rms_norm < 1.1),
    color="#0070FF",
    alpha=0.15,
)
axs[2].set_title(
    "Normalized Moving RMS (window size = 0.055s)", fontsize=fsize, fontweight="bold"
)
axs[2].set_xticklabels([])
axs[2].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color="#394555",
            label="RMS LE",
            markerfacecolor="#0070FF",
            markersize=markersize,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["5"],
            label="Pre-processed",
            alpha=0.80,
            markerfacecolor=colors_.colorplot["6"],
            markersize=markersize,
        ),
    ],
    prop={"weight": "bold", "size": 18},
    loc="upper right",
    bbox_to_anchor=(1, 0.5),
    borderaxespad=1,
)

# Subplot 3
# Filtered signal
axs[3].plot(
    time,
    tsx_sample,
    color=colors_.colorplot["6"],
    alpha=0.85,
    linewidth=0.5,
    label="pre-processed",
)
# Linear Envelope
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
    "Normalized Low-Pass filter (cutOff = 4Hz)", fontsize=fsize, fontweight="bold"
)
axs[3].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["7"],
            label="LP LE",
            markerfacecolor=colors_.colorplot["8"],
            markersize=markersize,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["5"],
            label="Pre-processed",
            alpha=0.80,
            markerfacecolor=colors_.colorplot["6"],
            markersize=markersize,
        ),
    ],
    prop={"weight": "bold", "size": 18},
    loc="upper right",
    bbox_to_anchor=(1, 0.5),
    borderaxespad=1,
)


for ax in axs:
    ax.set_ylabel("A.Units", fontsize=30, weight="bold", labelpad=10)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylim(-1.1, 1.1)

axs[3].set_xlabel("Time (s)", fontsize=30, labelpad=15)
axs[3].tick_params(axis="x", labelsize=20)


# adding a rectangle to define the different behaviors
highlight_times = dict(eupnea_base=[9.8, 12.4])

for behavior, time_ in highlight_times.items():
    rect = patches.Rectangle(
        (time_[0], (-1 * axs[0].get_ylim()[1]) + 0.1),
        time_[1] - time_[0],
        2,
        linewidth=3,
        edgecolor="black",
        facecolor="none",
        transform=axs[0].transData,
        alpha=0.7,
    )

    fig.add_artist(rect)
plt.tight_layout()  # Adjust the layout
plt.show()
# %%
fig.savefig("../../figures/composed_envelop_comparison_35seconds.png", dpi=600)

# %%
# save data to csv
# import pandas as pd

# # %%
# data_to_csv = dict(
#     {
#         "time": time,
#         "pre-processed": tsx_sample,
#         "ma": tsx_ma,
#         "mrms": tsx_rms,
#         "lp": tsx_lp,
#     }
# )

# # %%
# df_ = pd.DataFrame(data_to_csv)

# # %%
# df_.to_csv("../../data/composed_envelop_comparison_35seconds.csv")
# %%
# Plot zoom area
# %%
start_zoomed = int(9.8 * fs)
end_zoomed = int(12.4 * fs)

tsx_sample_zoomed = tsx_sample[start_zoomed:end_zoomed]

tsx_ma_norm_zoomed = tsx_ma_norm[start_zoomed:end_zoomed]
tsx_rms_norm_zoomed = tsx_rms_norm[start_zoomed:end_zoomed]
tsx_lp_norm_zoomed = tsx_lp_norm[start_zoomed:end_zoomed]
tsx_tdt_norm_zoomed = tsx_tdt_norm[start_zoomed:end_zoomed]


time_zoomed = time[start_zoomed:end_zoomed]

fig, axs = plt.subplots(4, 1, figsize=(15, 15))
fsize = 30
markersize = 20
# Subplot 0
axs[0].plot(
    time_zoomed,
    tsx_sample_zoomed,
    color=colors_.colorplot["6"],
    alpha=1,
    linewidth=0.8,
    label="pre-processed",
)
axs[0].set_title(
    "Pre-processed unilateral diaphragm EMG signal",
    fontsize=fsize,
    fontweight="bold",
)
axs[0].set_xticklabels([])
axs[0].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["5"],
            label="Pre-processed",
            markerfacecolor=colors_.colorplot["6"],
            markersize=markersize,
        ),
    ],
    prop={"weight": "bold", "size": 18},
    loc="upper right",
    bbox_to_anchor=(1, 0.5),
    borderaxespad=1,
)


# Subplot 1
# Filtered signal
axs[1].plot(
    time_zoomed,
    tsx_sample_zoomed,
    color=colors_.colorplot["6"],
    alpha=0.85,
    linewidth=0.8,
    label="pre-processed",
)
# Linear Envelope
axs[1].plot(
    time_zoomed,
    tsx_ma_norm_zoomed,
    color=colors_.colorplot["1"],
    linewidth=3,
    label="MA LE",
)
axs[1].fill_between(
    time_zoomed,
    tsx_ma_norm_zoomed,
    0,
    where=(tsx_ma_norm_zoomed < 1.1),
    color=colors_.colorplot["2"],
    alpha=0.15,
)
axs[1].set_title(
    "Normalized Moving Averager (window size = 0.055s)",
    fontsize=fsize,
    fontweight="bold",
)
axs[1].set_xticklabels([])
axs[1].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["1"],
            label="MA LE",
            markerfacecolor=colors_.colorplot["2"],
            markersize=markersize,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["5"],
            label="Pre-processed",
            alpha=0.80,
            markerfacecolor=colors_.colorplot["6"],
            markersize=markersize,
        ),
    ],
    prop={"weight": "bold", "size": 18},
    loc="upper right",
    bbox_to_anchor=(1, 0.5),
    borderaxespad=1,
)

# Subplot 2
# filtered Signal
axs[2].plot(
    time_zoomed,
    tsx_sample_zoomed,
    color=colors_.colorplot["6"],
    alpha=0.85,
    linewidth=0.8,
    label="pre-processed",
)
# Linear Envelope RMS
axs[2].plot(
    time_zoomed, tsx_rms_norm_zoomed, color="#0070FF", linewidth=3, label="RMS LE"
)
axs[2].fill_between(
    time_zoomed,
    tsx_rms_norm_zoomed,
    0,
    where=(tsx_rms_norm_zoomed < 1.1),
    color="#0070FF",
    alpha=0.15,
)
axs[2].set_title(
    "Normalized Moving RMS (window size = 0.055s)", fontsize=fsize, fontweight="bold"
)
axs[2].set_xticklabels([])
axs[2].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color="#394555",
            label="RMS LE",
            markerfacecolor="#0070FF",
            markersize=markersize,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["5"],
            label="Pre-processed",
            alpha=0.80,
            markerfacecolor=colors_.colorplot["6"],
            markersize=markersize,
        ),
    ],
    prop={"weight": "bold", "size": 18},
    loc="upper right",
    bbox_to_anchor=(1, 0.5),
    borderaxespad=1,
)

# Subplot 3
# Filtered signal
axs[3].plot(
    time_zoomed,
    tsx_sample_zoomed,
    color=colors_.colorplot["6"],
    alpha=0.85,
    linewidth=0.8,
    label="pre-processed",
)
# Linear Envelope
axs[3].plot(
    time_zoomed,
    tsx_lp_norm_zoomed,
    color=colors_.colorplot["7"],
    linewidth=3,
    label="LP LE",
)
axs[3].fill_between(
    time_zoomed,
    tsx_lp_norm_zoomed,
    0,
    where=(tsx_lp_norm_zoomed < 1.1),
    color=colors_.colorplot["8"],
    alpha=0.15,
)
axs[3].set_title(
    "Normalized Low-Pass filter (cutOff = 4Hz)", fontsize=fsize, fontweight="bold"
)
axs[3].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["7"],
            label="LP LE",
            markerfacecolor=colors_.colorplot["8"],
            markersize=markersize,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["5"],
            label="Pre-processed",
            alpha=0.80,
            markerfacecolor=colors_.colorplot["6"],
            markersize=markersize,
        ),
    ],
    prop={"weight": "bold", "size": 18},
    loc="upper right",
    bbox_to_anchor=(1, 0.5),
    borderaxespad=1,
)


for ax in axs:
    ax.set_ylabel("A.Units", fontsize=30, weight="bold", labelpad=10)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylim(-1.1, 1.1)

axs[3].set_xlabel("Time (s)", fontsize=30, labelpad=15)
axs[3].tick_params(axis="x", labelsize=20)


plt.tight_layout()  # Adjust the layout
plt.show()
# %%
fig.savefig("../../figures/composed_envelop_comparison_35seconds_ZOOMED.png", dpi=600)

# %%
