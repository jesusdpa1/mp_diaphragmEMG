"""
Author: jpenaloza
Description: script to generate plots for eupnic breathing awake 
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
from mp_diaphragmEMG.filters import notch_filter, butter_bandpass_filter

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
tsx_emg_r = data_block.analogsignals[0]
tsx_emg_l = data_block.analogsignals[1]
fs = float(tsx_emg_r.sampling_rate)

# %%
# choose the window size to work with
start = int(5 * fs)
end = int(12 * fs)

# %%
# right diaphragm
tsx_right = tsx_emg_r.magnitude.squeeze()
tsx_right_notch = notch_filter(tsx_right, fs=fs)
tsx_right_bp_r = butter_bandpass_filter(
    tsx=tsx_right_notch, lowcut=0.1, highcut=1000, fs=fs, order=4
)

# left diaphragm

tsx_left = tsx_emg_l.magnitude.squeeze()  # sample dataset
tsx_left_notch = notch_filter(tsx_left, fs=fs)
tsx_left_bp_r = butter_bandpass_filter(
    tsx=tsx_left_notch, lowcut=0.1, highcut=1000, fs=fs, order=4
)
# %%
tsx_right_sample = tsx_right_bp_r[start:end]
tsx_left_sample = tsx_left_bp_r[start:end]
# %%
plt.plot(tsx_right_sample)
plt.plot(tsx_left_sample)

# %%
# get the relative time to use in the plot
# this is calculated by getting the length of the array and multiply it by 1/fs

time = np.arange(0, tsx_right_sample.__len__()) * (1 / fs)

fig, axs = plt.subplots(2, 1, figsize=(15, 6))

# Subplot 0
axs[0].plot(
    time, tsx_right_sample, color=colors_.colorplot["1"], label="Right", linewidth=0.5
)
axs[0].set_title(
    "Awake eupnic breathing bilateral diaphragm EMG recording",
    fontsize=16,
    fontweight="bold",
)
axs[0].set_xticklabels([])
axs[0].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["1"],
            label="Right",
            markerfacecolor=colors_.colorplot["1"],
            markersize=15,
        ),
    ],
    prop={"weight": "bold"},
    loc="lower right",
    bbox_to_anchor=(1, 0),
    borderaxespad=1,
)

# Subplot 1
axs[1].plot(
    time, tsx_left_sample, color=colors_.colorplot["2"], label="Left", linewidth=0.5
)
axs[1].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["2"],
            label="Left",
            markerfacecolor=colors_.colorplot["2"],
            markersize=15,
        ),
    ],
    prop={"weight": "bold"},
    loc="lower right",
    bbox_to_anchor=(1, 0),
    borderaxespad=1,
)

# Set the same y-axis range for both subplots
ymin = min(np.min(tsx_right_sample), np.min(tsx_left_sample)) * 1.1
ymax = max(np.max(tsx_right_sample), np.max(tsx_left_sample)) * 1.1
for ax in axs:
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("V", fontsize=14)

axs[1].set_xlabel("Relative Time (s)", fontsize=14)

plt.tight_layout()  # Adjust the layout
plt.show()
# %%
fig.savefig("figs/awake_eupnic_original_left-right.png", dpi=600)

# %%
