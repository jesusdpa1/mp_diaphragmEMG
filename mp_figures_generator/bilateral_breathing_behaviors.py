"""
Author: jpenaloza
Description: script to compare envelop calculation to ffc comb filter
"""

# %%
# load the libraries required to run this script
# filesystem paths

from pathlib import Path

# Plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from mp_emgdiaphragm.mpstyle import load_plt_config, ColorPalette

# Data loading and manipulation
from neo.io import CedIO
import numpy as np

# units
# import quantities as pq

# filter and envelop functions
from mp_emgdiaphragm.filters import notch_filter, butter_bandpass_filter

# %%
load_plt_config()
colors_ = ColorPalette()
# define home (Drive were data is located)
home = Path.home()
# define the path to the data file
eupneic_path = r"Documents\data\Methods-Paper_EMG\A12_Baseline_Eupnea.smrx"
sniffing_path = r"Documents\data\Methods-Paper_EMG\A12_Baseline_SniffingLonger.smrx"
sigh_path = r"Documents\data\Methods-Paper_EMG\A12_Baseline_Sigh.smrx"
# build the path to the data file
eupneic_working_path = home.joinpath(eupneic_path)
sniffing_working_path = home.joinpath(sniffing_path)
sigh_working_path = home.joinpath(sigh_path)

print(eupneic_working_path)
print(sniffing_working_path)
print(sigh_working_path)
# %%
# load EUPNIC data using CedIO
eupneic_reader = CedIO(filename=eupneic_working_path)
eupneic_data = eupneic_reader.read()
eupneic_data_block = eupneic_data[0].segments[0]


# load SNIFFING DATA data using CedIO
sniffing_reader = CedIO(filename=sniffing_working_path)
sniffing_data = sniffing_reader.read()
sniffing_data_block = sniffing_data[0].segments[0]


# load SIGH data using CedIO
sigh_reader = CedIO(filename=sigh_working_path)
sigh_data = sigh_reader.read()
sigh_data_block = sigh_data[0].segments[0]

# %%
# extract the timeseries to work with and its sample frenquency for each breathing behavior
# Eupnic breathing
tsx_emg_eupneic_r = eupneic_data_block.analogsignals[0]
tsx_emg_eupneic_l = eupneic_data_block.analogsignals[1]
# Sniffing
tsx_emg_sniffing_r = sniffing_data_block.analogsignals[0]
tsx_emg_sniffing_l = sniffing_data_block.analogsignals[1]
# Sigh
tsx_emg_sigh_r = sigh_data_block.analogsignals[0]
tsx_emg_sigh_l = sigh_data_block.analogsignals[1]
# extract the sampling frequency
fs = float(tsx_emg_eupneic_r.sampling_rate)  # assuming all sampling rates are the same
for ts in [tsx_emg_eupneic_r, tsx_emg_sniffing_r, tsx_emg_sigh_r]:
    print(f"sampling rate: {ts.sampling_rate}")
# %%
# eupneic breathing
# Right
tsx_eupneic_r = tsx_emg_eupneic_r.magnitude.squeeze()
tsx_eupneic_notch_r = notch_filter(tsx_eupneic_r, fs=fs)
tsx_eupneic_bp_r = butter_bandpass_filter(
    tsx=tsx_eupneic_notch_r, lowcut=0.1, highcut=1000, fs=fs, order=4
)
# Left
tsx_eupneic_l = tsx_emg_eupneic_l.magnitude.squeeze()
tsx_eupneic_notch_l = notch_filter(tsx_eupneic_l, fs=fs)
tsx_eupneic_bp_l = butter_bandpass_filter(
    tsx=tsx_eupneic_notch_l, lowcut=0.1, highcut=1000, fs=fs, order=4
)


# sniffing breathing
# Right
tsx_sniffing_r = tsx_emg_sniffing_r.magnitude.squeeze()
tsx_sniffing_notch_r = notch_filter(tsx_sniffing_r, fs=fs)
tsx_sniffing_bp_r = butter_bandpass_filter(
    tsx=tsx_sniffing_notch_r, lowcut=0.1, highcut=1000, fs=fs, order=4
)
# Left
tsx_sniffing_l = tsx_emg_sniffing_l.magnitude.squeeze()
tsx_sniffing_notch_l = notch_filter(tsx_sniffing_r, fs=fs)
tsx_sniffing_bp_l = butter_bandpass_filter(
    tsx=tsx_sniffing_notch_l, lowcut=0.1, highcut=1000, fs=fs, order=4
)

# sigh breathing
# Right
tsx_sigh_r = tsx_emg_sigh_r.magnitude.squeeze()
tsx_sigh_notch_r = notch_filter(tsx_sigh_r, fs=fs)
tsx_sigh_bp_r = butter_bandpass_filter(
    tsx=tsx_sigh_notch_r, lowcut=0.1, highcut=1000, fs=fs, order=4
)
# Left
tsx_sigh_l = tsx_emg_sigh_l.magnitude.squeeze()
tsx_sigh_notch_l = notch_filter(tsx_sigh_l, fs=fs)
tsx_sigh_bp_l = butter_bandpass_filter(
    tsx=tsx_sigh_notch_l, lowcut=0.1, highcut=1000, fs=fs, order=4
)
# %%

tsx_eupneic_sample_r = tsx_eupneic_bp_r[int(1 * fs) : int(7 * fs)]
tsx_eupneic_sample_l = tsx_eupneic_bp_l[int(1 * fs) : int(7 * fs)]

tsx_sniffing_sample_r = tsx_sniffing_bp_r[int(4 * fs) : int(10 * fs)]
tsx_sniffing_sample_l = tsx_sniffing_bp_l[int(4 * fs) : int(10 * fs)]

tsx_sigh_sample_r = tsx_sigh_bp_r[int(8 * fs) : int(14 * fs)]
tsx_sigh_sample_l = tsx_sigh_bp_l[int(8 * fs) : int(14 * fs)]

for a in [tsx_eupneic_sample_r, tsx_sniffing_sample_r, tsx_sigh_sample_r]:
    print(a.shape)
# %%
# test plot
plt.plot(tsx_eupneic_sample_r)
plt.plot(tsx_sniffing_sample_r)
plt.plot(tsx_sigh_sample_r)

# %%

plt.plot(tsx_eupneic_sample_l)
plt.plot(tsx_sniffing_sample_l)
plt.plot(tsx_sigh_sample_l)
# %%
# generate the figure
# get the relative time to use in the plot
# this is calculated by getting the length of the array and multiply it by 1/fs


time = np.arange(0, tsx_eupneic_sample_r.__len__()) * (1 / fs)

fig, axs = plt.subplots(3, 1, figsize=(10, 12))
fsize = 30
markersize = 20
# Subplot 0
# Right
axs[0].plot(
    time,
    tsx_eupneic_sample_r + 1.2,
    color=colors_.colorplot["1"],
    label="Right",
    linewidth=0.5,
)
# Left
axs[0].plot(
    time,
    tsx_eupneic_sample_l - 1.2,
    color=colors_.colorplot["2"],
    label="Left",
    linewidth=0.5,
)

axs[0].set_title("Eupneic Breathing", fontsize=fsize, fontweight="bold")
axs[0].set_xticklabels([])
axs[0].set_yticklabels([])
axs[0].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["1"],
            label="Right",
            markerfacecolor=colors_.colorplot["1"],
            markersize=markersize,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["2"],
            label="Left",
            markerfacecolor=colors_.colorplot["2"],
            markersize=markersize,
        ),
    ],
    prop={"weight": "bold", "size": 18},
    loc="lower right",
    bbox_to_anchor=(1, 0),
    borderaxespad=1,
)


# Subplot 1
axs[1].plot(
    time,
    tsx_sniffing_sample_r + 1.4,
    color=colors_.colorplot["7"],
    label="Right",
    linewidth=0.5,
)
# Left
axs[1].plot(
    time,
    tsx_sniffing_sample_l - 1.4,
    color=colors_.colorplot["8"],
    label="Left",
    linewidth=0.5,
)

axs[1].set_title("Sniffing", fontsize=fsize, fontweight="bold")
axs[1].set_xticklabels([])
axs[1].set_yticklabels([])
axs[1].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["7"],
            label="Right",
            markerfacecolor=colors_.colorplot["7"],
            markersize=markersize,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["8"],
            label="Left",
            markerfacecolor=colors_.colorplot["8"],
            markersize=markersize,
        ),
    ],
    prop={"weight": "bold", "size": 18},
    loc="lower right",
    bbox_to_anchor=(1, 0),
    borderaxespad=1,
)

# Subplot 2
axs[2].plot(
    time,
    tsx_sigh_sample_r + 1.4,
    color=colors_.colorplot["5"],
    label="Right",
    linewidth=0.5,
)
# Left
axs[2].plot(
    time,
    tsx_sigh_sample_l - 1.4,
    color=colors_.colorplot["6"],
    label="Left",
    linewidth=0.5,
)

axs[2].set_title("Sigh", fontsize=fsize, fontweight="bold")

axs[2].set_yticklabels([])
axs[2].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["5"],
            label="Right",
            markerfacecolor=colors_.colorplot["5"],
            markersize=markersize,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["6"],
            label="Left",
            markerfacecolor=colors_.colorplot["6"],
            markersize=markersize,
        ),
    ],
    prop={"weight": "bold", "size": 18},
    loc="lower right",
    bbox_to_anchor=(1, 0),
    borderaxespad=1,
)

# Set the same y-axis range for both subplots
ymin = (
    min(
        np.min(tsx_eupneic_sample_r),
        np.min(tsx_sniffing_sample_r),
        np.min(tsx_sigh_sample_r),
    )
    * 2.2
)
ymax = (
    max(
        np.max(tsx_eupneic_sample_r),
        np.max(tsx_sniffing_sample_r),
        np.max(tsx_sigh_sample_r),
    )
    * 2.2
)
for ax in axs:
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("A.Units", fontsize=30)
    ax.yaxis.set_major_locator(ticker.LinearLocator(7))

axs[2].set_xlabel("Time (s)", fontsize=30)
axs[2].tick_params(axis="x", labelsize=20)

plt.tight_layout()  # Adjust the layout
plt.show()
# %%
fig.savefig("../../figures/bilateral_breathing_behaviors.png", dpi=600)

# %%
