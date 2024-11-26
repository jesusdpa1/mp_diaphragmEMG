"""
Author: jpenaloza
Description: pipeline to plot longitudinal response

# choose file to plot
# data files Streeter lab:
#// [2] A21-2wk_streeterLab.smrx
# [3] A23-2wk_streeterLab.smrx
# [4] A23-4wk_streeterLab.smrx
# [5] A23-8wk_streeterLab.smrx
# [6] A23-BL_streeterLab.smrx
#// [7] A6-BL.smrx

# data files DaleLab:
# [0] A00-40Wk_daleLab.smrx
# [1] A00-BL_daleLab.smrx


data bandpass filtered 4Hz to 1000Hz butter, order 4
"""

# %%
# load the libraries required to run this script
# filesystem paths
from pathlib import Path

# Plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from mp_emgdiaphragm.mpstyle import load_plt_config, ColorPalette

# Data loading and manipulation
from neo.io import CedIO
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# units
# import quantities as pq

# filter and envelop functions
from mp_emgdiaphragm.gstats import dc_removal
from mp_emgdiaphragm.filters import notch_filter, butter_bandpass_filter
from mp_emgdiaphragm.linear_envelope_methods import (
    get_rms_envelope,
)


def print_list(x: list):
    print("\n".join(f"[{i}] {x.name}" for i, x in enumerate(x)))


# %%
load_plt_config()
colors_ = ColorPalette()
# define home (Drive were data is located)
home = Path.home()
# define the path to the data file
data_path = r"Documents\data\Methods-Paper_EMG\longitudinal-data"

# build the path to the data file
data_working_path = home.joinpath(data_path)

print(data_working_path)
# %%
# list all the files within the directory
data_list_path = list(data_working_path.glob("*.smrx"))
data_list_path.sort()
print_list(data_list_path)
# %%
# Load baseline recording to extract the min max values for future normalization
# A23-BL_streeterLab.smrx or A00-BL_daleLab.smrx
# (use for the group plotting, ie. streeterLab for streeterLab, daleLab-> daleLab)
baseline_working_path = data_list_path[1]
# %%
# load EUPNIC data using CedIO
baseline_reader = CedIO(filename=baseline_working_path)
baseline_data = baseline_reader.read()
baseline_data_block = baseline_data[0].segments[0]
print(baseline_working_path.name)
baseline_data_block
# %%
baseline_emg_r = baseline_data_block.analogsignals[0]
fs = float(baseline_emg_r.sampling_rate)  # assuming all sampling rates are the same
# %%
# potential gain discrepancies between dalelab and streeter lab
# multiply data by 10 to get 10x gain
baseline_r = baseline_emg_r.magnitude.squeeze() * 10
baseline_notch_r = notch_filter(baseline_r, fs=fs)
baseline_bp_r = dc_removal(
    butter_bandpass_filter(
        tsx=baseline_notch_r, lowcut=4, highcut=1000, fs=fs, order=4
    ),
    method="median",
)
# %%
# individual plots
# getting the envelope

title_name = baseline_working_path.name
start_time = 0
end_time = 10
start_sample = int(start_time * fs)
end_sample = int(end_time * fs)
baseline_sample_r = baseline_bp_r[start_sample:end_sample]

overlap = 0.99
window_type = "rectangular"
window_size = int(0.055 * fs)  # window size
window_beta = None
rectify_method = "none"

baseline_sample_r_rms = get_rms_envelope(
    baseline_sample_r, overlap, window_type, window_size, window_beta, rectify_method
)  # eupnic breathing moving rms envelope right
baseline_scaler = MinMaxScaler()
baseline_sample_r_rms_norm = baseline_scaler.fit_transform(
    baseline_sample_r_rms[..., np.newaxis]
)[:, 0]
plt.plot(baseline_sample_r_rms)


### END OF NORMALIZATION ACQUISITION BACK TO PLOTTING ###

# %%
# Plotting all values based on the same baseline mix max values
# choose file to plot
recording_working_path = data_list_path[1]
# %%
# load EUPNIC data using CedIO
print(recording_working_path.name)
recording_reader = CedIO(filename=recording_working_path)
recording_data = recording_reader.read()
recording_data_block = recording_data[0].segments[0]
recording_data_block
# %%
tsx_emg_r = recording_data_block.analogsignals[0]
fs = float(tsx_emg_r.sampling_rate)  # assuming all sampling rates are the same
# %%
trim_ = int(fs * 10)
tsx_r = tsx_emg_r.magnitude.squeeze() * 10
tsx_notch_r = notch_filter(tsx_r, fs=fs)
tsx_bp_r = dc_removal(
    butter_bandpass_filter(tsx=tsx_notch_r, lowcut=4, highcut=1000, fs=fs, order=4),
    method="mean",
)[0:trim_]
# %%
# individual plots
# getting the envelope

title_name = recording_working_path.name
start_time = 0
end_time = 10
start_sample = int(start_time * fs)
end_sample = int(end_time * fs)
tsx_sample_r = tsx_bp_r[start_sample:end_sample]

overlap = 0.99
window_type = "rectangular"
window_size = int(0.055 * fs)  # window size
window_beta = None
rectify_method = "none"

tsx_sample_r_rms = get_rms_envelope(
    tsx_sample_r, overlap, window_type, window_size, window_beta, rectify_method
)  # eupnic breathing moving rms envelope right
# using the baseline scaler
tsx_sample_r_rms_norm = baseline_scaler.transform(tsx_sample_r_rms[..., np.newaxis])[
    :, 0
]


time = np.linspace(start_time, end_time, tsx_sample_r.__len__())

fig, axs = plt.subplots(1, 1, figsize=(15, 6))
fsize = 30
markersize = 20
# Subplot 0
# Subplot 1
axs.set_title(title_name, fontsize=fsize, fontweight="bold", pad=20)
# Right side plot with RMS envelop
axs.plot(
    time,
    tsx_sample_r,
    color=colors_.colorplot["5"],
    alpha=0.95,
    linewidth=0.5,
    label="pre-processed",
)

axs.plot(
    time,
    tsx_sample_r_rms_norm,
    color="#0070FF",
    label="RMS LE",
    linewidth=0.9,
    path_effects=[pe.Stroke(linewidth=1, foreground="#394555"), pe.Normal()],
)

axs.fill_between(
    time,
    tsx_sample_r_rms_norm,
    0,
    where=(tsx_sample_r_rms_norm < 1.1),
    color="#0070FF",
    alpha=0.3,
)

axs.legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color="#394555",
            label="Norm RMS LE",
            markerfacecolor="#0070FF",
            markersize=markersize,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["5"],
            alpha=0.5,
            label="right hemi-diaphragm",
            markerfacecolor=colors_.colorplot["5"],
            markersize=markersize,
        ),
    ],
    prop={"weight": "bold", "size": 18},
    loc="lower right",
    bbox_to_anchor=(1, 0),
    borderaxespad=1,
)

# Set the same y-axis range for both subplots
ymin = -0.8
ymax = 1.2

axs.set_ylim(ymin, ymax)
axs.set_ylabel("V", fontsize=30, rotation=0, ha="center", va="center", labelpad=25)
axs.tick_params(axis="y", labelsize=20)

axs.set_xlabel("Time (s)", fontsize=30, labelpad=25)
axs.tick_params(axis="x", labelsize=20)
fig.tight_layout()

plt.tight_layout()  # Adjust the layout
plt.show()
# %%
fig.savefig(
    f"../../figures/composed_{recording_working_path.name}-recording_10xgain.png",
    dpi=600,
)
# %%
