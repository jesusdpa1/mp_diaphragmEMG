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
from mp_diaphragmEMG.mpstyle import load_plt_config, ColorPalette

# Data loading and manipulation
from neo.io import CedIO
import numpy as np

# units
# import quantities as pq

# filter and envelop functions
from mp_diaphragmEMG.gstats import min_max_normalization, dc_removal
from mp_diaphragmEMG.filters import notch_filter, butter_bandpass_filter, ffc_filter
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
data_path = r"Documents\data\Methods-Paper_EMG\noisy_signal.smrx"
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

# %%
# choose the window size to work with
start = int(10 * fs)
end = int(13 * fs)

# %%

tsx_sample = tsx_emg.magnitude.squeeze()  # sample dataset
tsx_emg_notch = notch_filter(tsx_sample, fs=fs)
tsx_emg_bp = butter_bandpass_filter(
    tsx=tsx_emg_notch, lowcut=0.1, highcut=1000, fs=fs, order=4
)

tsx_ffc = ffc_filter(tsx_sample, alpha=-1, fc=60.1, fs=fs)
tsx_ffc_notch = notch_filter(tsx_ffc, fs=fs)
tsx_ffc_bp = butter_bandpass_filter(
    tsx=tsx_ffc_notch, lowcut=0.1, highcut=1000, fs=fs, order=4
)
# %%
plt.plot(tsx_sample[start:end])
plt.plot(tsx_emg_bp[start:end])
plt.plot(tsx_ffc_bp[start:end])

# %%
# envelop
tsx_sample = tsx_ffc_bp[start:end]  # sample dataset
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
axs[0].plot(time, tsx_sample, color=colors_.colorplot["1"], label="Original")
axs[0].set_title("Original", fontsize=16, fontweight="bold")
axs[0].legend()


# Subplot 1
axs[1].plot(time, tsx_sample, color=colors_.colorplot["1"], alpha=0.3, label="Original")
axs[1].plot(time, tsx_ma_norm, color=colors_.colorplot["3"], label="MA LE")
axs[1].fill_between(
    time,
    tsx_ma_norm,
    0,
    where=(tsx_ma_norm < 1.1),
    color=colors_.colorplot["4"],
    alpha=0.15,
)
axs[1].set_title("Moving Averager", fontsize=16, fontweight="bold")
axs[1].legend()

# Subplot 2
axs[2].plot(time, tsx_sample, color=colors_.colorplot["1"], alpha=0.3, label="Original")
axs[2].plot(time, tsx_rms_norm, color=colors_.colorplot["5"], label="RMS LE")
axs[2].fill_between(
    time,
    tsx_rms_norm,
    0,
    where=(tsx_rms_norm < 1.1),
    color=colors_.colorplot["6"],
    alpha=0.15,
)
axs[2].set_title("RMS", fontsize=16, fontweight="bold")
axs[2].legend()

# Subplot 3
axs[3].plot(time, tsx_sample, color=colors_.colorplot["1"], alpha=0.3, label="Original")
axs[3].plot(time, tsx_lp_norm, color=colors_.colorplot["7"], label="LP LE")
axs[3].fill_between(
    time,
    tsx_lp_norm,
    0,
    where=(tsx_lp_norm < 1.1),
    color=colors_.colorplot["8"],
    alpha=0.15,
)
axs[3].set_title("LP", fontsize=16, fontweight="bold")
axs[3].legend()


for ax in axs:
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("A.Units", fontsize=14)

plt.tight_layout()  # Adjust the layout
plt.show()
# %%
