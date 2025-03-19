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
from mp_diaphragmEMG.filters import notch_filter, butter_bandpass_filter

# %%
load_plt_config()
colors_ = ColorPalette()
# define home (Drive were data is located)
home = Path.home()
# define the path to the data file
eupinic_path = r"Documents\data\Methods-Paper_EMG\A12_Baseline_Eupnea.smrx"
sniffing_path = r"Documents\data\Methods-Paper_EMG\A12_Baseline_Sniffing.smrx"
sigh_path = r"Documents\data\Methods-Paper_EMG\A12_Baseline_Sigh.smrx"
# build the path to the data file
eupinic_working_path = home.joinpath(eupinic_path)
sniffing_working_path = home.joinpath(sniffing_path)
sigh_working_path = home.joinpath(sigh_path)

print(eupinic_working_path)
print(sniffing_working_path)
print(sigh_working_path)
# %%
# load EUPNIC data using CedIO
eupinic_reader = CedIO(filename=eupinic_working_path)
eupinic_data = eupinic_reader.read()
eupinic_data_block = eupinic_data[0].segments[0]

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
tsx_emg_eupinic = eupinic_data_block.analogsignals[0]
tsx_emg_sniffing = sniffing_data_block.analogsignals[0]
tsx_emg_sigh = sigh_data_block.analogsignals[0]

# extract the sampling frequency
fs = float(tsx_emg_eupinic.sampling_rate)  # assuming all sampling rates are the same
for ts in [tsx_emg_eupinic, tsx_emg_sniffing, tsx_emg_sigh]:
    print(f"sampling rate: {ts.sampling_rate}")
# %%
# eupnic breathing
tsx_eupinic = tsx_emg_eupinic.magnitude.squeeze()
tsx_eupinic_notch = notch_filter(tsx_eupinic, fs=fs)
tsx_eupinic_bp = butter_bandpass_filter(
    tsx=tsx_eupinic_notch, lowcut=0.1, highcut=1000, fs=fs, order=4
)

# sniffing breathing
tsx_sniffing = tsx_emg_sniffing.magnitude.squeeze()
tsx_sniffing_notch = notch_filter(tsx_sniffing, fs=fs)
tsx_sniffing_bp = butter_bandpass_filter(
    tsx=tsx_sniffing_notch, lowcut=0.1, highcut=1000, fs=fs, order=4
)

# sigh breathing
tsx_sigh = tsx_emg_sigh.magnitude.squeeze()
tsx_sigh_notch = notch_filter(tsx_sigh, fs=fs)
tsx_sigh_bp = butter_bandpass_filter(
    tsx=tsx_sigh_notch, lowcut=0.1, highcut=1000, fs=fs, order=4
)
# %%

tsx_eupinic_sample = tsx_eupinic_bp[int(1 * fs) : int(7 * fs)]
tsx_sniffing_sample = tsx_sniffing_bp[int(0 * fs) : int(6 * fs)]
tsx_sigh_sample = tsx_sigh_bp[int(8 * fs) : int(14 * fs)]

print(tsx_eupinic_sample.shape)
print(tsx_sniffing_sample.shape)
print(tsx_sigh_sample.shape)
# %%
# test plot
plt.plot(tsx_eupinic_sample)
plt.plot(tsx_sniffing_sample)
plt.plot(tsx_sigh_sample)
# %%
# generate the figure
# get the relative time to use in the plot
# this is calculated by getting the length of the array and multiply it by 1/fs

time = np.arange(0, tsx_eupinic_sample.__len__()) * (1 / fs)

fig, axs = plt.subplots(3, 1, figsize=(15, 8))

# Subplot 0
axs[0].plot(
    time,
    tsx_eupinic_sample,
    color=colors_.colorplot["1"],
    label="Eupnic Breathing",
)
# axs[0].set_title("Eupnic Breathing", fontsize=16, fontweight="bold")
axs[0].set_xticklabels([])
axs[0].legend(loc="lower right", bbox_to_anchor=(1, 0), borderaxespad=1)

# Subplot 1
axs[1].plot(time, tsx_sniffing_sample, color=colors_.colorplot["6"], label="Sniffing")
# axs[1].set_title("Sniffing", fontsize=16, fontweight="bold")
axs[1].set_xticklabels([])
axs[1].legend(loc="lower right", bbox_to_anchor=(1, 0), borderaxespad=1)
# Subplot 2
axs[2].plot(
    time, tsx_sigh_sample, color=colors_.colorplot["3"], alpha=0.95, label="Sigh"
)
# axs[2].set_title("Sigh", fontsize=16, fontweight="bold")
axs[2].legend(loc="lower right", bbox_to_anchor=(1, 0), borderaxespad=1)

for ax in axs:
    ax.set_ylabel("A.Units", fontsize=14)

axs[2].set_xlabel("Arbitrary Time (s)", fontsize=14)

plt.tight_layout()  # Adjust the layout
plt.show()
# %%
fig.savefig("figs/breathing_behaviors.png", dpi=600)

# %%
