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
from mp_emgdiaphragm.mpstyle import load_plt_config, ColorPalette

# Data loading and manipulation
from neo.io import CedIO
import numpy as np
from scipy.signal import resample

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

# %%


# resample signal
fs_resample = int(fs // 8)
tsx_resample = resample(tsx_rms_norm, fs_resample)

plt.plot(tsx_resample)
# %%
import matplotlib.pyplot as plt
import ruptures as rpt

# detection
algo = rpt.Pelt(model="rbf").fit(tsx_resample)
result = algo.predict(pen=50)

# %%
rpt.display(tsx_resample, result)
plt.show()
# %%
algo = rpt.Dynp(model="l2").fit(tsx_resample)
result = algo.predict(n_bkps=4)

print(result)
# %%
algo_c = rpt.KernelCPD(kernel="rbf", min_size=2).fit(
    tsx_resample
)  # written in C, same class as before
# %%
algo_c
# %%
result = algo_c.predict(pen=200)
# %%
rpt.display(tsx_resample, result)
plt.show()
# %%
