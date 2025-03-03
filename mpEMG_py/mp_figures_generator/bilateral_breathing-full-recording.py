"""
Author: jpenaloza
Description: pipeline to plot full recording zooming in different behaviors
"""

# %%
# load the libraries required to run this script
# filesystem paths
from pathlib import Path

# Plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.patheffects as pe

from mp_emgdiaphragm.mpstyle import load_plt_config, ColorPalette

# Data loading and manipulation
from neo.io import CedIO
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# units
# import quantities as pq

# filter and envelop functions
from mp_emgdiaphragm.gstats import min_max_normalization, dc_removal
from mp_emgdiaphragm.filters import notch_filter, butter_bandpass_filter
from mp_emgdiaphragm.linear_envelope_methods import (
    get_rms_envelope,
)

# %%
load_plt_config()
colors_ = ColorPalette()
# define home (Drive were data is located)
home = Path.home()
# define the path to the data file
recording_path = (
    r"Documents\data\Methods-Paper_EMG\A6_Baseline_Eupnea_Activity_Eupnea.smrx"
)

# build the path to the data file
recording_working_path = home.joinpath(recording_path)

print(recording_working_path)
# %%
# load EUPNIC data using CedIO
recording_reader = CedIO(filename=recording_working_path)
recording_data = recording_reader.read()
recording_data_block = recording_data[0].segments[0]

# %%
tsx_emg_r = recording_data_block.analogsignals[0]
tsx_emg_l = recording_data_block.analogsignals[1]
fs = float(tsx_emg_r.sampling_rate)  # assuming all sampling rates are the same
# %%
trim_ = int(fs * 60)
tsx_r = tsx_emg_r.magnitude.squeeze()
tsx_notch_r = notch_filter(tsx_r, fs=fs)
tsx_bp_r = dc_removal(
    butter_bandpass_filter(tsx=tsx_notch_r, lowcut=0.1, highcut=1000, fs=fs, order=4),
    method="median",
)[0:trim_]

tsx_l = tsx_emg_l.magnitude.squeeze()
tsx_notch_l = notch_filter(tsx_l, fs=fs)
tsx_bp_l = dc_removal(
    butter_bandpass_filter(tsx=tsx_notch_l, lowcut=0.1, highcut=1000, fs=fs, order=4),
    method="median",
)[0:trim_]

# %%
# plot full recording filtered


time = np.arange(0, tsx_bp_r.__len__()) * (1 / fs)

fig, axs = plt.subplots(2, 1, figsize=(15, 6))

# Subplot 0
axs[0].plot(time, tsx_bp_r, color=colors_.colorplot["5"], label="Right", linewidth=0.5)
axs[0].set_title(
    "Awake bilateral diaphragm EMG recording", fontsize=30, fontweight="bold", pad=20
)
axs[0].set_xticklabels([])
axs[0].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["5"],
            label="Right",
            markerfacecolor=colors_.colorplot["5"],
            markersize=15,
        ),
    ],
    prop={"weight": "bold", "size": 18},
    loc="lower left",
    bbox_to_anchor=(0, -0.2),
    borderaxespad=1,
)


# Subplot 1
axs[1].plot(time, tsx_bp_l, color=colors_.colorplot["6"], label="Left", linewidth=0.5)
axs[1].legend(
    handles=[
        Line2D(
            [0],
            [0],
            marker="o",
            color=colors_.colorplot["6"],
            label="Left",
            markerfacecolor=colors_.colorplot["6"],
            markersize=15,
        ),
    ],
    prop={"weight": "bold", "size": 18},
    loc="lower left",
    bbox_to_anchor=(0, -0.2),
    borderaxespad=1,
)

# Set the same y-axis range for both subplots
ymin = min(np.min(tsx_bp_r), np.min(tsx_bp_r)) * 1.1
ymax = max(np.max(tsx_bp_l), np.max(tsx_bp_l)) * 1.1
for ax in axs:
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("V", fontsize=30, rotation=0, ha="center", va="center", labelpad=25)
    ax.tick_params(axis="y", labelsize=20)


axs[1].set_xlabel("Time (s)", fontsize=30, labelpad=25)
axs[1].tick_params(axis="x", labelsize=20)

# adding a rectangle to define the different behaviors
highlight_times = dict(
    eupnea_base=[10, 20], locomotion=[38, 45], sniffing=[47, 53], eupnea=[54, 59]
)

for behavior, time in highlight_times.items():
    rect = patches.Rectangle(
        (time[0], -2.8 * axs[0].get_ylim()[1]),
        time[1] - time[0],
        11,
        linewidth=2,
        edgecolor="black",
        facecolor="none",
        transform=axs[0].transData,
        alpha=0.7,
    )

    fig.add_artist(rect)


plt.tight_layout()  # Adjust the layout
plt.show()

# %%
fig.savefig("../../figures/composed_full-recording.png", dpi=600)
# %%
# Normalization based on Eupnea period

start_time = 0
end_time = 30
start_sample = int(start_time * fs)
end_sample = int(end_time * fs)
tsx_sample_r = tsx_bp_r[start_sample:end_sample]
tsx_sample_l = tsx_bp_l[start_sample:end_sample]

overlap = 0.99
window_type = "rectangular"
window_size = int(0.033 * fs)  # window size
window_beta = None
rectify_method = "None"

tsx_sample_r_rms = get_rms_envelope(
    tsx_sample_r, overlap, window_type, window_size, window_beta, rectify_method
)  # eupnic breathing moving rms envelope right

tsx_sample_l_rms = get_rms_envelope(
    tsx_sample_l, overlap, window_type, window_size, window_beta, rectify_method
)  # eupnic breathing moving rms envelope right

# normalization min max method
scaler_rms_r = MinMaxScaler()
tsx_sample_r_rms_norm = scaler_rms_r.fit_transform(tsx_sample_r_rms[..., np.newaxis])[
    :, 0
]

scaler_rms_l = MinMaxScaler()
tsx_sample_l_rms_norm = scaler_rms_l.fit_transform(tsx_sample_l_rms[..., np.newaxis])[
    :, 0
]

# %%
# individual plots
# getting the envelope
plot_names = dict(
    eupnea_base="Eupnea",
    locomotion="Locomotion",
    sniffing="Sniffing",
    eupnea="Return to Eupnea",
)

for behavior, time_ in highlight_times.items():
    title_name = plot_names[behavior]
    start_time = time_[0]
    end_time = time_[1]
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)
    tsx_sample_r = tsx_bp_r[start_sample:end_sample]
    tsx_sample_l = tsx_bp_l[start_sample:end_sample]

    overlap = 0.99
    window_type = "rectangular"
    window_size = int(0.033 * fs)  # window size
    window_beta = None
    rectify_method = "None"

    tsx_sample_r_rms = get_rms_envelope(
        tsx_sample_r, overlap, window_type, window_size, window_beta, rectify_method
    )  # eupnic breathing moving rms envelope right

    tsx_sample_l_rms = get_rms_envelope(
        tsx_sample_l, overlap, window_type, window_size, window_beta, rectify_method
    )  # eupnic breathing moving rms envelope right

    # normalization min max method
    tsx_sample_r_rms_norm = scaler_rms_r.transform(tsx_sample_r_rms[..., np.newaxis])[
        :, 0
    ]

    tsx_sample_l_rms_norm = scaler_rms_l.transform(tsx_sample_l_rms[..., np.newaxis])[
        :, 0
    ]

    time = np.linspace(start_time, end_time, tsx_sample_r.__len__())

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    fsize = 30
    markersize = 20
    # Subplot 0
    # Subplot 1
    axs[0].set_title(title_name, fontsize=fsize, fontweight="bold", pad=20)
    # Right side plot with RMS envelop
    axs[0].plot(
        time,
        tsx_sample_r,
        color=colors_.colorplot["5"],
        alpha=0.95,
        linewidth=0.5,
        label="pre-processed",
    )
    # ENVELOPE
    axs[0].plot(
        time,
        tsx_sample_r_rms_norm,
        color="#0070FF",
        label="RMS LE",
        linewidth=0.9,
        path_effects=[pe.Stroke(linewidth=1, foreground="#394555"), pe.Normal()],
    )

    axs[0].fill_between(
        time,
        tsx_sample_r_rms_norm,
        0,
        where=(tsx_sample_r_rms_norm < 1.1),
        color="#0070FF",
        alpha=0.3,
    )

    axs[0].set_xticklabels([])
    axs[0].legend(
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
        bbox_to_anchor=(1, -0.4),
        borderaxespad=1,
    )

    # 6
    # Left side plot with RMS envelope
    axs[1].plot(
        time,
        tsx_sample_l,
        color=colors_.colorplot["6"],
        alpha=0.95,
        linewidth=0.5,
        label=" left hemi-diaphragm",
    )

    axs[1].plot(
        time,
        tsx_sample_l_rms_norm,
        color=colors_.colorplot["7"],
        label="RMS LE",
        linewidth=0.9,
        path_effects=[
            pe.Stroke(linewidth=1, foreground=colors_.colorplot["8"]),
            pe.Normal(),
        ],
    )
    axs[1].fill_between(
        time,
        tsx_sample_l_rms_norm,
        0,
        where=(tsx_sample_l_rms_norm <= tsx_sample_l_rms_norm.max()),
        color=colors_.colorplot["8"],
        alpha=0.3,
    )

    axs[1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color=colors_.colorplot["7"],
                label="Norm RMS LE",
                markerfacecolor=colors_.colorplot["8"],
                markersize=markersize,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=colors_.colorplot["6"],
                alpha=0.5,
                label="left hemi-diaphragm",
                markerfacecolor=colors_.colorplot["6"],
                markersize=markersize,
            ),
        ],
        prop={"weight": "bold", "size": 18},
        loc="lower right",
        bbox_to_anchor=(1, -0.4),
        borderaxespad=1,
    )

    # Set the same y-axis range for both subplots
    ymin = -1.5
    ymax = 2.5
    for ax in axs:
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(
            "V", fontsize=30, rotation=0, ha="center", va="center", labelpad=25
        )
        ax.tick_params(axis="y", labelsize=20)

    axs[1].set_xlabel("Time (s)", fontsize=30, labelpad=25)
    axs[1].tick_params(axis="x", labelsize=20)
    fig.tight_layout()

    plt.tight_layout()  # Adjust the layout
    plt.show()
    fig.savefig(f"../../figures/composed_{title_name}-recording.png", dpi=600)
# %%
