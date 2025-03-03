"""
Author: jpenalozaa
Description: functions containing plotting configurations 
"""

# Random code for asthetics
import matplotlib as mpl


# Color scheme
#
class ColorPalette:
    facecolor = "#e5ecf6"
    colorplot = dict(
        [
            ["1", "#481652"],
            ["2", "#9E17EC"],
            ["3", "#417796"],
            ["4", "#179DEB"],
            ["5", "#6B2A2A"],
            ["6", "#EB9017"],
            ["7", "#406B64"],
            ["8", "#17EBC6"],
            ["9", "#EBBC17"],
            ["10", "#C6E129"],
        ]
    )


def load_plt_config():
    # General config changes to plot with MatplotLib
    # Set your custom Matplotlib configurations, including spine color
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["axes.facecolor"] = "#e5ecf6"
    mpl.rcParams["axes.spines.bottom"] = True
    mpl.rcParams["axes.spines.top"] = True
    mpl.rcParams["axes.spines.right"] = True
    mpl.rcParams["axes.spines.left"] = True
    mpl.rcParams["font.family"] = "segoe ui"
    mpl.rcParams["font.size"] = 12

    # Set the spine color to white
    mpl.rcParams["axes.edgecolor"] = "white"
    mpl.rcParams["grid.color"] = "white"
