import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import rootpy.plotting as rpp

def set_style():
    """
    Definition:
    -----------
            Set the plotting style to ATLAS-style and then point this function to
            "None" so that it can only be called once

    Args:
    -----------
            None
    """
    logging.getLogger("plot_atlas").info("Setting ATLAS style")
    rpp.set_style("ATLAS", mpl=True)
    mpl.rcParams["figure.figsize"] = (6, 6)
    # Force Helvetica in mathmode
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = "Helvetica"
    mpl.rcParams["mathtext.fontset"] = "custom"
    # mpl.rcParams["mathtext.default"] = "regular"
    mpl.rcParams["mathtext.default"] = "sf"
    mpl.rcParams["mathtext.rm"] = "serif"
    mpl.rcParams["mathtext.tt"] = "sans"
    mpl.rcParams["mathtext.it"] = "sans:italic"
    mpl.rcParams["mathtext.bf"] = "sans:bold"
    set_style.func_code = (lambda: None).func_code


def use_atlas_labels(ax, lumi=40.0):
    """
    Definition:
    -----------
            Draw ATLAS labels

    Args:
    -----------
            ax = axes on which to draw the ATLAS text
            fontsize = fontsize (default 18)
            lumi = integrated luminosity
    """
    # -- Add ATLAS text
    plt.text(0.03, 0.92, "ATLAS", va="bottom", ha="left", color="black", size=18, fontname="sans-serif", weight="bold", style="oblique", transform=ax.transAxes)
    plt.text(0.22, 0.92, "Internal", va="bottom", ha="left", color="black", size=18, fontname="sans-serif", transform=ax.transAxes)
    plt.text(0.03, 0.83, r"$\sqrt{{s}} = 13\ TeV, {} fb^{{-1}}$".format(lumi), va="bottom", ha="left", color="black", size=16, fontname="sans-serif", transform=ax.transAxes)

    # -- Force axis labels into correct position
    ax.xaxis.label.set_ha("right")
    ax.xaxis.label.set_va("bottom")
    ax.xaxis.label.set_position((1., 0))
    ax.xaxis.set_label_coords(1., -0.15)
    ax.yaxis.label.set_ha("right")
    ax.yaxis.label.set_va("top")
    ax.yaxis.label.set_position((0., 1))
    ax.yaxis.set_label_coords(-0.2, 1.)
