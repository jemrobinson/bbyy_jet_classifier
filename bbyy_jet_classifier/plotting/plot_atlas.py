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


def use_atlas_labels(ax, fontsize=18, lumi=1.04):
    """
    Definition:
    -----------
            Set the plotting style to ATLAS-style and then point this function to
            "None" so that it can only be called once

    Args:
    -----------
            ax = axes on which to draw the ATLAS text
            fontsize = fontsize (default 18)
            lumi = integrated luminosity
    """
    # -- Add ATLAS text
    plt.text(0.02, 0.9, "ATLAS", va="bottom", ha="left", color="black", size=fontsize,
             fontname="sans-serif", weight="bold", style="oblique", transform=ax.transAxes)
    plt.text((fontsize / 100.) - 0.01, 0.9, "Internal", va="bottom", ha="left", color="black", size=fontsize,
             fontname="sans-serif", transform=ax.transAxes)
    plt.text(0.02, 0.83, r"$\sqrt{{s}} = 13 TeV, {} fb^{{-1}}$".format(lumi), va="bottom",
             ha="left", color="black", size=fontsize, fontname="sans-serif", transform=ax.transAxes)
    # -- Force axis labels into correct position
    print ax.xaxis.label
    print ax.xaxis.label.get_text()
    # plt.xlabel(ax.xaxis.label.get_text(), position=(1., 0), va="bottom", ha="right")
    # ax.xaxis.set_label_coords(1., -0.15)
    # plt.ylabel(ax.yaxis.label.get_text(), position=(0., 1.), va="top", ha="right")
    # ax.yaxis.set_label_coords(-0.2, 1.)
    ax.xaxis.label.set_ha("right")
    ax.xaxis.label.set_va("bottom")
    ax.xaxis.label.set_position((1., 0))
    ax.yaxis.label.set_ha("right")
    ax.yaxis.label.set_va("top")
    ax.yaxis.label.set_position((0., 1))
