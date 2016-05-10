import matplotlib as mpl
import matplotlib.pyplot as plt
import rootpy.plotting as rpp


def set_style():
    """Set the plotting style to ATLAS-style and then point this function to
       "None" so that it can only be called once
    """
    rpp.set_style("ATLAS", mpl=True)
    mpl.rcParams['figure.figsize'] = (6, 6)
    set_style.func_code = (lambda: None).func_code


def atlas_label(ax, fontsize=20, lumi=1.04):
    # -- Add ATLAS text
    plt.text(0.05, 0.9, "ATLAS", va="bottom", ha="left", color="black", size=fontsize,
             fontname="sans-serif", weight="bold", style="oblique", transform=ax.transAxes)
    plt.text(0.23, 0.9, "Internal", va="bottom", ha="left", color="black", size=fontsize,
             fontname="sans-serif", transform=ax.transAxes)
    plt.text(0.05, 0.83, r"$\sqrt{s} = 13 TeV :\ \ \int Ldt = " + str(lumi) + "\ fb^{-1}$",  # change number according to actual data used!
             va="bottom", ha="left", color="black", size=fontsize, fontname="sans-serif", transform=ax.transAxes)
