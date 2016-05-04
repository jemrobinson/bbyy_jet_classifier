"""
performance.py
author: Luke de Oliveira (lukedeo@stanford.edu)

Usage:
------
>>> weights = np.ones(n_samples)
>>> # -- going to match bkg to signal
>>> weights[signal == True] = get_weights(sig_pt, bkg_pt)
>>> discs = {}
>>> add_curve(r"\tau_{32}", "red", calculate_roc(signal, tau_32, weights=weights))
>>> fg = ROC_plotter(discs)
>>> fg.savefig("myroc.pdf")

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve


def get_weights(target, actual, bins=10, cap=10, match=True):
    """
    re-weights a actual distribution to a target.

    Args:
        target (array/list): observations drawn from target distribution
        actual (array/list): observations drawn from distribution to
            match to the target.

        bins (numeric or list/array of numerics): bins to use to do weighting

        cap (numeric): maximum weight value.

        match (bool): whether to make the sum of weights in actual equal to the
            number of samples in target

    Returns:
        numpy.array: returns array of shape len(actual).

    """
    target_counts, target_bins = np.histogram(target, bins=bins)
    counts, _ = np.histogram(actual, bins=target_bins)
    counts = (1.0 * counts)
    counts = np.array([max(a, 0.0001) for a in counts])
    multiplier = np.array((target_counts / counts).tolist() + [1.0])
    print type(target_counts), type(target_bins)

    weights = np.array([min(multiplier[target_bins.searchsorted(point) - 1], cap) for point in actual])

    if match:
        weights *= (len(target) / np.sum(weights))

    return weights


def calculate_roc(labels, discriminant, weights=None):
    """
    Definition:
    -----------
        Use the scikit-learn roc_curve function to calculate signal efficiency and background rejection

    Args:
    -----
        labels = an array of 1/0 representing signal/background
        discriminant  = an array that represents the discriminant
        weights = sample weights for each point
                  assert(weights.shape == discriminant.shape)

    Returns:
    --------
        tuple: (signal_efficiency, background_rejection) where each are arrays

    """
    fpr, tpr, _ = roc_curve(labels, discriminant, sample_weight=weights)
    sig_eff = tpr
    bkg_rej = 1.0 / fpr if fpr > 0.0 else 0.0
    return sig_eff, bkg_rej


def ROC_plotter(curves, min_eff=0, max_eff=1, linewidth=1.4, pp=False, signal="",
                background="", title="", logscale=True, ymax=10**4, ymin=1):

    fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
    ax = fig.add_subplot(111)
    plt.xlim(min_eff, max_eff)
    plt.grid(b=True, which="minor")
    plt.grid(b=True, which="major")
    max_ = 0
    for tagger, data in curves.iteritems():
        sel = (data["efficiency"] >= min_eff) & (data["efficiency"] <= max_eff)
        if np.max(data["rejection"][sel]) > max_:
            max_ = np.max(data["rejection"][sel])
        plt.plot(data["efficiency"][sel], data["rejection"][sel], "-", label=r"" + tagger, color=data["color"], linewidth=linewidth)

    ax = plt.subplot(1, 1, 1)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    plt.ylim(ymin, ymax)
    if logscale:
        ax.set_yscale("log")

    ax.set_xlabel(r"$\varepsilon_{\mathrm{signal}}$")
    ax.set_ylabel(r"$1 / \varepsilon_{\mathrm{background}}$")

    plt.legend()
    plt.title(r"" + title)
    if pp:
        plt.savefig(fig)
    else:
        return fig


def add_curve(name, color, curve_pair, dictref):
    dictref.update(
        {
            name: {
                "efficiency": curve_pair[0],
                "rejection": curve_pair[1],
                "color": color
            }
        }
    )
