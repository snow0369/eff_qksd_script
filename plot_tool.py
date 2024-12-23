from typing import List, Tuple, Optional

import numpy as np
from matplotlib import pyplot as plt


def default_color_table(len_color):
    return plt.cm.jet(np.linspace(0, 1, len_color))


def plot_histogram(ax, prob_list: List[np.ndarray], label_list: List[str],
                   n_bins=100, bound_x: Optional[Tuple[float, float]] = None,
                   v_bars:Optional[List[float]]=None,
                   v_labels:Optional[List[str]]=None,
                   color_list=None, max_alpha=1.0):
    if bound_x is None:
        min_x = min([np.min(p) for p in prob_list])
        max_x = max([np.max(p) for p in prob_list])
    else:
        min_x, max_x = bound_x

    bins = np.linspace(min_x, max_x, n_bins)
    delta_bins = bins[1] - bins[0]
    if color_list is None:
        color_list = default_color_table(len(prob_list))
    else:
        color_list = color_list(np.linspace(0, 1, len(prob_list)))

    for c, prob, label in zip(color_list, prob_list, label_list):
        ax.hist(prob, alpha=max_alpha / len(prob_list), bins=bins, weights=np.ones_like(prob) / len(prob),
                color=c, label=label)
    if v_bars is not None:
        v_color_list = default_color_table(len(v_bars))
        for v, l, c in zip(v_bars, v_labels, v_color_list):
            ax.vlines(v, 0, 1, linewidths=1.0, label=l, color=c)

    ax.set_xlim([min_x - delta_bins, max_x + delta_bins])
