"""
After the classifiers are trained, add the weights to this file and create histograms to see the distribution of the weights.
Run this script from the same directory as the current script. (not from the root of the project, unlike the other scripts)
"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import seaborn as sns

from utils.helper_functions import create_dir


class Meta:
    def __init__(self, name, weights):
        self.name = name
        self.weights = weights


def load_weights_from_csv(filename):
    weights = []
    with open(filename, 'r') as f:
        next(f)  # Skip the header
        for line in f:
            weights.append([float(x.strip('"')) for x in line.strip().split(",")])
    return weights


def histogram_from_weights(title, metas: List[Meta]):
    dir = 'graphs'
    create_dir(dir)
    for meta in metas:
        weights = meta.weights
        name = meta.name

        max = np.max([abs(float(x)) for x in weights])
        data = [float(x) / max for x in weights]

        sns.kdeplot(data, bw_adjust=.5, label=name)

    title = f"{title} plot of 512 dimensions"
    plt.title(title)

    plt.legend(loc='upper right')

    plt.xlim(-1, 1)
    plt.xlabel("Weight (normalized between -1 and 1)")
    plt.ylabel("Density")

    tikzplotlib.save(f"{dir}/{title}_kde_plot_512_dims.tex")
    plt.savefig(f"{dir}/{title}_kde_plot_512_dims.pdf")
    plt.show()


def flatten(l):
    """Flatten a list of lists.
    list of lists (x[0] = [a, b, c], x[1] = [d, e, f] ...)
    --> to a single list (x = [a, b, c, d, e, f, ...])
    """

    return [item for sublist in l for item in sublist]


if __name__ == '__main__':
    # histograms_512_dims_continuous()

    #  module 0
    s = flatten(load_weights_from_csv("modul0/wandb_export sim modul=0 layer=-1.csv"))
    g = flatten(load_weights_from_csv("modul0/wandb_export gim modul=0 layer=-1.csv"))
    c = flatten(load_weights_from_csv("modul0/wandb_export cpc modul=0 layer=2 v2.csv"))
    histogram_from_weights("Module 0", [Meta("GIM", g),
                                        Meta("SIM", s),
                                        Meta("CPC", c)
                                        ])

    # module 1
    s = flatten(load_weights_from_csv("modul1/wandb_export sim modul=1 layer=-1.csv"))
    g = flatten(load_weights_from_csv("modul1/wandb_export gim modul=1 layer=-1.csv"))
    c = flatten(load_weights_from_csv("modul1/wandb_export cpc modul=0 layer=5 v2.csv"))
    histogram_from_weights("Module 1", [Meta("GIM", g),
                                        Meta("SIM", s),
                                        Meta("CPC", c)
                                        ])

    # module 2
    s = flatten(load_weights_from_csv("modul2/wandb_export sim modul=2 layer=-1.csv"))
    g = flatten(load_weights_from_csv("modul2/wandb_export gim modul=2 layer=-1.csv"))
    c = flatten(load_weights_from_csv("modul2/wandb_export cpc modul=0 layer=7 v2.csv"))
    histogram_from_weights("module 2",
                           [Meta("GIM", g),
                            Meta("SIM", s),
                            Meta("CPC", c)
                            ])
