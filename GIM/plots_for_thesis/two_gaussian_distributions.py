# %%
"""
This script generates a scatter plot of two Gaussian distributions.
Both distributions have the same covariance matrix but different means, the distributions are two-dimensional.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# seaborn theme
sns.set_theme()

# Searborn color theme with white background without grid
sns.set_style("whitegrid")


# generate two Gaussian distributions
def generate_gaussian(mean, cov, n):
    return np.random.multivariate_normal(mean, cov, n)

# plot the two Gaussian distributions


def plot_gaussian(mean1, mean2, cov, n):
    # generate the two Gaussian distributions
    x1, y1 = generate_gaussian(mean1, cov, n).T
    x2, y2 = generate_gaussian(mean2, cov, n).T

    # plot the two Gaussian distributions
    plt.scatter(x1, y1, c='b', marker='o', label='Gaussian 1', s=5)
    plt.scatter(x2, y2, c='r', marker='o', label='Gaussian 2', s=5)


    # plot a circel to represent standard normal distribution
    circle = plt.Circle((0, 0), 3, color='k', fill=False)
    plt.gcf().gca().add_artist(circle)

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)

    # plot size
    plt.rcParams["figure.figsize"] = (20, 20)

    # plot should have same width and height
    plt.gca().set_aspect('equal', adjustable='box')

    # save plot as highresolution png
    plt.savefig('two_gaussian_distributions.png', dpi=1200)


    plt.show()



# main
if __name__ == "__main__":
    plot_gaussian([-3, -3], [3, 3], [[1, 0], [0, 1]], 400)
