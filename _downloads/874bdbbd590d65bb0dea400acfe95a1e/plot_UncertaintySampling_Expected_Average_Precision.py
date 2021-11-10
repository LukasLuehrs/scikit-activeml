"""
Expected Average Precision
==========================
"""

# %%
# 

import numpy as np
from matplotlib import pyplot as plt, animation
from sklearn.datasets import make_blobs
from skactiveml.utils import MISSING_LABEL, unlabeled_indices, labeled_indices
from skactiveml.visualization import plot_utility, plot_decision_boundary
from skactiveml.pool import UncertaintySampling
from skactiveml.classifier import PWC

random_state = np.random.RandomState(0)

# Build a dataset.
X, y_true = make_blobs(n_samples=100, n_features=2,
                  centers=[[0, 1], [-3, .5], [-1, -1], [2, 1], [1, -.5]],
                  cluster_std=.7, random_state=random_state)
y_true = y_true % 2
y = np.full(shape=y_true.shape, fill_value=MISSING_LABEL)

# Initialise the classifier.
clf = PWC(classes=[0, 1], random_state=random_state)
# Initialise the query strategy.
qs = UncertaintySampling(method='expected_average_precision', random_state=random_state)

# Preparation for plotting.
fig, ax = plt.subplots()
feature_bound = [[min(X[:, 0]), min(X[:, 1])], [max(X[:, 0]), max(X[:, 1])]]
artists = []

import os
if os.getenv('N_CYCLES'):
    print('Looks like N_CYCLES!')
else:
    print('Ohhh... There is no N_CYCLES!')

# The active learning cycle:
n_cycles = 25
for c in range(n_cycles):
    # Fit the classifier.
    clf.fit(X, y)

    # Set X_cand to the unlabeled instances.
    unlbld_idx = unlabeled_indices(y)
    X_cand = X[unlbld_idx]
    X_labeled = X[labeled_indices(y)]

    # Query the next instance/s.
    query_params = {"clf": clf, "X": X, "y": y}
    query_idx = unlbld_idx[qs.query(X_cand, **query_params)]

    # Plot the labeled data.
    coll_old = list(ax.collections)
    title = ax.text(
        0.5, 1.05, f"Decision boundry after acquring {c} labels",
        size=plt.rcParams["axes.titlesize"], ha="center",
        transform=ax.transAxes
    )
    ax = plot_utility(qs, {'clf': clf, 'X': X, 'y': y}, X_cand=X, feature_bound=feature_bound, ax=ax)
    ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap="coolwarm", marker=".", zorder=2)
    ax.scatter(X_labeled[:, 0], X_labeled[:, 1], c="grey", alpha=.8, marker=".", s=300)
    ax = plot_decision_boundary(clf, feature_bound, ax=ax)

    coll_new = list(ax.collections)
    coll_new.append(title)
    artists.append([x for x in coll_new if (x not in coll_old)])

    # Label the queried instances.
    y[query_idx] = y_true[query_idx]

ani = animation.ArtistAnimation(fig, artists, interval=1000, blit=True)

# %%
# .. rubric:: References:
# 
# The implementation of this strategy is based on :footcite:t:`wang2018uncertainty`.
#
# .. footbibliography::


