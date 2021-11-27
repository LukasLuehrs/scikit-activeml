from copy import deepcopy

import numpy as np

from skactiveml.base import SingleAnnotPoolBasedQueryStrategy
from skactiveml.utils import simple_batch


class QBC(SingleAnnotPoolBasedQueryStrategy):
    """Greedy Sampling on the feature space

    This class implements query by committee

    Parameters
    ----------
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.
    k_bootstraps: int, optional (default=1)
        The number of members in a committee.
    """

    def __init__(self, random_state=None, k_bootstraps=5):
        super().__init__(random_state=random_state)
        self.k_bootstraps = k_bootstraps

    def query(self, X_cand, reg, X, y, batch_size=1,
              return_utilities=False):
        """Query the next instance to be labeled.

        Parameters
        ----------
        X_cand: array-like, shape (n_candidates, n_features)
            Unlabeled candidate samples.
        X: array-like, shape (n_samples, n_features)
            Complete training data set.
        reg: SkactivemlRegressor
            Regressor to predict the data.
        y: array-like, shape (n_samples)
            Values of the training data set.
        batch_size: int, optional (default=1)
            The number of instances to be selected.
        return_utilities: bool, optional (default=False)
            If True, the utilities are additionally returned.

        Returns
        -------
        query_indices: np.ndarray, shape (batch_size)
            The index of the queried instance.
        utilities: np.ndarray, shape (batch_size, n_candidates)
            The utilities of all instances in X_cand
            (only returned if return_utilities is True).
        """

        rng = np.random.default_rng(self.random_state)
        n_samples = len(X)
        k = min(n_samples, self.k_bootstraps)
        learners = [deepcopy(reg) for _ in range(k)]
        sample_indices = np.arange(n_samples)
        subsets_indices = [rng.choice(sample_indices, size=n_samples*(k-1)//k)
                           for _ in range(k)]

        for learner, subset_indices in zip(learners, subsets_indices):
            X_for_learner = X[subset_indices]
            y_for_learner = y[subset_indices]
            learner.fit(X_for_learner, y_for_learner)

        results = np.array([learner.predict(X_cand) for learner in learners])
        utilities = np.std(results, axis=0)

        return simple_batch(utilities, return_utilities=return_utilities)




