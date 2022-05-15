import numpy as np
from sklearn import clone

from skactiveml.base import (
    SingleAnnotatorPoolQueryStrategy,
    ProbabilisticRegressor,
)
from skactiveml.utils import (
    check_type,
    simple_batch,
    MISSING_LABEL,
    check_random_state,
)
from skactiveml.pool.regression.utils._integration import (
    conditional_expect,
    reshape_dist,
)
from skactiveml.pool.regression.utils._model_fitting import (
    update_reg,
)


class MutualInformationGainMaximization(SingleAnnotatorPoolQueryStrategy):
    """Regression based Mutual Information Gain Maximization

    This class implements a mutual information based selection strategies, where
    it is assumed that the prediction probability for different samples
    are independent.

    Parameters
    ----------
    integration_dict: dict,
        Dictionary for integration arguments, i.e. `integration method` etc.,
        used for calculating the expected `y` value for the candidate samples.
        For details see method `conditional_expect`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.

    References
    ----------
    [1] Elreedy, Dina and F Atiya, Amir and I Shaheen, Samir. A novel active
        learning regression framework for balancing the exploration-exploitation
        trade-off, page 651 and subsequently, 2019.

    """

    def __init__(
        self,
        integration_dict=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            random_state=random_state,
            missing_label=missing_label,
        )
        if integration_dict is not None:
            self.integration_dict = integration_dict
        else:
            self.integration_dict = {"method": "assume_linear"}

    def query(
        self,
        X,
        y,
        reg,
        fit_reg=True,
        sample_weight=None,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        reg: ProbabilisticRegressor
            Predicts the entropy and the y-values the candidate samples
            could have.
        fit_reg : bool, optional (default=True)
            Defines whether the classifier should be fitted on `X`, `y`, and
            `sample_weight`.
        sample_weight: array-like of shape (n_samples), optional (default=None)
            Weights of training samples in `X`.
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If candidates is None, the unlabeled samples from (X,y) are
            considered as candidates.
            If candidates is of shape (n_candidates) and of type int,
            candidates is considered as the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the
            candidates are directly given in candidates (not necessarily
            contained in X). This is not supported by all query strategies.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or
            numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        """
        (X, y, candidates, batch_size, return_utilities,) = self._validate_data(
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
            reset=True,
        )

        check_type(reg, "reg", ProbabilisticRegressor)
        check_type(fit_reg, "fit_reg", bool)
        check_type(
            self.integration_dict,
            "self.integration_dict",
            dict,
        )

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        if fit_reg:
            reg = clone(reg).fit(X, y, sample_weight)

        utilities_cand = self._mutual_information(
            X, X_cand, mapping, reg, X, y, sample_weight
        )

        if mapping is None:
            utilities = utilities_cand
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand

        return simple_batch(
            utilities,
            self.random_state_,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )

    def _mutual_information(
        self,
        X_eval,
        X_cand,
        mapping,
        reg,
        X,
        y,
        sample_weight=None,
    ):
        """Calculates the mutual information gain over the evaluation set if each
        candidate where to be labeled.

        Parameters
        ----------
        X_eval : array-like of shape (n_samples, n_features)
            The samples where the information gain should be evaluated.
        X_cand : array-like of shape (n_candidate_samples, n_features)
            The candidate samples that are potentially labeled.
        mapping : array-like of shape (n_samples,) or None
            The potential mapping between the candidate samples and the
            training data set.
        reg: ProbabilisticRegressor
            Estimates the entropy, predicts values.
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        sample_weight: array-like of shape (n_samples), optional (default=None)
            Weights of training samples in `X`.

        Returns
        -------
        query_indices : numpy.ndarray of shape (n_candidate_samples)
            The expected information gain for each candidate sample.
        """

        prior_entropy = np.sum(reg.predict(X_eval, return_entropy=True)[1])

        def new_entropy(idx, x_cand, y_pot):
            reg_new = update_reg(
                reg,
                X,
                y,
                sample_weight=sample_weight,
                y_update=y_pot,
                idx_update=idx,
                X_update=x_cand,
                mapping=mapping,
            )
            _, entropy_cand = reg_new.predict(X_eval, return_entropy=True)
            potentials_post_entropy = np.sum(entropy_cand)
            return potentials_post_entropy

        cond_entropy = conditional_expect(
            X_cand,
            new_entropy,
            reg,
            random_state=self.random_state_,
            include_idx=True,
            include_x=True,
            **self.integration_dict
        )

        mi_gain = prior_entropy - cond_entropy

        return mi_gain


class KLDivergenceMaximization(SingleAnnotatorPoolQueryStrategy):
    """Regression based Kullback Leibler Divergence Maximization

    This class implements a Kullback Leibler divergence based selection
    strategies, where it is assumed that the prediction probability for
    different samples are independent.

    Parameters
    ----------
    integration_dict_target_val: dict, optional (default=None)
        Dictionary for integration arguments, i.e. `integration method` etc.,
        used for calculating the expected `y` value for the candidate samples.
        For details see method `conditional_expect`.
    integration_dict_cross_entropy: dict, optional (default=None)
        Dictionary for integration arguments, i.e. `integration method` etc.,
        used for calculating the cross entropy between the updated conditional
        estimator by the `X_cand` value and the old conditional estimator.
        For details see method `conditional_expect`.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.

    References
    ----------
    [1] Elreedy, Dina and F Atiya, Amir and I Shaheen, Samir. A novel active
        learning regression framework for balancing the exploration-exploitation
        trade-off, page 651 and subsequently, 2019.

    """

    def __init__(
        self,
        integration_dict_target_val=None,
        integration_dict_cross_entropy=None,
        missing_label=MISSING_LABEL,
        random_state=None,
    ):
        super().__init__(
            random_state=random_state,
            missing_label=missing_label,
        )

        if integration_dict_target_val is not None:
            self.integration_dict_target_val = integration_dict_target_val
        else:
            self.integration_dict_target_val = {"method": "assume_linear"}

        if integration_dict_cross_entropy is not None:
            self.integration_dict_cross_entropy = integration_dict_cross_entropy
        else:
            self.integration_dict_cross_entropy = {
                "method": "gauss_hermite",
                "n_integration_samples": 10,
            }

    def query(
        self,
        X,
        y,
        reg,
        fit_reg=True,
        sample_weight=None,
        candidates=None,
        batch_size=1,
        return_utilities=False,
    ):
        """Determines for which candidate samples labels are to be queried.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        reg: ProbabilisticRegressor
            Estimates the entropy and the cross entropy and the potential
            y-values for the candidate samples.
        fit_reg : bool, optional (default=True)
            Defines whether the regressor should be fitted on `X`, `y`, and
            `sample_weight`.
        sample_weight: array-like of shape (n_samples), optional (default=None)
            Weights of training samples in `X`.
        candidates : None or array-like of shape (n_candidates), dtype=int or
            array-like of shape (n_candidates, n_features),
            optional (default=None)
            If candidates is None, the unlabeled samples from (X,y) are
            considered as candidates.
            If candidates is of shape (n_candidates) and of type int,
            candidates is considered as the indices of the samples in (X,y).
            If candidates is of shape (n_candidates, n_features), the
            candidates are directly given in candidates (not necessarily
            contained in X). This is not supported by all query strategies.
        batch_size : int, optional (default=1)
            The number of samples to be selected in one AL cycle.
        return_utilities : bool, optional (default=False)
            If true, also return the utilities based on the query strategy.

        Returns
        -------
        query_indices : numpy.ndarray of shape (batch_size)
            The query_indices indicate for which candidate sample a label is
            to queried, e.g., `query_indices[0]` indicates the first selected
            sample.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        utilities : numpy.ndarray of shape (batch_size, n_samples) or
            numpy.ndarray of shape (batch_size, n_candidates)
            The utilities of samples after each selected sample of the batch,
            e.g., `utilities[0]` indicates the utilities used for selecting
            the first sample (with index `query_indices[0]`) of the batch.
            Utilities for labeled samples will be set to np.nan.
            If candidates is None or of shape (n_candidates), the indexing
            refers to samples in X.
            If candidates is of shape (n_candidates, n_features), the indexing
            refers to samples in candidates.
        """
        (X, y, candidates, batch_size, return_utilities,) = self._validate_data(
            X,
            y,
            candidates,
            batch_size,
            return_utilities,
            reset=True,
        )

        check_type(reg, "reg", ProbabilisticRegressor)
        check_type(fit_reg, "fit_reg", bool)

        X_cand, mapping = self._transform_candidates(candidates, X, y)

        if fit_reg:
            reg = clone(reg).fit(X, y, sample_weight)

        utilities_cand = self._kullback_leibler_divergence(
            X,
            X_cand,
            mapping,
            reg,
            X,
            y,
            sample_weight=sample_weight,
        )

        if mapping is None:
            utilities = utilities_cand
        else:
            utilities = np.full(len(X), np.nan)
            utilities[mapping] = utilities_cand

        return simple_batch(
            utilities,
            self.random_state_,
            batch_size=batch_size,
            return_utilities=return_utilities,
        )

    def _kullback_leibler_divergence(
        self,
        X_eval,
        X_cand,
        mapping,
        reg,
        X,
        y,
        sample_weight=None,
    ):
        """Calculates the mutual information gain over the evaluation set if
        each candidate sample where to be labeled.

        Parameters
        ----------
        X_eval : array-like of shape (n_samples, n_features)
            The samples where the information gain should be evaluated.
        X_cand : array-like of shape (n_candidate_samples, n_features)
            The candidate samples that determine the information gain.
        mapping : array-like of shape (n_candidate_samples,) or None
            A mapping between `X_cand` and `X` if it exists.
        reg: ProbabilisticRegressor
            Estimates the entropy, predicts values.
        X : array-like of shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples.
        y : array-like of shape (n_samples)
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.MISSING_LABEL.
        sample_weight: array-like of shape (n_samples), optional (default=None)
            Weights of training samples in `X`.

        Returns
        -------
        query_indices : numpy.ndarray of shape (n_candidate_samples)
            The expected information gain for each candidate sample.
        """

        def new_kl_divergence(idx, x_cand, y_pot):
            reg_new = update_reg(
                reg,
                X,
                y,
                sample_weight=sample_weight,
                y_update=y_pot,
                idx_update=idx,
                X_update=x_cand,
                mapping=mapping,
            )
            entropy_post = np.sum(
                reg_new.predict(X_eval, return_entropy=True)[1]
            )
            cross_ent = np.sum(
                cross_entropy(
                    X_eval,
                    reg_new,
                    reg,
                    integration_dict=self.integration_dict_cross_entropy,
                    random_state=self.random_state_,
                )
            )
            return cross_ent - entropy_post

        kl_div = conditional_expect(
            X_cand,
            new_kl_divergence,
            reg,
            random_state=self.random_state_,
            include_idx=True,
            include_x=True,
            **self.integration_dict_target_val
        )

        return kl_div


def cross_entropy(
    X_eval,
    true_reg,
    other_reg,
    integration_dict=None,
    random_state=None,
):
    """Calculates the cross entropy.

    Parameters
    ----------
    X_eval : array-like of shape (n_samples, n_features)
        The samples where the cross entropy should be evaluated.
    true_reg: ProbabilisticRegressor
        True distribution of the cross entropy.
    other_reg: ProbabilisticRegressor
        Evaluated distribution of the cross entropy.
    integration_dict: dict, optional default = None
        Dictionary for integration arguments, i.e. `integration method` etc..
        For details see method `conditional_expect`.
    random_state: numeric | np.random.RandomState, optional
        Random state for cross entropy calculation.

    Returns
    -------
    cross_ent : numpy.ndarray of shape (n_samples)
        The cross entropy.
    """

    if integration_dict is None:
        integration_dict = {}

    check_type(integration_dict, "integration_dict", dict)
    check_type(true_reg, "true_reg", ProbabilisticRegressor)
    check_type(other_reg, "other_reg", ProbabilisticRegressor)
    random_state = check_random_state(random_state)

    dist = reshape_dist(
        other_reg.predict_target_distribution(X_eval),
        shape=(len(X_eval), 1),
    )

    cross_ent = -conditional_expect(
        X_eval,
        dist.logpdf,
        reg=true_reg,
        random_state=random_state,
        **integration_dict,
        vector_func="both"
    )

    return cross_ent
