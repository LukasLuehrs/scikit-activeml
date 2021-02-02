import itertools
import warnings

import numpy as np

from scipy.special import factorial, gammaln
from sklearn import clone
from sklearn.metrics import pairwise_kernels
from sklearn.utils import check_array

from skactiveml.base import SingleAnnotPoolBasedQueryStrategy, \
    ClassFrequencyEstimator, SkactivemlClassifier
from skactiveml.classifier import PWC
from skactiveml.utils import check_classifier_params, check_X_y, \
    ExtLabelEncoder, simple_batch, check_random_state
from skactiveml.utils import rand_argmax, MISSING_LABEL, check_cost_matrix, \
    check_scalar, is_labeled


class McPAL(SingleAnnotPoolBasedQueryStrategy):
    """Multi-class Probabilistic Active Learning

    This class implements multi-class probabilistic active learning (McPAL) [1]
    strategy.

    Parameters
    ----------
    clf: BaseEstimator
        Probabilistic classifier for gain calculation.
    prior: float, optional (default=1)
        Prior probabilities for the Dirichlet distribution of the samples.
    m_max: int, optional (default=1)
        Maximum number of hypothetically acquired labels.
    random_state: numeric | np.random.RandomState, optional
        Random state for candidate selection.

    References
    ----------
    [1] Daniel Kottke, Georg Krempl, Dominik Lang, Johannes Teschner, and Myra
    Spiliopoulou.
        Multi-Class Probabilistic Active Learning,
        vol. 285 of Frontiers in Artificial Intelligence and Applications,
        pages 586-594. IOS Press, 2016
    """

    def __init__(self, clf, prior=1, m_max=1, random_state=None):
        super().__init__(random_state=random_state)
        self.clf = clf
        self.prior = prior
        self.m_max = m_max

    def query(self, X_cand, X, y, sample_weight=None, utility_weight=None,
              batch_size=1, return_utilities=False):
        """Query the next instance to be labeled.

        Parameters
        ----------
        X_cand: array-like, shape(n_candidates, n_features)
            Unlabeled candidate samples
        X: array-like (n_training_samples, n_features)
            Complete data set
        y: array-like (n_training_samples)
            Labels of the data set
        sample_weight: array-like, shape (n_training_samples),
                       optional (default=None)
            Weights for uncertain annotators
        batch_size: int, optional (default=1)
            The number of instances to be selected.
        utility_weight: array-like (n_candidate_samples)
            Densities for each instance in X
        return_utilities: bool (default=False)
            If True, the utilities are additionally returned.

        Returns
        -------
        query_indices: np.ndarray, shape (1)
            The index of the queried instance.
        utilities: np.ndarray, shape (1, n_candidates)
            The utilities of all instances in X_cand
            (only returned if return_utilities is True).
        """
        # Validate input
        X_cand, return_utilities, batch_size, utility_weight, random_state = \
            self._validate_data(X_cand, return_utilities, batch_size,
                                utility_weight, self.random_state, reset=True)

        # Calculate utilities and return the output
        clf = clone(self.clf)
        clf.fit(X, y, sample_weight)
        k_vec = clf.predict_freq(X_cand)
        utilities = utility_weight * _cost_reduction(k_vec, prior=self.prior,
                                                     m_max=self.m_max)

        return simple_batch(utilities, random_state,
                            batch_size=batch_size,
                            return_utilities=return_utilities)

    def _validate_data(self, X_cand, return_utilities, batch_size,
                       utility_weight, random_state, reset=True):
        X_cand, return_utilities, batch_size, random_state = \
            super()._validate_data(X_cand, return_utilities, batch_size,
                                   self.random_state, reset=True)
        # Check if the classifier and its arguments are valid
        if not isinstance(self.clf, ClassFrequencyEstimator):
            raise TypeError("'clf' must implement methods according to "
                            "'ClassFrequencyEstimator'.")
        check_classifier_params(self.clf.classes, self.clf.missing_label)

        # Check 'utility_weight'
        if utility_weight is None:
            utility_weight = np.ones(len(X_cand))
        utility_weight = check_array(utility_weight, ensure_2d=False)

        # Check if X_cand and utility_weight have the same length
        if not len(X_cand) == len(utility_weight):
            raise ValueError(
                "'X_cand' and 'utility_weight' must have the same length."
            )

        return X_cand, return_utilities, batch_size, utility_weight, \
            random_state


class XPAL(SingleAnnotPoolBasedQueryStrategy):

    def __init__(self, clf, scoring='error',
                 cost_vector=None, cost_matrix=None, custom_perf_func=None,
                 prior_cand=1.e-3, prior_eval=1.e-3,
                 estimator_metric='rbf', estimator_metric_dict=None,
                 batch_mode='greedy',
                 batch_labels_equal=False,
                 nonmyopic_max_cand=1, nonmyopic_neighbors='nearest',
                 nonmyopic_labels_equal=True,
                 nonmyopic_independent_probs=True,
                 random_state=None):
        """ XPAL
        The expected probabilistic active learning (XPAL)
        strategy is a generalization of the multi-class probabilistic active
        learning (McPAL) [1], the optimised probabilistic active learning
        (OPAL) [2] strategy, and the cost-sensitive probabilistic active
        learning (CsPAL) strategy due to consideration of a cost matrix for
        multi-class classification problems and the computation of the expected
        error on an evaluation set.

        Attributes
        ----------
        clf: SkactivemlClassifier
            Deterministic classifier to be used.
        scoring: string
            "error"
                equivalent to accuracy
            "cost-vector"
                requires cost-vector
            "misclassification-loss"
                requires cost-matrix
            "mean-abs-error"
            "macro-accuracy"
                optional: TODO: prior_class_probability
            "f1-score"
        cost_vector: array-like (n_classes)
            Vector denoting the misclassification cost of each class. Only
            used if scoring='cost-vector'.
        cost_matrix: array-like (n_classes, n_classes)
            Matrix denoting the misclassification cost. The element
            cost_matrix[i,j] denotes the cost of an instance with true class i
            that is predicted as class j.
            Only used if scoring='misclassification-loss'.
        custom_perf_func: callable
            Function measuring the performance of a given confusion matrix.
            Maps an (n_classes, n_classes) confusion matrix to a performance.
            Only used if scoring='custom'.
        prior_cand: float | array-like (n_classes)
            Prior of the probability distribution of the label of the candidate
            instance. If it is a float, it is multiplied with the optimal prior
            vector that is calculated dependent on the cost matrix.
        prior_eval: float | array-like (n_classes)
            Prior of the probability distribution of the label of the
            evaluation instance. If it is a float, it is multiplied with the
            optimal prior vector that is calculated dependent on the cost
            matrix.
        estimator_metric: string | callable
            The metric to use when estimating probabilities of simulated
            labels for candidates, candidate sets or evaluation instances.
            If metric is a string, it must be one of the metrics
            in sklearn.pairwise.PAIRWISE_KERNEL_FUNCTIONS.
            Alternatively, if metric is a callable function, it is called on
            each pair of instances (rows) and the resulting value recorded. The
            callable should take two rows from X as input and return the
            corresponding kernel value as a single number. This means that
            callables from :mod:`sklearn.metrics.pairwise` are not allowed, as
            they operate on matrices, not single samples. Use the string
            identifying the kernel instead.
        estimator_metric_dict: dict
            Dictionary with additional parameters for the probability estimator
            defined by estimator_metric.
        batch_mode: string
            Set the batch mode for the candidate selection. Must be one of the
            following:
            "greedy": Greedily select the candidates for the batch
            "full": Check all combinations of candidates with a given batch
            size
        batch_labels_equal: bool
            If True, all labels of the instances in one (greedy/full) batch are
            simulated to be equal. Otherwise, all label combinations are
            simulated.
        nonmyopic_max_cand: int
            If the value is 1, the method is myopic. If it is greater than 1,
            multiple instances at the same/similar locations are simulated. If
            nonmyopic_look_ahead is too low, the utilities might be zero when
            lots of instances are already labeled. If it is too large, the
            execution time is increased.
        nonmyopic_neighbors: str
            Determine which candidates are used for non-myopic simulation.
            Must be one of the following:
            "same":    Simulate all candidates at the same location
            "nearest": Consider the instances that are nearest to the selected
                       candidate
        nonmyopic_labels_equal: bool
            If True, all labels of the instances in the non-myopic look-ahead
            set are simulated to be equal. Otherwise, all label combinations
            are simulated.
        nonmyopic_independent_probs: bool
            If True, the label probabilities within one nonmyopic set are
            assumed to be independent to reduce computational cost.
        random_state : numeric | np.random.RandomState
            Random state to use.

        References
        ----------
        [1] TODO: xpal arxiv, diss
        """
        super().__init__(random_state=random_state)

        self.clf = clf
        self.scoring = scoring
        self.cost_vector = cost_vector
        self.cost_matrix = cost_matrix
        self.custom_perf_func = custom_perf_func
        self.prior_cand = prior_cand
        self.prior_eval = prior_eval
        self.estimator_metric = estimator_metric
        self.estimator_metric_dict = estimator_metric_dict
        self.batch_mode = batch_mode
        self.batch_labels_equal = batch_labels_equal
        self.nonmyopic_max_cand = nonmyopic_max_cand
        self.nonmyopic_neighbors = nonmyopic_neighbors
        self.nonmyopic_labels_equal = nonmyopic_labels_equal
        self.nonmyopic_independent_probs = nonmyopic_independent_probs

    def _validate_input(self, X_cand, X, y, X_eval, batch_size, sample_weight,
                        return_utilities):

        # init parameters
        # TODO: implement and adapt, what happens if X_eval=self
        # random_state = check_random_state(self.random_state, len(X_cand),
        #                                   len(X_eval))
        self._random_state = check_random_state(self.random_state)

        check_classifier_params(self._classes, self._missing_label,
                                self.cost_matrix)
        check_scalar(self.prior_cand, target_type=float, name='prior_cand',
                     min_val=0, min_inclusive=False)
        check_scalar(self.prior_eval, target_type=float, name='prior_eval',
                     min_val=0, min_inclusive=False)

        self._batch_size = batch_size
        check_scalar(self._batch_size, target_type=int, name='batch_size',
                     min_val=1)
        if len(X_cand) < self._batch_size:
            warnings.warn(
                "'batch_size={}' is larger than number of candidate samples "
                "in 'X_cand'. Instead, 'batch_size={}' was set ".format(
                    self._batch_size, len(X_cand)))
            self._batch_size = len(X_cand)

        # TODO: check if X_cand, X, X_eval have similar num_features
        # TODO: check if X, y match
        # TODO: check if weight match with X_cand
        if sample_weight is None:
            self.sample_weight_ = np.ones(len(X))
        else:
            self.sample_weight_ = sample_weight
        # TODO: check if return_utilities is bool

    def query(self, X_cand, X, y, X_eval=None, batch_size=1,
              sample_weight_cand=None, sample_weight=None,
              sample_weight_eval=None, return_utilities=False, **kwargs):
        """

        Attributes
        ----------
        X: array-like (n_training_samples, n_features)
            Labeled samples
        y: array-like (n_training_samples)
            Labels of labeled samples
        X_cand: array-like (n_samples, n_features)
            Unlabeled candidate samples
        X_eval: array-like (n_samples, n_features) or string
            Unlabeled evaluation samples
        """

        ### TEST query parameters
        X_cand, return_utilities, batch_size, random_state = \
            self._validate_data(X_cand, return_utilities, batch_size,
                                self.random_state, reset=True)

        # TODO X, y will be tested by clf when is fitted; X_cand, X equal num features
        # CHECK X_eval will be tested by clf when predicted

        # TODO sample_weight, sample_weight_cand, sample_weight_eval should have correct size
        # TODO what about classifiers that do not support sample_weight?
        if sample_weight_cand is None:
            sample_weight_cand = np.ones(len(X_cand))
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        if sample_weight_eval is None:
            if X_eval is None:
                sample_weight_eval = np.ones(len(X_cand))
            else:
                sample_weight_eval = np.ones(len(X_eval))

        ### TEST init parameters
        # CHECK self.clf
        if not isinstance(self.clf, SkactivemlClassifier):
            raise TypeError('clf has to be from type SkactivemlClassifier. The '
                            'given type is {}. Use the wrapper in '
                            'skactiveml.classifier to use a sklearn '
                            'classifier.'.format(type(self.clf)))

        missing_label = self.clf.missing_label
        label_encoder = ExtLabelEncoder(missing_label=missing_label,
                                        classes=self.clf.classes).fit(y)
        classes = label_encoder.classes_
        n_classes = len(classes)

        # TODO self.metric, self.cost_vector, self.cost_matrix, self.custom_perf_func will be checked in _transform_metric
        # TODO: maybe return dperf function instead of 3 variables
        scoring, cost_matrix, perf_func = \
            _transform_scoring(self.scoring, self.cost_vector, self.cost_matrix,
                               self.custom_perf_func, n_classes=n_classes)
        scoring_decomposable = (scoring == 'misclassification-loss')

        # TODO self.prior_cand, self.prior_eval, if prior is vector, use the vector instead
        opt_prior = calculate_optimal_prior(cost_matrix)
        self._prior_cand = self.prior_cand * opt_prior
        self._prior_eval = self.prior_eval * opt_prior

        #self.estimator_metric, self.estimator_metric_dict
        if self.estimator_metric_dict is None:
            estimator_metric_dict = {}
        else:
            estimator_metric_dict = self.estimator_metric_dict

        if self.estimator_metric == 'rbf' and \
                estimator_metric_dict is None:
            # TODO: include std, citations, check gamma-bandwidth transformation
            bandwidth = estimate_bandwidth(X.shape[0], X.shape[1])
            estimator_metric_dict = {'gamma': 1/bandwidth}

        # TODO self.batch_mode
        # TODO self.pre_sim_labels_equal
        # TODO self.cand_sim_labels_equal
        # TODO self.nonmyopic_look_ahead
        # TODO self.nonmyopic_neighbors
        # TODO self.nonmyopic_independence


        if self.batch_mode == 'full' and self.nonmyopic_max_cand > 1:
            raise NotImplementedError("batch_mode = 'full' can only be "
                                      "combined with nonmyopic_max_cand = 1")

        """
        CODE
        """
        # MERGING INSTANCES FOR FASTER PROCESSING
        X_ = np.concatenate([X_cand, X], axis=0)
        y_ = np.concatenate([np.full(len(X_cand), missing_label), y], axis=0)
        sample_weight_ = np.concatenate([sample_weight_cand, sample_weight])
        if X_eval is None:
            sample_weight_eval_ = np.concatenate([sample_weight_eval,
                                                  np.full(len(X), 0)])

        idx_X_cand = list(range(len(X_cand)))
        idx_X = list(range(len(X_cand), len(X_cand)+len(X)))
        idx_X_lbld = np.where(is_labeled(y_, missing_label))[0]

        K = lambda X1, X2: pairwise_kernels(X1, X2,
                                            metric=self.estimator_metric,
                                            **estimator_metric_dict)


        # CALCULATING PRE-COMPUTED KERNELS FOR PROB ESTIMATION
        calc_cand_cand_sim = (self.nonmyopic_max_cand > 1 and
                              self.nonmyopic_neighbors == 'nearest') \
                             or batch_size > 1
        sim_cand = _calc_sim(K, X_cand, X_cand, X_,
                             idx_Y1=None if calc_cand_cand_sim else [],
                             idx_Y2=idx_X_lbld)

        if X_eval is not None:
            sim_eval = _calc_sim(K, X_eval, X_cand, X_, idx_Y2=idx_X_lbld)
        else:
            sim_eval = np.concatenate([
                _calc_sim(K, X_cand, X_cand, X_, idx_Y2=idx_X_lbld),
                _calc_sim(K, X_, X_cand, X_, idx_Y1=[], idx_Y2=[])], axis=0)

        # INITIALIZE PROB ESTIMATION
        cand_prob_est = PWC(metric="precomputed", classes=classes,
                            missing_label=missing_label,
                            class_prior=self._prior_cand,
                            random_state=random_state)
        eval_prob_est = PWC(metric="precomputed", classes=classes,
                            missing_label=missing_label,
                            class_prior=self._prior_eval,
                            random_state=random_state)

        # CODE
        utilities = np.full([batch_size, len(X_cand)], np.nan, dtype=float)
        best_indices = np.empty([batch_size], int)

        n_iter = 1 if self.batch_mode == 'full' else batch_size

        for i_iter in range(n_iter):
            idx_preselected = best_indices[:i_iter]
            idx_remaining = np.setdiff1d(idx_X_cand, idx_preselected)

            if self.batch_mode == 'full':
                idx_candlist_set = \
                    list(itertools.combinations(idx_remaining, batch_size))
            elif self.batch_mode == 'greedy':
                idx_candlist_set = _get_nonmyopic_cand_set(
                    neighbors=self.nonmyopic_neighbors,
                    cand_idx=idx_remaining,
                    sim_cand=sim_cand[:, idx_X_cand],
                    M=self.nonmyopic_max_cand)
            else:
                raise ValueError("batch_mode must either be 'greedy' or 'full'")

            iter_utilities = probabilistic_gain(
                clf=self.clf,
                X=X_,
                y=y_,
                X_eval=X_eval,
                sample_weight=sample_weight_,
                sample_weight_eval=sample_weight_eval,
                idx_candlist_set=idx_candlist_set,
                idx_preselected=list(idx_preselected),
                idx_train=idx_X,
                cand_prob_est=cand_prob_est,
                sim_cand=sim_cand,
                eval_prob_est=eval_prob_est,
                sim_eval=sim_eval,
                scoring_decomposable=scoring_decomposable,
                scoring_cost_matrix=cost_matrix,
                scoring_perf_func=perf_func,
                batch_labels_equal=self.batch_labels_equal,
                nonmyopic_independent_probs=self.nonmyopic_independent_probs,
                nonmyopic_labels_equal=self.nonmyopic_labels_equal
            )

            if self.batch_mode == 'greedy':
                # calculate mean gain for non-myopic gains
                M = self.nonmyopic_max_cand
                iter_utilities = iter_utilities.reshape(-1, M)
                iter_utilities /= np.arange(1, M + 1).reshape(-1, M)
                iter_utilities = np.nanmax(iter_utilities, axis=1)

                utilities[i_iter, idx_remaining] = iter_utilities

                cur_best_idx = rand_argmax(iter_utilities,
                                           random_state=random_state)
                best_indices[i_iter] = idx_remaining[cur_best_idx]
            elif self.batch_mode == 'full':
                # TODO: get utilities for each individual instance by maximizing all
                #  utilities where the instance is contined
                cur_best_idx = rand_argmax([iter_utilities], axis=1,
                                           random_state=random_state)
                best_indices[:] = idx_candlist_set[cur_best_idx[0]]


        if return_utilities:
            return best_indices, utilities
        else:
            return best_indices


def probabilistic_gain(clf, X, y, X_eval,
                       sample_weight, sample_weight_eval,
                       idx_candlist_set, idx_preselected, idx_train,
                       cand_prob_est, sim_cand,
                       eval_prob_est, sim_eval,
                       scoring_decomposable,
                       scoring_cost_matrix, scoring_perf_func,
                       batch_labels_equal=False,
                       nonmyopic_independent_probs=False,
                       nonmyopic_labels_equal=True):


    # TODO: check sim shapes and values for training data, candidates etc
    # np.set_printoptions(precision=1, suppress=True)
    # print(sim_cand.T)
    # print(sim_eval.T)

    idx_cand_unique = \
        np.unique(list(itertools.chain(*idx_candlist_set))).tolist()
    idx_train = list(idx_train)
    idx_preselected = list(idx_preselected)

    # TODO check if estimator have same label encoder as clf
    y = np.array(y)
    cand_prob_est, _ = to_int_labels(cand_prob_est, X[idx_train], y[idx_train])
    eval_prob_est, _ = to_int_labels(eval_prob_est, X[idx_train], y[idx_train])
    clf, y = to_int_labels(clf, X, y)

    clf.fit(X[idx_train], y[idx_train], sample_weight[idx_train])
    if X_eval is None:
        pred_old_cand = np.full_like(y, clf.missing_label)
        pred_old_cand[idx_cand_unique] = clf.predict(X[idx_cand_unique])
    else:
        pred_old_cand = clf.predict(X_eval)

    label_encoder = ExtLabelEncoder(missing_label=clf.missing_label,
                                    classes=clf.classes).fit(y)
    classes = label_encoder.transform(label_encoder.classes_)



    prob_cand_X = np.full([len(X), len(classes)], np.nan, float)
    cand_prob_est.fit(X[idx_train], y[idx_train], sample_weight[idx_train])
    prob_cand_X[idx_cand_unique + idx_preselected] = \
        cand_prob_est.predict_proba(sim_cand[idx_cand_unique + idx_preselected, :][:, idx_train])

    # TODO cand_idx_set filtern bei indpendent (reihenfolge ignorieren)
    # TODO bei independent funktioniert gleichheit nicht (exp gezeigt)
    size_set = len(set([tuple(sorted(x)) for x in idx_candlist_set]))
    print('Einsparpotenzial',
          len(idx_candlist_set) - size_set, len(idx_candlist_set))

    if nonmyopic_independent_probs:
        sm_cand_idx_set = []
        map_cand_to_sm = np.zeros(len(idx_candlist_set), int)
        for c_idx, c in enumerate(idx_candlist_set):
            c = sorted(c)
            try:
                sm_idx = sm_cand_idx_set.index(c)
                map_cand_to_sm[c_idx] = sm_idx
            except:
                map_cand_to_sm[c_idx] = len(sm_cand_idx_set)
                sm_cand_idx_set.append(c)
    else:
        sm_cand_idx_set = idx_candlist_set
        map_cand_to_sm = np.arange(len(idx_candlist_set))

    # include pre_sel_cand
    length_cand_idx_set = np.asarray([len(c) for c in sm_cand_idx_set], int)
    y_sim_lists_pre = _get_y_sim_list(classes=classes,
                                      n_instances=len(idx_preselected),
                                      labels_equal=batch_labels_equal)
    y_sim_lists_cand = [_get_y_sim_list(classes=classes, n_instances=n,
                                        labels_equal=nonmyopic_labels_equal)
                        for n in range(np.max(length_cand_idx_set)+1)]
    y_sim_lists = [list(itertools.product(y_sim_lists_pre, y_sim_list_cand))
                   for y_sim_list_cand in y_sim_lists_cand]

    utilities = np.full(len(idx_candlist_set), np.nan, float)
    sm_utilities = np.full(len(sm_cand_idx_set), np.nan, float)

    for i_cand_idx, cand_idx in enumerate(sm_cand_idx_set):
        # TODO include pre_sel_cand_idx
        cand_idx = list(cand_idx)

        idx_new = idx_preselected + cand_idx + idx_train
        X_new = X[idx_new]
        sim_eval_new = sim_eval[:, idx_new]
        sample_weight_new = sample_weight[idx_new]

        y_sim_list = y_sim_lists[len(cand_idx)]

        # TODO Label encoder for y_sim_list
        # TODO not only for non-myopic?
        if nonmyopic_independent_probs:
            prob_y_sim = \
                np.prod(prob_cand_X[idx_preselected + cand_idx,
                                    [a+b for a,b in y_sim_list]], axis=1)
        else:
            prob_y_sim = _dependent_cand_prob(cand_idx, idx_preselected,
                                              idx_train,
                                              y_sim_list,
                                              X, y, sample_weight, prob_cand_X,
                                              cand_prob_est, sim_cand,
                                              pre_independence=nonmyopic_independent_probs)


        sm_utilities[i_cand_idx] = 0
        for i_y_sim, (y_sim_pre, y_sim_cand) in enumerate(y_sim_list):
            # TODO pre_sel_cand
            y_new = np.concatenate([y_sim_pre, y_sim_cand, y[idx_train]], axis=0)

            eval_prob_est.fit(X_new, y_new, sample_weight_new)

            new_clf = clf.fit(X_new, y_new, sample_weight_new)
            if X_eval is None:
                # TODO: sim_eval_new must correspond to indices in X
                prob_eval = eval_prob_est.predict_proba(sim_eval_new[cand_idx])
                pred_new = new_clf.predict(X[cand_idx])
                pred_old = pred_old_cand[cand_idx]
                sample_weight_eval_new = sample_weight_eval[cand_idx]
                # TODO: check density weights
            else:
                prob_eval = eval_prob_est.predict_proba(sim_eval_new)
                pred_new = new_clf.predict(X_eval)
                pred_old = pred_old_cand
                sample_weight_eval_new = sample_weight_eval

            sm_utilities[i_cand_idx] += \
                _dperf(prob_eval, pred_old, pred_new,
                       sample_weight_eval=sample_weight_eval_new,
                       decomposable=scoring_decomposable,
                       cost_matrix=scoring_cost_matrix,
                       perf_func=scoring_perf_func) \
                * prob_y_sim[i_y_sim]



    utilities = sm_utilities[map_cand_to_sm]
    return utilities


def _calc_sim(K, X, Y1, Y2, idx_Y1=None, idx_Y2=None, default=np.nan):
    if idx_Y1 is None:
        sim_1 = K(X, Y1)
    else:
        sim_1 = np.full([len(X), len(Y1)], default, float)
        if len(idx_Y1) > 0:
            sim_1[:, idx_Y1] = K(X, Y1[idx_Y1])


    if idx_Y1 is None:
        sim_2 = K(X, Y2)
    else:
        sim_2 = np.full([len(X), len(Y2)], default, float)
        if len(idx_Y2) > 0:
            sim_2[:, idx_Y2] = K(X, Y2[idx_Y2])

    return np.concatenate([sim_1, sim_2], axis=1)



def _get_nonmyopic_cand_set(neighbors, cand_idx, sim_cand, M):
    if neighbors == 'same':
        cand_idx_set = np.tile(cand_idx, [M, 1]).T.tolist()
    elif neighbors == 'nearest':
        # TODO check correctness
        cand_idx_set = (-sim_cand[cand_idx][:,cand_idx]).argsort(axis=1)[:,:M].tolist()
    else:
        raise ValueError('neighbor_mode unknown')
    res = []
    for ca in cand_idx_set:
        for i in range(len(ca)):
            res.append(ca[:i+1])
    return res

def to_int_labels(est, X, y):
    est = clone(est)
    est.fit(X, y)
    label_encoder = ExtLabelEncoder(missing_label=est.missing_label,
                                    classes=est.classes).fit(y)
    classes = label_encoder.transform(label_encoder.classes_)
    y = label_encoder.transform(y)
    y[np.isnan(y)] = -1
    y = y.astype(int)
    est.missing_label = -1
    est.classes = classes.astype(int)
    return est, y


def _dependent_cand_prob(cand_idx, pre_sel_cand_idx, train_idx, y_sim_list,
                         X, y, sample_weight,prob_cand_X,
                         prob_est, sim_cand, pre_independence):
    # TODO: label encoder

    prob_y_sim = np.ones(len(y_sim_list))
    for i_y_sim, (pre_y_sim, cand_y_sim) in enumerate(y_sim_list):
        if pre_independence:
            prob_y_sim[i_y_sim] = np.prod(prob_cand_X[pre_sel_cand_idx, pre_y_sim])
            idx_pre = pre_sel_cand_idx
            y_pre = pre_y_sim
            idx_sim = cand_idx
            y_sim = cand_y_sim
        else:
            idx_pre = []
            y_pre = []
            idx_sim = pre_sel_cand_idx + cand_idx
            y_sim = pre_y_sim + cand_y_sim

        for i_y in range(len(y_sim)):
            idx_new = idx_pre + idx_sim[:i_y] + train_idx
            X_new = X[idx_new]
            y_new = np.concatenate([y_pre, y_sim[:i_y], y[train_idx]], axis=0)
            sample_weight_new = sample_weight[idx_new]
            prob_est.fit(X_new, y_new, sample_weight_new)

            sim_new = \
                sim_cand[idx_sim[i_y:i_y + 1],:][:, idx_new]
            prob_cand_y = prob_est.predict_proba(sim_new)
            prob_y_sim[i_y_sim] *= prob_cand_y[0][y_sim[i_y]]
    return prob_y_sim

def _get_y_sim_list(classes, n_instances, labels_equal=True):
    if n_instances == 0:
        return [[]]
    else:
        classes = np.asarray(classes, int)
        if labels_equal:
            return (np.tile(np.asarray(classes), [n_instances, 1]).T).tolist()
        else:
            return [list(x) for x in itertools.product(classes, repeat=n_instances)]


def _transform_scoring(metric, cost_matrix, cost_vector, perf_func, n_classes):
    # TODO warning if matrix/vector is given but not used?
    if metric == 'error':
        metric = 'misclassification-loss'
        cost_matrix = 1 - np.eye(n_classes)
        perf_func = None
    elif metric == 'cost-vector':
        if cost_vector is None or cost_vector.shape != (n_classes):
            raise ValueError("For metric='cost-vector', the argument "
                             "'cost_vector' must be given when initialized and "
                             "must have shape (n_classes)")
        metric = 'misclassification-loss'
        cost_matrix = cost_vector.reshape(-1, 1) \
                      @ np.ones([1, n_classes])
        np.fill_diagonal(cost_matrix, 0)
        perf_func = None
    elif metric == 'misclassification-loss':
        if cost_matrix is None:
            raise ValueError("'cost_matrix' cannot be None for "
                             "metric='misclasification-loss'")
        check_cost_matrix(cost_matrix, n_classes)
        metric = 'misclassification-loss'
        cost_matrix = cost_matrix
        perf_func = None
    elif metric == 'mean-abs-error':
        metric = 'misclassification-loss'
        row_matrix = np.arange(n_classes).reshape(-1, 1) \
                     @ np.ones([1, n_classes])
        cost_matrix = abs(row_matrix - row_matrix.T)
        perf_func = None
    elif metric == 'macro-accuracy':
        metric = 'custom'
        perf_func = macro_accuracy_func
        cost_matrix = None
    elif metric == 'f1-score':
        metric = 'custom'
        perf_func = f1_score_func
        cost_matrix = None
    elif metric == 'cohens-kappa':
        # TODO: implement
        metric = 'custom'
        perf_func = perf_func
        cost_matrix = None
    else:
        raise ValueError("Metric '{}' not implemented. Use "
                         "metric='custom' instead.".format(metric))
    return metric, cost_matrix, perf_func

def _dperf(probs, pred_old, pred_new, sample_weight_eval,
           decomposable, cost_matrix=None, perf_func=None):
    if decomposable:
        # TODO: check if cost_matrix is correct
        pred_changed = (pred_new != pred_old)
        return np.sum(sample_weight_eval[pred_changed, np.newaxis] *
                      probs[pred_changed, :] *
                      (cost_matrix.T[pred_old[pred_changed]] -
                       cost_matrix.T[pred_new[pred_changed]])) / \
               len(probs)
    else:
        # TODO: check if perf_func is correct
        n_classes = probs.shape[1]
        conf_mat_old = np.zeros([n_classes, n_classes])
        conf_mat_new = np.zeros([n_classes, n_classes])
        probs = probs * sample_weight_eval[:, np.newaxis]
        for y_pred in range(n_classes):
            conf_mat_old[:, y_pred] += np.sum(probs[pred_old == y_pred], axis=0)
            conf_mat_new[:, y_pred] += np.sum(probs[pred_new == y_pred], axis=0)
        return perf_func(conf_mat_new) - perf_func(conf_mat_old)

def estimate_bandwidth(n_samples, n_features):
    nominator = 2 * n_samples * n_features
    denominator = (n_samples - 1) * np.log((n_samples - 1) / ((np.sqrt(2) * 10 ** -6) ** 2))
    bandwidth = np.sqrt(nominator / denominator)
    return bandwidth

def score_recall(conf_matrix):
    return conf_matrix[-1, -1] / conf_matrix[-1, :].sum()


def macro_accuracy_func(conf_matrix):
    return np.mean(conf_matrix.diagonal() / conf_matrix.sum(axis=1))


def score_accuracy(conf_matrix):
    return conf_matrix.diagonal().sum() / conf_matrix.sum()


def score_precision(conf_matrix):
    pos_pred = conf_matrix[:, -1].sum()
    return conf_matrix[-1, -1] / pos_pred if pos_pred > 0 else 0


def f1_score_func(conf_matrix):
    recall = score_recall(conf_matrix)
    precision = score_precision(conf_matrix)
    norm = recall + precision
    return 2 * recall * precision / norm if norm > 0 else 0


def calculate_optimal_prior(cost_matrix=None):
    n_classes = len(cost_matrix)
    if cost_matrix is None:
        return np.full([n_classes], 1. / n_classes)
    else:
        M = np.ones([1, len(cost_matrix)]) @ np.linalg.inv(cost_matrix)
        if np.all(M[0] >= 0):
            return M[0] / np.sum(M)
        else:
            return np.full([n_classes], 1. / n_classes)


def _cost_reduction(k_vec_list, C=None, m_max=2, prior=1.e-3):
    """Calculate the expected cost reduction.

    Calculate the expected cost reduction for given maximum number of
    hypothetically acquired labels, observed labels and cost matrix.

    Parameters
    ----------
    k_vec_list: array-like, shape (n_samples, n_classes)
        Observed class labels.
    C: array-like, shape = (n_classes, n_classes)
        Cost matrix.
    m_max: int
        Maximal number of hypothetically acquired labels.
    prior : float | array-like, shape (n_classes)
       Prior value for each class.

    Returns
    -------
    expected_cost_reduction: array-like, shape (n_samples)
        Expected cost reduction for given parameters.
    """
    # Check if 'prior' is valid
    check_scalar(prior, 'prior', (float, int),
                 min_inclusive=False, min_val=0)

    # Check if 'm_max' is valid
    check_scalar(m_max, 'm_max', int, min_val=1)

    n_classes = len(k_vec_list[0])
    n_samples = len(k_vec_list)

    # check cost matrix
    C = 1 - np.eye(n_classes) if C is None else np.asarray(C)

    # generate labelling vectors for all possible m values
    l_vec_list = np.vstack([_gen_l_vec_list(m, n_classes)
                            for m in range(m_max + 1)])
    m_list = np.sum(l_vec_list, axis=1)
    n_l_vecs = len(l_vec_list)

    # compute optimal cost-sensitive decision for all combination of k-vectors
    # and l-vectors
    tile = np.tile(k_vec_list, (n_l_vecs, 1, 1))
    k_l_vec_list = np.swapaxes(tile, 0, 1) + l_vec_list
    y_hats = np.argmin(k_l_vec_list @ C, axis=2)

    # add prior to k-vectors
    prior = prior * np.ones(n_classes)
    k_vec_list = np.asarray(k_vec_list) + prior

    # all combination of k-, l-, and prediction indicator vectors
    combs = [k_vec_list, l_vec_list, np.eye(n_classes)]
    combs = np.asarray([list(elem)
                        for elem in list(itertools.product(*combs))])

    # three factors of the closed form solution
    factor_1 = 1 / euler_beta(k_vec_list)
    factor_2 = multinomial(l_vec_list)
    factor_3 = euler_beta(np.sum(combs, axis=1)).reshape(n_samples, n_l_vecs,
                                                         n_classes)

    # expected classification cost for each m
    m_sums = np.asarray(
        [factor_1[k_idx]
         * np.bincount(m_list, factor_2 * [C[:, y_hats[k_idx, l_idx]]
                                           @ factor_3[k_idx, l_idx]
                                           for l_idx in range(n_l_vecs)])
         for k_idx in range(n_samples)]
    )

    # compute classification cost reduction as difference
    gains = np.zeros((n_samples, m_max)) + m_sums[:, 0].reshape(-1, 1)
    gains -= m_sums[:, 1:]

    # normalize  cost reduction by number of hypothetical label acquisitions
    gains /= np.arange(1, m_max + 1)

    return np.max(gains, axis=1)


def _gen_l_vec_list(m_approx, n_classes):
    """
    Creates all possible class labeling vectors for given number of
    hypothetically acquired labels and given number of classes.

    Parameters
    ----------
    m_approx: int
        Number of hypothetically acquired labels..
    n_classes: int,
        Number of classes

    Returns
    -------
    label_vec_list: array-like, shape = [n_labelings, n_classes]
        All possible class labelings for given parameters.
    """

    label_vec_list = [[]]
    label_vec_res = np.arange(m_approx + 1)
    for i in range(n_classes - 1):
        new_label_vec_list = []
        for labelVec in label_vec_list:
            for newLabel in label_vec_res[label_vec_res
                                          - (m_approx - sum(labelVec))
                                          <= 1.e-10]:
                new_label_vec_list.append(labelVec + [newLabel])
        label_vec_list = new_label_vec_list

    new_label_vec_list = []
    for labelVec in label_vec_list:
        new_label_vec_list.append(labelVec + [m_approx - sum(labelVec)])
    label_vec_list = np.array(new_label_vec_list, int)

    return label_vec_list


def euler_beta(a):
    """
    Represents Euler beta function:
    B(a(i)) = Gamma(a(i,1))*...*Gamma(a_n)/Gamma(a(i,1)+...+a(i,n))

    Parameters
    ----------
    a: array-like, shape (m, n)
        Vectors to evaluated.

    Returns
    -------
    result: array-like, shape (m)
        Euler beta function results [B(a(0)), ..., B(a(m))
    """
    return np.exp(np.sum(gammaln(a), axis=1) - gammaln(np.sum(a, axis=1)))


def multinomial(a):
    """
    Computes Multinomial coefficient:
    Mult(a(i)) = (a(i,1)+...+a(i,n))!/(a(i,1)!...a(i,n)!)

    Parameters
    ----------
    a: array-like, shape (m, n)
        Vectors to evaluated.

    Returns
    -------
    result: array-like, shape (m)
        Multinomial coefficients [Mult(a(0)), ..., Mult(a(m))
    """
    return factorial(np.sum(a, axis=1)) / np.prod(factorial(a), axis=1)
