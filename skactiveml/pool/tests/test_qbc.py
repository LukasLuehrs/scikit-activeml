import numpy as np
import unittest

from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import BaggingClassifier

from skactiveml.classifier import PWC
from skactiveml.utils import MISSING_LABEL
from skactiveml.pool._qbc import QBC, average_kl_divergence, vote_entropy


class TestQBC(unittest.TestCase):

    def setUp(self):
        self.random_state = 41
        self.X_cand = np.array([[8, 1, 6, 8], [9, 1, 6, 5], [5, 1, 6, 5]])
        self.X = np.array(
            [[1, 2, 5, 9], [5, 8, 4, 6], [8, 4, 5, 9], [5, 4, 8, 5]])
        self.y = np.array([0, 0, 1, 1])
        self.classes = np.array([0, 1])
        self.clf = PWC(classes=self.classes, random_state=self.random_state)
        self.kwargs = dict(X_cand=self.X_cand, X=self.X, y=self.y)

    def test_init_param_clf(self):
        selector = QBC(clf=PWC(), random_state=self.random_state)
        selector.query(**self.kwargs)
        self.assertTrue(hasattr(selector, 'clf'))
        # selector = QBC(clf=GaussianProcessClassifier(
        #    random_state=self.random_state), random_state=self.random_state)
        selector.query(**self.kwargs)
        selector = QBC(clf='string')
        self.assertRaises(TypeError, selector.query, **self.kwargs)
        selector = QBC(clf=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs)
        selector = QBC(clf=1)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

    def test_init_param_ensemble(self):
        selector = QBC(clf=self.clf, ensemble=None)
        self.assertTrue(hasattr(selector, 'ensemble'))
        selector.query(**self.kwargs)
        self.assertTrue(isinstance(selector._clf.estimator, BaggingClassifier))

        selector = QBC(clf=self.clf, ensemble='String')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = QBC(clf=self.clf, ensemble=RandomForestClassifier,
                       n_estimators=5)
        selector.query(**self.kwargs)

    def test_init_param_method(self):
        selector = QBC(clf=self.clf)
        self.assertTrue(hasattr(selector, 'method'))
        selector = QBC(clf=self.clf, method='String')
        self.assertRaises(ValueError, selector.query, **self.kwargs)
        selector = QBC(clf=self.clf, method=1)
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = QBC(clf=self.clf, method='KL_divergence')
        selector.query(**self.kwargs)
        selector = QBC(clf=GaussianProcessRegressor, method='KL_divergence')
        self.assertRaises(TypeError, selector.query, **self.kwargs)

        selector = QBC(clf=self.clf, method='vote_entropy')
        selector.query(**self.kwargs)

    def test_init_param_classes(self):
        selector = QBC(clf=PWC())
        self.assertTrue(hasattr(selector, 'missing_label'))
        selector.query(X_cand=[[1]], X=[[1]], y=[0])
        # self.assertRaises(ValueError, selector.query, X_cand=[[1]], X=[[1]],
        #                  y=[np.nan])
        # selector = QBC(clf=PWC, classes=self.classes)
        # selector.query(X_cand=[[1]], X=[[1]], y=[2])
        # selector.query(X_cand=[[1]], X=[[1]], y=[0])
        # selector.query(X_cand=[[1]], X=[[1]], y=[np.nan])

    def test_init_param_missing_label(self):
        selector = QBC(clf=self.clf, missing_label='string')
        self.assertTrue(hasattr(selector, 'missing_label'))
        self.assertRaises(ValueError, selector.query, X_cand=[[1]], X=[[1]],
                          y=[np.nan])
        selector = QBC(clf=self.clf, classes=self.classes)
        selector.query(X_cand=[[1]], X=[[1]], y=[MISSING_LABEL])

    def test_init_param_random_state(self):
        qbc = QBC(clf=self.clf, random_state='string')
        self.assertRaises(ValueError, qbc.query, self.X_cand, self.X, self.y)
        selector = QBC(clf=self.clf, random_state=self.random_state)
        self.assertTrue(hasattr(selector, 'random_state'))
        self.assertRaises(ValueError, selector.query, X_cand=[[1]], X=self.X,
                          y=self.y)

    def test_query_param_X_cand(self):
        selector = QBC(clf=self.clf)
        self.assertRaises(ValueError, selector.query, X_cand=[], X=self.X,
                          y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=None, X=self.X,
                          y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=np.nan, X=self.X,
                          y=self.y)

    def test_query_param_X(self):
        selector = QBC(clf=self.clf)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=None, y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X='string', y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=[], y=self.y)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X[0:-1], y=self.y)

    def test_query_param_y(self):
        selector = QBC(clf=self.clf)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=None)
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y='string')
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=[])
        self.assertRaises(ValueError, selector.query, X_cand=self.X_cand,
                          X=self.X, y=self.y[0:-1])

        selector.query(X_cand=[[1]], X=[[1]], y=[0])
        selector.query(X_cand=[[1]], X=[[1]], y=[MISSING_LABEL])

    def test_query_param_batch_size(self):
        selector = QBC(clf=self.clf)
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          batch_size=1.2)
        self.assertRaises(ValueError, selector.query, **self.kwargs,
                          batch_size=0)

    def test_query_param_return_utilities(self):
        selector = QBC(clf=self.clf)
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          return_utilities=None)
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          return_utilities=[])
        self.assertRaises(TypeError, selector.query, **self.kwargs,
                          return_utilities=0)

        L = list(selector.query(**self.kwargs, return_utilities=True))
        self.assertTrue(len(L) == 2)
        L = list(selector.query(**self.kwargs, return_utilities=False))
        self.assertTrue(len(L) == 1)

    def test_query(self):
        selector = QBC(clf=self.clf, random_state=self.random_state)
        best_indices1, utilities1 = selector.query(**self.kwargs,
                                                   return_utilities=True)
        selector = QBC(clf=self.clf, random_state=self.random_state)
        best_indices2, utilities2 = selector.query(**self.kwargs,
                                                   return_utilities=True)
        np.testing.assert_array_equal(utilities1, utilities2)
        np.testing.assert_array_equal(best_indices1, best_indices2)


class TestAverageKlDivergence(unittest.TestCase):
    def setUp(self):
        self.probas = np.array([[[0.3, 0.7], [0.4, 0.6]],
                                [[0.2, 0.8], [0.5, 0.5]]])
        self.scores = np.array([0.00670178182226764, 0.005059389928987596])

    def test_param_probas(self):
        self.assertRaises(ValueError, average_kl_divergence, 'string')
        self.assertRaises(ValueError, average_kl_divergence, 1)
        self.assertRaises(ValueError, average_kl_divergence,
                          np.ones((1,)))
        self.assertRaises(ValueError, average_kl_divergence,
                          np.ones((1, 1)))
        self.assertRaises(ValueError, average_kl_divergence,
                          np.ones((1, 1, 1, 1)))

    def test_average_kl_divergence(self):
        scores = average_kl_divergence(self.probas)
        np.testing.assert_array_equal(scores, self.scores)


class TestVoteEntropy(unittest.TestCase):
    def setUp(self):
        self.classes = np.array([0, 1, 2])
        self.votes = np.array([[0, 0, 2],
                               [1, 0, 2],
                               [2, 1, 2]])
        self.scores = np.array([1, 0.5793801643, 0])

    def test_param_votes(self):
        self.assertRaises(ValueError, vote_entropy, votes='string',
                          classes=self.classes)
        self.assertRaises(ValueError, vote_entropy, votes=1,
                          classes=self.classes)
        self.assertRaises(ValueError, vote_entropy, votes=[1],
                          classes=self.classes)
        self.assertRaises(ValueError, vote_entropy, votes=[[[1]]],
                          classes=self.classes)

    def test_param_classes(self):
        vote_entropy(votes=self.votes, classes='string')
        self.assertRaises(ValueError, vote_entropy, votes=self.votes,
                          classes='class')
        self.assertRaises(TypeError, vote_entropy, votes=self.votes,
                          classes=1)
        self.assertRaises(TypeError, vote_entropy, votes=self.votes,
                          classes=[[1]])
#        self.assertRaises(ValueError, vote_entropy, votes=self.votes,
#                          classes=[MISSING_LABEL, 1])

    def test_vote_entropy(self):
        scores = vote_entropy(votes=self.votes, classes=self.classes)
        np.testing.assert_array_equal(scores.round(10), self.scores.round(10))


if __name__ == '__main__':
    unittest.main()
