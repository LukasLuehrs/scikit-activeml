import itertools
import unittest

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import (
    GaussianProcessRegressor,
)
from sklearn.linear_model import LinearRegression

from skactiveml.pool.regression.utils import (
    conditional_expect,
    reshape_dist,
)
from skactiveml.regressor import (
    SklearnProbabilisticRegressor,
    NICKernelRegressor,
    SklearnRegressor,
)


class TestApproximation(unittest.TestCase):
    def setUp(self):
        self.random_state = 0

    def test_conditional_expectation_params(self):
        def dummy_func_1(y):
            return np.zeros_like(y)

        def dummy_func_2(x, y):
            return np.zeros_like(y)

        def dummy_func_3(idx, x, y):
            return np.zeros_like(y)

        X = np.arange(4 * 2).reshape(4, 2)
        y = np.arange(4, dtype=float)
        reg = NICKernelRegressor().fit(X, y)

        illegal_argument_dict = {
            "X": ["illegal", np.arange(3)],
            "func": ["illegal", dummy_func_2],
            "reg": [
                "illegal",
                SklearnRegressor(LinearRegression()),
            ],
            "method": ["illegal", 7, dict],
            "quantile_method": ["illegal", 7, dict],
            "n_integration_samples": ["illegal", 0, dict],
            "quad_dict": ["illegal", 7, dict],
            "random_state": ["illegal", dict],
            "include_x": ["illegal", dict, 7],
            "include_idx": ["illegal", dict, 7],
            "vector_func": ["illegal", dict, 7],
        }

        for parameter in illegal_argument_dict:
            for illegal_argument in illegal_argument_dict[parameter]:
                param_dict = dict(
                    X=X,
                    func=dummy_func_1,
                    reg=reg,
                    method="quantile",
                )
                param_dict[parameter] = illegal_argument
                self.assertRaises(
                    (TypeError, ValueError), conditional_expect, **param_dict
                )

        param_dict = dict(
            X=X,
            func=dummy_func_1,
            reg=reg,
            method="quantile",
        )
        dummy_funcs = [
            dummy_func_1,
            dummy_func_2,
            dummy_func_3,
        ]
        for include_idx, include_x in itertools.product(
            [False, True], [False, True]
        ):
            n_free_parameters = include_x + include_idx
            correct_func = dummy_funcs[include_x + include_idx]

            param_dict["func"] = correct_func
            param_dict["include_x"] = include_x
            param_dict["include_idx"] = include_idx
            y_int = conditional_expect(**param_dict)
            np.testing.assert_array_equal(np.zeros_like(y), y_int)
            for i in range(1, 3):
                incorrect_func = dummy_funcs[(n_free_parameters + i) % 3]
                param_dict["func"] = incorrect_func
                self.assertRaises(
                    (TypeError, ValueError), conditional_expect, **param_dict
                )

    def test_conditional_expectation(self):

        reg = SklearnProbabilisticRegressor(
            estimator=GaussianProcessRegressor()
        )
        X_train = np.array([[0, 2, 3], [1, 3, 4], [2, 4, 5], [3, 6, 7]])
        y_train = np.array([-1, 2, 1, 4])
        reg.fit(X_train, y_train)

        parameters_1 = [
            {"method": "assume_linear"},
            {
                "method": "monte_carlo",
                "n_integration_samples": 2,
            },
            {
                "method": "monte_carlo",
                "n_integration_samples": 4,
            },
            {
                "method": None,
                "quantile_method": "trapezoid",
            },
            {
                "method": "quantile",
                "quantile_method": "simpson",
            },
            {
                "method": "quantile",
                "quantile_method": "romberg",
            },
            {
                "method": "quantile",
                "quantile_method": "trapezoid",
            },
            {
                "method": "quantile",
                "quantile_method": "average",
            },
            {
                "method": "quantile",
                "quantile_method": "quadrature",
            },
            {"method": "dynamic_quad"},
            {"method": "gauss_hermite"},
        ]

        parameters_2 = [
            {"vector_func": True},
            {"vector_func": False},
            {"vector_func": "both"},
        ]

        X = np.arange(2 * 3).reshape((2, 3))

        for parameter_1, parameter_2 in itertools.product(
            parameters_1, parameters_2
        ):
            parameter = {**parameter_1, **parameter_2}

            def dummy_func(idx, x, y):
                if parameter["vector_func"] == "both":
                    if parameter["method"] == "dynamic_quad":
                        self.assertTrue(isinstance(idx, int))
                        self.assertTrue(isinstance(y, float))
                        self.assertEqual(x.shape, (3,))
                        return 0
                    else:
                        self.assertEqual(y.ndim, 2)
                        self.assertEqual(len(y), 2)
                        self.assertEqual(idx.dtype, int)
                        np.testing.assert_array_equal(x, X)
                        np.testing.assert_array_equal(idx, np.arange(2))
                        return np.zeros_like(y)
                elif parameter["vector_func"]:
                    self.assertEqual(y.ndim, 2)
                    self.assertEqual(len(y), 2)
                    self.assertEqual(idx.dtype, int)
                    np.testing.assert_array_equal(x, X)
                    np.testing.assert_array_equal(idx, np.arange(2))
                    return np.zeros_like(y)
                else:
                    self.assertTrue(isinstance(idx, int))
                    self.assertTrue(isinstance(y, float))
                    self.assertEqual(x.shape, (3,))
                    return 0

            res = conditional_expect(
                X=X,
                func=dummy_func,
                reg=reg,
                include_x=True,
                include_idx=True,
                **parameter
            )

            np.testing.assert_array_equal(res, np.zeros(2))

    def test_reshape_distribution(self):
        dist = norm(loc=np.array([0, 0]))
        reshape_dist(dist, shape=(2, 1))
        self.assertEqual(dist.kwds["loc"].shape, (2, 1))
        self.assertRaises(TypeError, reshape_dist, dist, "illegal")
        self.assertRaises(TypeError, reshape_dist, "illegal", (2, 1))
