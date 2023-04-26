# -*- coding: utf-8 -*-
"""Automated tests based on the skbase test suite template."""
from skbase.testing import (
    BaseFixtureGenerator,
    QuickTester,
    TestAllObjects as _TestAllObjects
)

from ..base import BaseRegressor


class PackageConfig:
    """Contains package config variables for test classes."""

    # class variables which can be overridden by descendants
    # ------------------------------------------------------

    # package to search for objects
    # expected type: str, package/module name, relative to python environment root
    package_name = "pydata_skbase.mini_sklearn_v5"

    # list of object types (class names) to exclude
    # expected type: list of str, str are class names
    exclude_objects = "ClassName"  # exclude classes from extension templates

    # list of valid tags
    # expected type: list of str, str are tag names
    valid_tags = ["estimator_type", "regressor_type", "transformer_type"]


class TestAllObjects(PackageConfig, _TestAllObjects):
    """Generic tests for all objects in the mini package."""


class TestAllRegressors(PackageConfig, BaseFixtureGenerator, QuickTester):
    """Generic tests for all regressors in the mini package."""

    # class variables which can be overridden by descendants
    # ------------------------------------------------------

    # which object types are generated; None=all, or class (passed to all_objects)
    object_type_filter = BaseRegressor

    def test_input_output_contract(self, object_instance):
        """Tests that y output is pd.DataFrame and has same columns as y."""
        import pandas as pd

        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split

        X, y = load_diabetes(return_X_y=True, as_frame=True)
        y = pd.DataFrame(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        object_instance.fit(X_train, y_train)
        y_pred = object_instance.predict(X_test)

        assert isinstance(y_pred, pd.DataFrame)
        assert (y_pred.index == X_test.index).all()
        assert (y_pred.columns == y_train.columns).all()
