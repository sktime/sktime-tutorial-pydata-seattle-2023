# -*- coding: utf-8 -*-
"""Base class and template for regressors and transformers."""
from skbase.base import BaseEstimator


class BaseRegressor(BaseEstimator):
    """Base class for supervised regressors."""

    def fit(self, X, y):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to

        Returns
        -------
        self : reference to self
        """
        raise NotImplementedError

    def predict(self, X):
        """Predict labels for data from features.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y : pandas DataFrame, same length as `X`
            labels predicted for `X`
        """
        raise NotImplementedError


class BaseTransformer(BaseEstimator):
    """Base class for transformers."""

    def fit(self, X):
        """Fit transformer to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            data to fit transformer to

        Returns
        -------
        self : reference to self
        """
        raise NotImplementedError

    def transform(self, X):
        """Transform data with fitted transformer.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to transform

        Returns
        -------
        Xt : pandas DataFrame, same length as `X`
            transformed version of `X`
        """
        raise NotImplementedError
