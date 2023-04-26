# -*- coding: utf-8 -*-
"""Base class and template for regressors and transformers."""
from .base import BaseTransformer


class Scaler(BaseTransformer):
    """Scale data by minmax or standard normalization strategy.

    Parameters
    ----------
    strategy : string, one of "minmax" or "std"
        "std" = each column of X is subtracted mean, divided standard deviation
        "minmax" = each column is linearly scaled to minimum = 0, maximum = 1

    Attributes
    ----------
    X_min_ : pd.Series, index = index of X in fit
        column-wise minimum of X in fit
    X_max_ : pd.Series, index = index of X in fit
        column-wise maximum of X in fit
    X_span_ : pd.Series, index = index of X in fit
        column-wise maximum minus minimum, of X in fit
    X_mean_ : pd.Series, index = index of X in fit
        column-wise mean, of X in fit
    X_std_ : pd.Series, index = index of X in fit
        column-wise standard deviation, of X in fit
    """

    def __init__(self, strategy="std"):
        self.strategy = strategy

        super(Scaler, self).__init__()

        # this is fine here because of how set_params works!
        if strategy not in ["std", "minmax"]:
            raise ValueError(
                "in Scaler, strategy must be one of the strings "
                f'"std" or "minmax", but found {strategy}'
            )

    def fit(self, X):
        """Fit transformer to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Changes state to "fitted" = sets is_fitted flag to True

        Parameters
        ----------
        X : pandas DataFrame
            data to fit transformer to

        Returns
        -------
        self : reference to self
        """
        # store columns in self for later check with transform
        self._X_columns = X.columns

        # min, max, max-min as fitted params, for minmax
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_span = X_max - X_min

        # mean, std as fitted params, for std
        X_std = X.std(axis=0)
        X_mean = X.mean(axis=0)

        self.X_min_ = X_min
        self.X_max_ = X_max
        self.X_span_ = X_span
        self.X_std_ = X_std
        self.X_mean_ = X_mean

        # this should be the underscore tag
        # the skbase BaseEstimator handles is_fitted in dependency of this
        self._is_fitted = True

        return self

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
        if not (X.columns == self._X_columns).all():
            raise ValueError(
                "X in transform must have same columns as X in fit, "
                f"columns in fit were {self._X_columns}, "
                f"but in transform found X.columns = {X.columns}"
            )

        if not self.is_fitted:
            raise RuntimeError(
                f"error in estimator {self}, attempt to call transform, "
                "but fit has not been called yet"
            )

        strategy = self.strategy

        if strategy == "minmax":
            X_min = self.X_min_
            X_span = self.X_span_

            Xt = (X - X_min) / X_span

        elif strategy == "std":
            X_mean = self.X_mean_
            X_std = self.X_std_

            Xt = (X - X_mean) / X_std

        return Xt
